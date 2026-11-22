"""Reusable optimization and precision parity gates for model tests."""

import contextlib
import os
import statistics
from time import perf_counter

import e3nn
import pytest
import torch

from nequip.data import AtomicData, AtomicDataDict
from nequip.utils import _global_options
from nequip.utils.compile_utils import (
    AUTOCAST_FP32_EXPERIMENT_TIERS,
    DEFAULT_AUTOCAST_CONFIG,
    configure_model_for_autocast,
    configure_model_for_compile,
    get_autocast_stats,
    get_compile_stats,
    training_autocast_context,
)
from nequip.utils.test import set_irreps_debug
from nequip.utils.unittests.model_tests import BaseModelTests


# Calibrated against H100 jobs job-58e06d60 and job-cf1ed1e6. The latter ran
# with A4's diagnostics and printed the TF32 energy values the former hid:
# 2.227e-4 and 2.264e-4 eV/atom on Allegro NoSOC/SOC, above the provisional
# 1e-4 gate. 5e-4 gives ~2.2x headroom over the measured worst case.
PRECISION_TOLERANCES = {
    "fp32": {"force_rel": 1.0e-5, "energy_per_atom_ev": 1.0e-5},
    "tf32": {"force_rel": 8.0e-3, "energy_per_atom_ev": 5.0e-4},
    # Recording-only sanity bounds. H100 force error was 7.755e-2, and A4's
    # best CPU whitelist still reached 1.7803e-2. The production force gate
    # remains 1e-2: bf16 is not accepted for force-supervised use.
    "bf16": {"force_rel": 1.0e-1, "energy_per_atom_ev": 1.0e-2},
}

BF16_FORCE_SUPERVISION_REL_LIMIT = 1.0e-2

_PHYSICAL_OUTPUTS = (
    AtomicDataDict.TOTAL_ENERGY_KEY,
    AtomicDataDict.PER_ATOM_ENERGY_KEY,
    AtomicDataDict.FORCE_KEY,
    AtomicDataDict.SPIN_FORCE_KEY,
    AtomicDataDict.SPIN_FORCE_VERT_KEY,
    AtomicDataDict.STRESS_KEY,
    AtomicDataDict.VIRIAL_KEY,
    AtomicDataDict.ATOM_VIRIAL_KEY,
)

_FULL_FX_CURVE_RAN = False


def _assert_bf16_force_supervision(errors):
    force_error = errors[AtomicDataDict.FORCE_KEY]
    assert force_error <= BF16_FORCE_SUPERVISION_REL_LIMIT, (
        "forces bf16 production relative error "
        f"{force_error:.3e} > {BF16_FORCE_SUPERVISION_REL_LIMIT:.3e}"
    )


def _require_compile_gate():
    if os.getenv("SPINGNN_TEST_COMPILE") != "1":
        pytest.skip("compile gate disabled")


def _require_autocast_gate():
    if os.getenv("SPINGNN_TEST_AUTOCAST") != "1":
        pytest.skip("autocast gate disabled")


def _clone_data(data):
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }


def _cast_data(data, dtype, device):
    cast = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device=device)
            if value.is_floating_point():
                value = value.to(dtype=dtype)
        cast[key] = value
    return cast


def _relative_error(reference, candidate):
    reference = reference.detach().to(dtype=torch.float64, device="cpu")
    candidate = candidate.detach().to(dtype=torch.float64, device="cpu")
    denominator = torch.linalg.vector_norm(reference).clamp_min(1.0e-14)
    return float(torch.linalg.vector_norm(candidate - reference) / denominator)


@contextlib.contextmanager
def _default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@contextlib.contextmanager
def _e3nn_jit_script_fx(enabled):
    previous = e3nn.get_optimization_defaults()["jit_script_fx"]
    e3nn.set_optimization_defaults(jit_script_fx=enabled)
    try:
        yield
    finally:
        e3nn.set_optimization_defaults(jit_script_fx=previous)


@contextlib.contextmanager
def _compile_without_global_debug_hooks():
    set_irreps_debug(False)
    try:
        yield
    finally:
        set_irreps_debug(True)


class _BaseParityTests:
    @staticmethod
    def _build_model(config, device, dtype):
        config, _ = config
        with _default_dtype(dtype):
            return BaseModelTests.make_model(config, device=device)

    @staticmethod
    def _batch_dict(atomic_batch_spin, device):
        # The second fixture frame is H2. It still exercises E/F/spin-force/
        # stress and second-order gradients without making the opt-in gate
        # needlessly scale with the larger first molecule.
        data = atomic_batch_spin.get_example(1).clone().to(device)
        return AtomicDataDict.with_batch(AtomicData.to_AtomicDataDict(data))

    @staticmethod
    def _training_batch_dict(atomic_batch_spin, device):
        data = AtomicData.to_AtomicDataDict(atomic_batch_spin.clone().to(device))
        data = {
            key: value
            for key, value in data.items()
            if key not in AtomicDataDict.ALL_ENERGY_KEYS
        }
        batch = data[AtomicDataDict.BATCH_KEY]
        data[AtomicDataDict.NUM_BATCH_KEY] = torch.tensor(
            int(batch.max().item()) + 1,
            dtype=torch.long,
            device=device,
        )
        return data

    @staticmethod
    def _compare_outputs(reference, candidate, rel_limit):
        fields = _PHYSICAL_OUTPUTS + (AtomicDataDict.SPIN_LENGTH_KEY,)
        reference_keys = set(reference).intersection(fields)
        candidate_keys = set(candidate).intersection(fields)
        assert (
            reference_keys == candidate_keys
        ), f"output presence differs: {reference_keys ^ candidate_keys}"
        compared = []
        for key in fields:
            if key in reference and key in candidate:
                assert reference[key].shape == candidate[key].shape, key
                error = _relative_error(reference[key], candidate[key])
                assert error <= rel_limit, f"{key} relative error {error:.3e}"
                compared.append(key)
        assert AtomicDataDict.TOTAL_ENERGY_KEY in compared
        assert AtomicDataDict.FORCE_KEY in compared

    @staticmethod
    def _force_training_step(model, data):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-4)
        optimizer.zero_grad(set_to_none=True)
        output = model(_clone_data(data))
        loss = output[AtomicDataDict.FORCE_KEY].square().mean()
        for key in (
            AtomicDataDict.SPIN_FORCE_KEY,
            AtomicDataDict.SPIN_FORCE_VERT_KEY,
        ):
            if key in output:
                loss = loss + output[key].square().mean()
        loss.backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if name.startswith("_orig_mod."):
                    name = name[len("_orig_mod.") :]
                gradients[name] = parameter.grad.detach().clone()
        optimizer.step()
        return gradients

    @staticmethod
    def _loss_training_step(model, optimizer, data):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        device = data[AtomicDataDict.POSITIONS_KEY].device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = perf_counter()
        output = model(_clone_data(data))
        loss = output[AtomicDataDict.FORCE_KEY].square().mean()
        loss = loss + output[AtomicDataDict.SPIN_FORCE_VERT_KEY].square().mean()
        loss = loss + output[AtomicDataDict.STRESS_KEY].square().mean()
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - started
        return float(loss.detach()), elapsed

    @staticmethod
    def _compare_gradients(reference, candidate, rel_limit):
        # None skips an optimizer update; zero still participates in weight
        # decay and momentum. Their parameter sets must therefore agree.
        assert (
            reference.keys() == candidate.keys()
        ), f"gradient presence differs: {reference.keys() ^ candidate.keys()}"
        assert reference
        for key in reference:
            error = _relative_error(reference[key], candidate[key])
            assert error <= rel_limit, f"gradient {key} relative error {error:.3e}"


@pytest.mark.compile
class BaseCompileParityTests(_BaseParityTests):
    def test_eager_pipeline_selfcheck(self, config, atomic_batch_spin):
        # Global ScriptFunctions remain scripted with jit_script_fx=False.
        # Their cold profiling stages can change derivative rounding between
        # calls. Hold execution fixed for this bitwise control, then restore it.
        with torch.jit.optimized_execution(False):
            _require_compile_gate()
            if torch.get_default_dtype() is not torch.float32:
                pytest.skip("compile gate self-check runs once in fp32")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Match the compile gate's eager model construction.
            with _e3nn_jit_script_fx(False):
                reference = self._build_model(config, device, torch.float32)
                candidate = self._build_model(config, device, torch.float32)
            candidate.load_state_dict(reference.state_dict())
            data = self._batch_dict(atomic_batch_spin, device)

            reference.eval()
            candidate.eval()
            reference_output = reference(_clone_data(data))
            candidate_output = candidate(_clone_data(data))
            for key in _PHYSICAL_OUTPUTS:
                if key in reference_output and key in candidate_output:
                    assert torch.equal(
                        reference_output[key], candidate_output[key]
                    ), key

            reference_grad = self._force_training_step(reference, data)
            candidate_grad = self._force_training_step(candidate, data)
            self._compare_gradients(reference_grad, candidate_grad, rel_limit=1.0e-4)

    def test_compile_parity(self, config, atomic_batch_spin):
        _require_compile_gate()
        if torch.get_default_dtype() is not torch.float32:
            pytest.skip("compile gate parity runs once in fp32")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with _e3nn_jit_script_fx(False):
            reference = self._build_model(config, device, torch.float32)
            candidate = self._build_model(config, device, torch.float32)
            candidate.load_state_dict(reference.state_dict())
            state_keys = tuple(candidate.state_dict())
            configure_model_for_compile(
                candidate,
                {
                    "spingnn_compile": {
                        "enabled": True,
                        "scope": "full_fx",
                        "dynamic": True,
                        "inductor_options": {},
                    }
                },
            )
        assert tuple(candidate.state_dict()) == state_keys
        data = self._training_batch_dict(atomic_batch_spin, device)

        reference.eval()
        candidate.eval()
        reference_output = reference(_clone_data(data))
        with _compile_without_global_debug_hooks():
            compiled_output = candidate(_clone_data(data))

        self._compare_outputs(reference_output, compiled_output, rel_limit=1.0e-5)
        reference_grad = self._force_training_step(reference, data)
        with _compile_without_global_debug_hooks():
            candidate_grad = self._force_training_step(candidate, data)
        self._compare_gradients(reference_grad, candidate_grad, rel_limit=1.0e-4)
        stats = get_compile_stats(candidate)
        assert stats["targets"] == 1
        assert stats["scopes"] == ["full_fx"]
        assert stats["compile_count"] == 2
        assert stats["full_fx_expand_count"] == 2
        assert stats["cache_modes"] == ["eval", "train"]
        assert stats["cache_entries"] == 2
        assert stats["eager_grad_bridge_calls"] == 0
        for sizes in stats["trace_sizes"].values():
            assert len(set(sizes)) == 3
            assert all(
                size >= 5
                and all(size % divisor for divisor in range(2, int(size**0.5) + 1))
                for size in sizes
            )

    def test_full_fx_training_curve(self, config, atomic_batch_spin):
        _require_compile_gate()
        if torch.get_default_dtype() is not torch.float32:
            pytest.skip("full-FX training curve runs once in fp32")
        global _FULL_FX_CURVE_RAN
        if _FULL_FX_CURVE_RAN:
            pytest.skip("full-FX 20-step curve already covered in this suite")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with _e3nn_jit_script_fx(False):
            reference = self._build_model(config, device, torch.float32)
            candidate = self._build_model(config, device, torch.float32)
            candidate.load_state_dict(reference.state_dict())
            configure_model_for_compile(
                candidate,
                {
                    "spingnn_compile": {
                        "enabled": True,
                        "scope": "full_fx",
                        "dynamic": True,
                        "inductor_options": {},
                    }
                },
            )
        data = self._training_batch_dict(atomic_batch_spin, device)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1.0e-5)
        candidate_optimizer = torch.optim.SGD(candidate.parameters(), lr=1.0e-5)
        eager_times = []
        full_fx_times = []
        loss_relatives = []
        with _compile_without_global_debug_hooks():
            for step in range(20):
                eager_loss, eager_time = self._loss_training_step(
                    reference, reference_optimizer, data
                )
                full_fx_loss, full_fx_time = self._loss_training_step(
                    candidate, candidate_optimizer, data
                )
                relative = abs(full_fx_loss - eager_loss) / max(
                    abs(eager_loss), 1.0e-14
                )
                assert (
                    relative <= 1.0e-4
                ), f"step {step} loss relative error {relative:.3e}"
                loss_relatives.append(relative)
                eager_times.append(eager_time)
                full_fx_times.append(full_fx_time)
        stats = get_compile_stats(candidate)
        assert stats["eager_grad_bridge_calls"] == 0
        assert stats["full_fx_expand_count"] == 1
        assert stats["cache_entries"] == 1
        print(
            "\nFULL_FX_WALLTIME "
            f"eager_seconds={statistics.median(eager_times[1:]):.9f} "
            f"full_fx_seconds={statistics.median(full_fx_times[1:]):.9f} "
            f"max_loss_relative={max(loss_relatives):.9e} "
            f"cache_entries={stats['cache_entries']} "
            "measurement=steady_state_median steps=19"
        )
        _FULL_FX_CURVE_RAN = True


@pytest.mark.compile
class BasePrecisionLadderTests(_BaseParityTests):
    def test_bf16_cpu_autocast(self, config, atomic_batch_spin):
        _require_autocast_gate()
        if torch.get_default_dtype() is not torch.float32:
            pytest.skip("autocast CPU case runs once in fp32")

        device = torch.device("cpu")
        model = self._build_model(config, device, torch.float32)
        state_keys = tuple(model.state_dict())
        autocast_config = {
            "train_autocast": {
                "enabled": True,
                "dtype": "bfloat16",
                "fp32_modules": list(DEFAULT_AUTOCAST_CONFIG["fp32_modules"]),
            }
        }
        configure_model_for_autocast(model, autocast_config, device_type="cpu")
        assert tuple(model.state_dict()) == state_keys
        data = self._batch_dict(atomic_batch_spin, device)

        model.eval()
        with training_autocast_context(autocast_config, "cpu"):
            output = model(_clone_data(data))

        for key in _PHYSICAL_OUTPUTS:
            if key in output and output[key].is_floating_point():
                assert output[key].dtype is torch.float32, key
        stats = get_autocast_stats(model)
        assert any(item["class_name"] == "ParaStressForceSpinOutput" for item in stats)
        for item in stats:
            assert item["calls"] > 0, item
            assert item["input_dtypes"] == ("torch.float32",), item
            assert item["output_dtypes"] == ("torch.float32",), item

    @staticmethod
    def _precision_errors(reference_output, candidate_output, reference_data):
        batch = reference_data[AtomicDataDict.BATCH_KEY]
        counts = torch.bincount(batch).to(dtype=torch.float64, device="cpu")
        energy_error = (
            reference_output[AtomicDataDict.TOTAL_ENERGY_KEY]
            .detach()
            .to(dtype=torch.float64, device="cpu")
            - candidate_output[AtomicDataDict.TOTAL_ENERGY_KEY]
            .detach()
            .to(dtype=torch.float64, device="cpu")
        ).abs() / counts.reshape(
            -1,
            *([1] * (reference_output[AtomicDataDict.TOTAL_ENERGY_KEY].ndim - 1)),
        )
        errors = {"energy_per_atom_ev": float(energy_error.max())}
        for key in (
            AtomicDataDict.FORCE_KEY,
            AtomicDataDict.SPIN_FORCE_KEY,
            AtomicDataDict.SPIN_FORCE_VERT_KEY,
            AtomicDataDict.STRESS_KEY,
        ):
            if key in reference_output and key in candidate_output:
                errors[key] = _relative_error(
                    reference_output[key], candidate_output[key]
                )
                errors[f"{key}_absolute"] = float(
                    (
                        reference_output[key]
                        .detach()
                        .to(dtype=torch.float64, device="cpu")
                        - candidate_output[key]
                        .detach()
                        .to(dtype=torch.float64, device="cpu")
                    )
                    .abs()
                    .max()
                )
        return errors

    @staticmethod
    def _assert_precision_errors(errors, tolerance, tolerance_name):
        energy_error = errors["energy_per_atom_ev"]
        energy_limit = tolerance["energy_per_atom_ev"]
        assert energy_error <= energy_limit, (
            f"total_energy {tolerance_name} absolute error "
            f"{energy_error:.3e} eV/atom > {energy_limit:.3e} eV/atom"
        )
        force_limit = tolerance["force_rel"]
        for key in (
            AtomicDataDict.FORCE_KEY,
            AtomicDataDict.SPIN_FORCE_KEY,
            AtomicDataDict.SPIN_FORCE_VERT_KEY,
            AtomicDataDict.STRESS_KEY,
        ):
            if key not in errors:
                continue
            relative_error = errors[key]
            absolute_error = errors[f"{key}_absolute"]
            if key in (
                AtomicDataDict.SPIN_FORCE_KEY,
                AtomicDataDict.SPIN_FORCE_VERT_KEY,
            ):
                assert (
                    relative_error <= force_limit or absolute_error <= energy_limit
                ), (
                    f"{key} {tolerance_name} relative error "
                    f"{relative_error:.3e} > {force_limit:.3e}; "
                    f"absolute error {absolute_error:.3e} "
                    f"> {energy_limit:.3e}"
                )
            else:
                assert relative_error <= force_limit, (
                    f"{key} {tolerance_name} relative error "
                    f"{relative_error:.3e} > {force_limit:.3e}; "
                    f"absolute error {absolute_error:.3e}"
                )

    @staticmethod
    def _precision_case_name(config):
        config, _ = config
        builders = "+".join(config.get("model_builders", []))
        details = [f"lmax={config.get('l_max', 'na')}"]
        for key in (
            "nonlinearity_type",
            "lte3nn",
            "lsoc_layer",
            "ldelta_learning",
        ):
            if key in config:
                details.append(f"{key}={config[key]}")
        return ":".join((builders, *details)).replace(" ", "")

    def test_bf16_cpu_whitelist_tiers(self, config, atomic_batch_spin):
        _require_autocast_gate()
        if torch.get_default_dtype() is not torch.float32:
            pytest.skip("bf16 whitelist experiment runs once in fp32")

        device = torch.device("cpu")
        reference = self._build_model(config, device, torch.float64)
        base_data = self._batch_dict(atomic_batch_spin, device)
        reference_data = _cast_data(base_data, torch.float64, device)
        reference.eval()
        reference_output = reference(reference_data)
        case_name = self._precision_case_name(config)

        for tier_name, patterns in AUTOCAST_FP32_EXPERIMENT_TIERS.items():
            candidate = self._build_model(config, device, torch.float32)
            candidate.load_state_dict(reference.state_dict())
            autocast_config = {
                "train_autocast": {
                    "enabled": True,
                    "dtype": "bfloat16",
                    "fp32_modules": list(patterns),
                }
            }
            configure_model_for_autocast(candidate, autocast_config, device_type="cpu")
            candidate_data = _cast_data(base_data, torch.float32, device)
            candidate.eval()
            with training_autocast_context(autocast_config, "cpu"):
                candidate_output = candidate(candidate_data)
            errors = self._precision_errors(
                reference_output, candidate_output, reference_data
            )
            assert all(torch.isfinite(torch.tensor(value)) for value in errors.values())
            self._assert_precision_errors(
                errors,
                PRECISION_TOLERANCES["bf16"],
                f"bf16_{tier_name}_recording",
            )
            stats = get_autocast_stats(candidate)
            force_error = errors[AtomicDataDict.FORCE_KEY]
            try:
                _assert_bf16_force_supervision(errors)
                force_supervision_pass = True
            except AssertionError:
                force_supervision_pass = False
            spin_error = errors.get(AtomicDataDict.SPIN_FORCE_KEY, float("nan"))
            spin_absolute = errors.get(
                f"{AtomicDataDict.SPIN_FORCE_KEY}_absolute", float("nan")
            )
            stress_error = errors.get(AtomicDataDict.STRESS_KEY, float("nan"))
            print(
                "BF16_WHITELIST_RESULT "
                f"case={case_name} tier={tier_name} "
                f"energy_abs_ev_atom={errors['energy_per_atom_ev']:.9e} "
                f"force_rel={force_error:.9e} "
                f"spin_force_rel={spin_error:.9e} "
                f"spin_force_abs={spin_absolute:.9e} "
                f"stress_rel={stress_error:.9e} "
                f"force_supervision_pass="
                f"{str(force_supervision_pass).lower()} "
                f"matched={len(stats)}"
            )

    def _run_precision_case(
        self, config, atomic_batch_spin, device, candidate_dtype, tolerance_name
    ):
        reference = self._build_model(config, device, torch.float64)
        candidate = self._build_model(config, device, candidate_dtype)
        candidate.load_state_dict(reference.state_dict())
        base_data = self._batch_dict(atomic_batch_spin, device)
        reference_data = _cast_data(base_data, torch.float64, device)
        candidate_data = _cast_data(base_data, candidate_dtype, device)

        reference.eval()
        candidate.eval()
        reference_output = reference(reference_data)
        candidate_output = candidate(candidate_data)
        # Delta fixtures must exercise an actual pair interaction rather
        # than passing relative-error gates with an identically zero head.
        if getattr(self, "require_precision_signal", False):
            for key in (AtomicDataDict.TOTAL_ENERGY_KEY, AtomicDataDict.FORCE_KEY):
                signal = float(torch.linalg.vector_norm(reference_output[key]).detach())
                assert (
                    signal > 1.0e-8
                ), f"degenerate precision sample: {key} norm {signal}"
                print(f"PRECISION_SIGNAL field={key} reference_norm={signal:.9e}")
        tolerance = PRECISION_TOLERANCES[tolerance_name]
        errors = self._precision_errors(
            reference_output, candidate_output, reference_data
        )
        case_name = self._precision_case_name(config)
        fields = [
            f"case={case_name}",
            f"tier={tolerance_name}",
            f"energy_abs_ev_atom={errors['energy_per_atom_ev']:.9e}",
        ]
        for key in (
            AtomicDataDict.FORCE_KEY,
            AtomicDataDict.SPIN_FORCE_KEY,
            AtomicDataDict.SPIN_FORCE_VERT_KEY,
            AtomicDataDict.STRESS_KEY,
        ):
            if key in errors:
                fields.append(f"{key}_rel={errors[key]:.9e}")
                fields.append(f"{key}_abs={errors[f'{key}_absolute']:.9e}")
        print("PRECISION_RESULT " + " ".join(fields))
        self._assert_precision_errors(errors, tolerance, tolerance_name)
        return errors

    def test_fp32_cpu_precision(self, config, atomic_batch_spin):
        _require_compile_gate()
        if torch.get_default_dtype() is not torch.float32:
            pytest.skip("precision ladder CPU case runs once in fp32")
        self._run_precision_case(
            config,
            atomic_batch_spin,
            torch.device("cpu"),
            torch.float32,
            "fp32",
        )

    @pytest.mark.gpu
    def test_tf32_gpu_precision(self, config, atomic_batch_spin):
        _require_compile_gate()
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        if torch.get_default_dtype() is not torch.float32:
            pytest.skip("precision ladder TF32 case runs once in fp32")
        # These legacy getters raise on a CUDA process that has mixed the legacy
        # and new TF32 APIs (see nequip.utils._global_options). Use the tolerant
        # reader so this test cannot die in its own save/restore preamble; a
        # per-backend value that could not be read is not restored, because
        # restoring a fabricated one would silently change global state for
        # later tests.
        previous_matmul, previous_cudnn = _global_options._read_legacy_allow_tf32()
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self._run_precision_case(
                config,
                atomic_batch_spin,
                torch.device("cuda"),
                torch.float32,
                "tf32",
            )
        finally:
            if previous_matmul is not None:
                torch.backends.cuda.matmul.allow_tf32 = previous_matmul
            if previous_cudnn is not None:
                torch.backends.cudnn.allow_tf32 = previous_cudnn

    @pytest.mark.gpu
    @pytest.mark.bf16
    def test_bf16_gpu_precision(self, config, atomic_batch_spin):
        """Measure and record bf16; bf16 is NOT a supported training precision.

        bf16 autocast was evaluated for force-supervised training and rejected:
        its force relative error does not meet the 1e-2 production bound
        RELIABLY. The 2026-08-31 H100 measurements ranged from 2.8e-3 to 4.9e-2
        across model configs -- some configs meet the bound, the worst are 3-5x
        over it, and force supervision needs every config to hold. This case is
        informational (`bf16` marker, deselected from the required lanes) and
        prints a BF16_MEASUREMENT record with a per-config
        ``meets_production_bound`` field for the log.

        An earlier revision asserted ``force_error > limit`` as an anti-rot
        tripwire; that encoded the false premise that NO config can meet the
        bound and failed immediately on the configs that do. Do not restore a
        per-config assertion in either direction: the abandonment premise is
        about reliability across configs, which a single-config test cannot
        judge. ``_run_precision_case`` still applies the bf16 recording bounds,
        so a catastrophic numerical blow-up fails regardless.
        """
        _require_compile_gate()
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        if torch.get_default_dtype() is not torch.float32:
            pytest.skip("precision ladder bf16 case runs once in fp32")
        errors = self._run_precision_case(
            config,
            atomic_batch_spin,
            torch.device("cuda"),
            torch.bfloat16,
            "bf16",
        )
        force_error = errors[AtomicDataDict.FORCE_KEY]
        print(
            "BF16_MEASUREMENT "
            f"force_rel={force_error:.6e} "
            f"energy_per_atom_ev={errors['energy_per_atom_ev']:.6e} "
            f"production_limit={BF16_FORCE_SUPERVISION_REL_LIMIT:.3e} "
            f"meets_production_bound="
            f"{str(force_error <= BF16_FORCE_SUPERVISION_REL_LIMIT).lower()} "
            "status=informational_not_a_gate"
        )
