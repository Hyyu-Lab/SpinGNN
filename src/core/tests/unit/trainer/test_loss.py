import math

import pytest
import torch
import yaml
from nequip.data import AtomicDataDict
from nequip.train import Loss
from nequip.train._loss import PerAtomLoss, PerFrameLoss, SimpleLoss

# all the config to test init
# only the last one will be used to test the loss and mae
dicts = (
    {AtomicDataDict.TOTAL_ENERGY_KEY: (3.0, "MSELoss")},
    {AtomicDataDict.TOTAL_ENERGY_KEY: "MSELoss"},
    [AtomicDataDict.FORCE_KEY, AtomicDataDict.TOTAL_ENERGY_KEY],
    {AtomicDataDict.FORCE_KEY: (1.0, "PerSpeciesMSELoss")},
    {AtomicDataDict.FORCE_KEY: (1.0), "k": (1.0, torch.nn.L1Loss())},
    AtomicDataDict.TOTAL_ENERGY_KEY,
    {
        AtomicDataDict.TOTAL_ENERGY_KEY: (3.0, "L1Loss"),
        AtomicDataDict.FORCE_KEY: (1.0),
        "k": 1.0,
    },
)
nan_dict = {
    AtomicDataDict.TOTAL_ENERGY_KEY: (3.0, "L1Loss", {"ignore_nan": True}),
    AtomicDataDict.FORCE_KEY: (1.0, "MSELoss", {"ignore_nan": True}),
    "k": 1.0,
}


class TestInit:
    @pytest.mark.parametrize("loss", dicts, indirect=True)
    def test_init(self, loss):

        assert len(loss.funcs) == len(loss.coeffs)
        for key, value in loss.coeffs.items():
            assert isinstance(value, torch.Tensor)


class TestLoss:
    @pytest.mark.parametrize("loss", dicts[-2:], indirect=True)
    def test_loss(self, loss, data):

        pred, ref = data

        loss_value = loss(pred, ref)

        loss_value, contrib = loss_value
        assert len(contrib) > 0
        assert isinstance(contrib, dict)
        for key, value in contrib.items():
            assert isinstance(value, torch.Tensor)

        assert isinstance(loss_value, torch.Tensor)

    def test_per_species(self, data):

        pred, ref = data

        config = {AtomicDataDict.FORCE_KEY: (1.0, "PerSpeciesMSELoss")}
        loss = Loss(coeffs=config)

        l, contb = loss(pred, ref)

        # first half data are specie 1
        loss_ref_1 = torch.square(
            pred[AtomicDataDict.FORCE_KEY][:5] - ref[AtomicDataDict.FORCE_KEY][:5]
        ).mean()
        loss_ref_0 = torch.square(
            pred[AtomicDataDict.FORCE_KEY][5:] - ref[AtomicDataDict.FORCE_KEY][5:]
        ).mean()

        assert torch.isclose(
            contb[AtomicDataDict.FORCE_KEY], (loss_ref_0 + loss_ref_1) / 2.0
        )

    def test_per_atom(self, data):

        pred, ref = data

        config = {AtomicDataDict.TOTAL_ENERGY_KEY: (1.0, "PerAtomMSELoss")}
        loss = Loss(coeffs=config)

        l, contb = loss(pred, ref)

        # first graph
        loss_ref_1 = torch.square(
            (
                pred[AtomicDataDict.TOTAL_ENERGY_KEY][0]
                - ref[AtomicDataDict.TOTAL_ENERGY_KEY][0]
            )
            / 3.0
        )
        # second graph
        loss_ref_2 = torch.square(
            (
                pred[AtomicDataDict.TOTAL_ENERGY_KEY][1]
                - ref[AtomicDataDict.TOTAL_ENERGY_KEY][1]
            )
            / 7.0
        )
        loss_ref = (loss_ref_1 + loss_ref_2) / 2.0

        assert torch.isclose(l, loss_ref)


class TestNaN:
    def test_loss(self, data_w_NaN):

        pred, ref, wo_nan_pred, wo_nan_ref = data_w_NaN

        loss = Loss(coeffs=nan_dict)
        l, contb = loss(pred, ref)
        l_wo_nan, contb_wo_nan = loss(wo_nan_pred, wo_nan_ref)

        assert torch.isclose(l_wo_nan, l)
        for k in contb:
            assert torch.isclose(contb_wo_nan[k], contb[k])

    def test_per_species(self, data_w_NaN):

        pred, ref, wo_nan_pred, wo_nan_ref = data_w_NaN

        config = {
            AtomicDataDict.FORCE_KEY: (1.0, "PerSpeciesMSELoss", {"ignore_nan": True})
        }
        loss = Loss(coeffs=config)

        l, contb = loss(pred, ref)
        l_wo_nan, contb_wo_nan = loss(wo_nan_pred, wo_nan_ref)

        assert torch.isclose(l_wo_nan, l)
        for k in contb:
            assert torch.isclose(contb_wo_nan[k], contb[k])

    def test_per_atom(self, data_w_NaN):

        pred, ref, wo_nan_pred, wo_nan_ref = data_w_NaN

        config = {
            AtomicDataDict.TOTAL_ENERGY_KEY: (
                1.0,
                "PerAtomMSELoss",
                {"ignore_nan": True},
            )
        }
        loss = Loss(coeffs=config)
        l_wo_nan, contb_wo_nan = loss(wo_nan_pred, wo_nan_ref)

        l, contb = loss(pred, ref)

        assert torch.isclose(l_wo_nan, l)
        for k in contb:
            assert torch.isclose(contb_wo_nan[k], contb[k])

        # first half data are specie 1
        loss_ref = torch.square(
            (
                pred[AtomicDataDict.TOTAL_ENERGY_KEY][0]
                - ref[AtomicDataDict.TOTAL_ENERGY_KEY][0]
            )
            / 3.0
        )

        assert torch.isclose(l, loss_ref)

    @pytest.mark.parametrize(
        ("atom_counts", "label_shape", "nan_mode"),
        [
            ([2], (1,), "all"),
            ([2, 3, 4], (2, 2), "partial_and_frame"),
            ([1, 2, 3, 4], (), "frame"),
        ],
    )
    def test_per_atom_nan_batch_and_label_shapes(
        self, atom_counts, label_shape, nan_mode
    ):
        batch_size = len(atom_counts)
        shape = (batch_size,) + label_shape
        pred_value = torch.arange(
            math.prod(shape), dtype=torch.get_default_dtype()
        ).reshape(shape)
        pred_value = (pred_value / 10.0).requires_grad_()
        ref_value = pred_value.detach() + 0.25

        if nan_mode == "all":
            ref_value[:] = torch.nan
        elif nan_mode == "partial_and_frame":
            ref_value[0, 0, 0] = torch.nan
            ref_value[-1] = torch.nan
        else:
            ref_value[-1] = torch.nan

        batch = torch.repeat_interleave(
            torch.arange(batch_size), torch.as_tensor(atom_counts)
        )
        pred = {AtomicDataDict.TOTAL_ENERGY_KEY: pred_value}
        ref = {
            AtomicDataDict.TOTAL_ENERGY_KEY: ref_value,
            AtomicDataDict.BATCH_KEY: batch,
        }

        loss = PerAtomLoss("MSELoss", {"ignore_nan": True})
        actual = loss(pred, ref, AtomicDataDict.TOTAL_ENERGY_KEY, mean=False)

        valid = torch.isfinite(ref_value)
        n_atoms = torch.as_tensor(atom_counts, dtype=pred_value.dtype).reshape(
            (batch_size,) + (1,) * len(label_shape)
        )
        expected = (
            torch.square(pred_value - torch.nan_to_num(ref_value, nan=0.0))
            * valid
            / torch.square(n_atoms)
        )
        assert actual.shape == shape
        assert torch.isfinite(actual).all()
        assert torch.allclose(actual, expected)

        reduced = loss(pred, ref, AtomicDataDict.TOTAL_ENERGY_KEY)
        expected_reduced = expected.sum() / valid.sum().clamp_min(1)
        assert torch.isfinite(reduced)
        assert torch.allclose(reduced, expected_reduced)
        reduced.backward()
        assert torch.isfinite(pred_value.grad).all()


class _FrameSumObject:
    """Loss object referenced by dotted name from a plain-string config."""

    def __call__(self, pred, ref, key, mean=True):
        return (pred[key] - ref[key]).abs().sum()

    def sum_and_count(self, pred, ref, key):
        return self(pred, ref, key), torch.as_tensor(1)


_FRAME_SUM_OBJECT = _FrameSumObject()


class TestLossByName:
    def test_class_and_dotted_name(self, data):
        pred, ref = data
        key = AtomicDataDict.FORCE_KEY
        expected = torch.nn.functional.l1_loss(pred[key], ref[key])
        # configs launched through yaml.safe_load can only hold plain strings
        config = yaml.safe_load(
            f"""
            by_name: {{{key}: [1.0, L1Loss]}}
            by_dotted_name: {{{key}: [1.0, torch.nn.L1Loss]}}
            by_dotted_object: {{{key}: [1.0, {__name__}._FRAME_SUM_OBJECT]}}
            """
        )
        for coeffs in (
            config["by_name"],
            config["by_dotted_name"],
            {key: (1.0, torch.nn.L1Loss)},
        ):
            loss = Loss(coeffs=coeffs)
            assert type(loss.funcs[key]) is SimpleLoss
            assert loss.funcs[key].func_name == "L1Loss"
            value, contrib = loss(pred, ref)
            assert torch.allclose(value, expected)
            numerator, count = loss.funcs[key].sum_and_count(pred, ref, key)
            assert torch.allclose(numerator / count, expected)

        loss = Loss(coeffs=config["by_dotted_object"])
        assert loss.funcs[key] is _FRAME_SUM_OBJECT
        value, _ = loss(pred, ref)
        assert torch.allclose(value, (pred[key] - ref[key]).abs().sum())

    def test_unknown_name(self):
        with pytest.raises(NameError, match="not found in module"):
            Loss(coeffs={AtomicDataDict.FORCE_KEY: (1.0, "NoSuchLoss")})


def _ragged_per_frame_data(atom_counts, n_components=9, seed=0):
    generator = torch.Generator().manual_seed(seed)
    n_atoms = sum(atom_counts)
    dtype = torch.get_default_dtype()
    pred = torch.randn(n_atoms, n_components, generator=generator, dtype=dtype)
    ref = torch.randn(n_atoms, n_components, generator=generator, dtype=dtype)
    batch = torch.repeat_interleave(
        torch.arange(len(atom_counts)), torch.as_tensor(atom_counts)
    )
    return pred.requires_grad_(), ref, batch


class TestPerFrame:
    key = AtomicDataDict.BORN_EFFECTIVE_CHARGE_KEY

    def test_registered_names(self):
        config = yaml.safe_load(
            f"""
            {self.key}: [4, PerFrameL1Loss, {{mask_field: routed}}]
            {AtomicDataDict.TOTAL_ENERGY_KEY}: [1.0, PerFrameMSELoss]
            """
        )
        loss = Loss(coeffs=config)
        bec_loss = loss.funcs[self.key]
        assert isinstance(bec_loss, PerFrameLoss)
        assert bec_loss.func_name == "L1Loss"
        assert bec_loss.mask_field == "routed"
        assert float(loss.coeffs[self.key]) == 4.0
        energy_loss = loss.funcs[AtomicDataDict.TOTAL_ENERGY_KEY]
        assert isinstance(energy_loss, PerFrameLoss)
        assert energy_loss.func_name == "MSELoss"
        assert energy_loss.mask_field is None

    def test_ragged_frames_sum_and_count(self):
        atom_counts = [2, 5, 1]
        pred, ref, batch = _ragged_per_frame_data(atom_counts)
        loss = PerFrameLoss("L1Loss")
        pred_d = {self.key: pred}
        ref_d = {self.key: ref, AtomicDataDict.BATCH_KEY: batch}

        expected = torch.stack(
            [(pred[batch == f] - ref[batch == f]).abs().mean() for f in range(3)]
        )
        per_frame = loss(pred_d, ref_d, self.key, mean=False)
        assert per_frame.shape == (3,)
        assert torch.allclose(per_frame, expected)

        numerator, count = loss.sum_and_count(pred_d, ref_d, self.key)
        assert count.item() == 3
        assert torch.allclose(numerator, expected.sum())
        assert torch.allclose(loss(pred_d, ref_d, self.key), expected.mean())

        # every frame has equal weight: d/dpred of S is sign / (atoms_f * 9)
        numerator.backward()
        scale = 1.0 / (torch.as_tensor(atom_counts, dtype=pred.dtype)[batch] * 9)
        assert torch.allclose(
            pred.grad, torch.sign(pred.detach() - ref) * scale.unsqueeze(-1)
        )

    def test_per_frame_field(self):
        energy = AtomicDataDict.TOTAL_ENERGY_KEY
        batch = torch.as_tensor([0, 0, 1, 1, 1])
        pred = {energy: torch.as_tensor([[1.0], [2.0]])}
        ref = {energy: torch.as_tensor([[1.5], [0.0]]), AtomicDataDict.BATCH_KEY: batch}
        loss = PerFrameLoss("L1Loss")
        assert torch.allclose(
            loss(pred, ref, energy, mean=False),
            torch.as_tensor([0.5, 2.0], dtype=torch.get_default_dtype()),
        )
        numerator, count = loss.sum_and_count(pred, ref, energy)
        assert count.item() == 2
        assert torch.allclose(numerator, torch.as_tensor(2.5, dtype=numerator.dtype))

    @pytest.mark.parametrize("level", ["frame", "atom"])
    @pytest.mark.parametrize("source", ["pred", "ref"])
    def test_mask(self, level, source):
        atom_counts = [2, 3, 4]
        pred, ref, batch = _ragged_per_frame_data(atom_counts, seed=1)
        frame_keep = torch.as_tensor([True, False, True])
        if level == "frame":
            mask = frame_keep
            atom_keep = frame_keep[batch]
        else:
            # frame 2 keeps only its first two atoms
            atom_keep = frame_keep[batch].clone()
            atom_keep[batch == 2] = torch.as_tensor([True, True, False, False])
            mask = atom_keep.unsqueeze(-1)
        pred_d = {self.key: pred}
        ref_d = {self.key: ref, AtomicDataDict.BATCH_KEY: batch}
        (pred_d if source == "pred" else ref_d)["keep"] = mask

        loss = PerFrameLoss("L1Loss", {"mask_field": "keep"})
        expected = torch.stack(
            [
                (pred[(batch == f) & atom_keep] - ref[(batch == f) & atom_keep])
                .abs()
                .mean()
                if frame_keep[f]
                else torch.zeros((), dtype=pred.dtype)
                for f in range(3)
            ]
        )
        numerator, count = loss.sum_and_count(pred_d, ref_d, self.key)
        # masked-out frames add zero to S but still count in C
        assert count.item() == 3
        assert torch.allclose(numerator, expected.sum())
        assert torch.allclose(loss(pred_d, ref_d, self.key), expected.sum() / 3)
        numerator.backward()
        assert torch.all(pred.grad[~atom_keep] == 0)
        assert torch.all(pred.grad[atom_keep] != 0)

    def test_all_masked_keeps_graph(self):
        pred, ref, batch = _ragged_per_frame_data([2, 3], seed=2)
        pred_d = {self.key: pred, "keep": torch.as_tensor([False, False])}
        ref_d = {self.key: ref, AtomicDataDict.BATCH_KEY: batch}
        loss = PerFrameLoss("L1Loss", {"mask_field": "keep"})
        numerator, count = loss.sum_and_count(pred_d, ref_d, self.key)
        assert count.item() == 2
        assert numerator.item() == 0.0
        assert numerator.requires_grad
        numerator.backward()
        assert pred.grad is not None
        assert torch.all(pred.grad == 0)

    def test_ref_key_and_nan(self):
        pred, ref, batch = _ragged_per_frame_data([2, 3], seed=3)
        ref_with_nan = ref.clone()
        ref_with_nan[0, :3] = float("nan")
        pred_d = {"aux_bec": pred}
        ref_d = {self.key: ref_with_nan, AtomicDataDict.BATCH_KEY: batch}
        loss = PerFrameLoss("L1Loss", {"ref_key": self.key, "ignore_nan": True})
        per_frame = loss(pred_d, ref_d, "aux_bec", mean=False)
        valid = ref_with_nan == ref_with_nan
        expected = torch.stack(
            [
                (pred[batch == f] - ref[batch == f]).abs()[valid[batch == f]].mean()
                for f in range(2)
            ]
        )
        assert torch.allclose(per_frame, expected)

    def test_bad_shapes(self):
        pred, ref, batch = _ragged_per_frame_data([2, 3], seed=4)
        ref_d = {self.key: ref, AtomicDataDict.BATCH_KEY: batch}
        loss = PerFrameLoss("L1Loss")
        with pytest.raises(ValueError, match="matches neither"):
            loss({self.key: pred[:4]}, {self.key: ref[:4], "batch": batch}, self.key)
        masked = PerFrameLoss("L1Loss", {"mask_field": "keep"})
        with pytest.raises(ValueError, match="one value per frame or per atom"):
            masked(
                {self.key: pred, "keep": torch.ones(2, 2, dtype=torch.bool)},
                ref_d,
                self.key,
            )

    def test_mask_single_frame_without_batch_and_missing_mask(self):
        from nequip.data import AtomicData
        from nequip.utils.torch_geometric import Batch

        pred, ref, _ = _ragged_per_frame_data([3], seed=5)
        positions = torch.eye(3, dtype=ref.dtype)
        loss = PerFrameLoss("L1Loss", {"mask_field": "keep"})
        # a single frame without a batch field, with a per-frame mask
        for keep, expected in (
            (True, (pred - ref).abs().mean()),
            (False, torch.zeros((), dtype=pred.dtype)),
        ):
            ref_d = {
                self.key: ref,
                AtomicDataDict.POSITIONS_KEY: positions,
                "keep": torch.as_tensor([keep]),
            }
            numerator, count = loss.sum_and_count({self.key: pred}, ref_d, self.key)
            assert count.item() == 1
            assert torch.allclose(numerator, expected)
        # a mask field that is neither predicted nor in the data, also for a Batch
        data = AtomicData.from_points(
            pos=positions, r_max=2.0, born_effective_charges=ref
        )
        for container in (
            AtomicData.to_AtomicDataDict(data),
            Batch.from_data_list([data]),
        ):
            with pytest.raises(KeyError, match="neither in the predictions"):
                loss({self.key: pred}, container, self.key)

    @pytest.mark.parametrize("ignore_nan", [False, True])
    @pytest.mark.parametrize("level", ["frame", "atom"])
    @pytest.mark.parametrize("func", ["L1Loss", "MSELoss"])
    def test_masked_nan_placeholder_targets(self, func, level, ignore_nan):
        pred, ref, batch = _ragged_per_frame_data([2, 3, 4], seed=6)
        mask, atom_keep = _per_frame_test_mask(level, batch)
        # unlabeled (masked-out) rows carry NaN placeholder targets
        placeholder = ref.clone()
        placeholder[~atom_keep] = float("nan")
        loss = PerFrameLoss(func, {"mask_field": "keep", "ignore_nan": ignore_nan})

        def run(prediction, target):
            ref_d = {self.key: target, AtomicDataDict.BATCH_KEY: batch, "keep": mask}
            numerator, count = loss.sum_and_count(
                {self.key: prediction}, ref_d, self.key
            )
            numerator.backward()
            return numerator, count

        numerator, count = run(pred, placeholder)
        assert count.item() == 3
        assert torch.isfinite(numerator)
        assert torch.isfinite(pred.grad).all()
        assert torch.all(pred.grad[~atom_keep] == 0)
        # exactly the result with finite targets on the masked rows
        finite_pred = pred.detach().clone().requires_grad_()
        finite_numerator, _ = run(finite_pred, ref)
        assert torch.equal(numerator, finite_numerator)
        assert torch.equal(pred.grad, finite_pred.grad)
        expected = sum(
            loss.func(pred[(batch == f) & atom_keep], ref[(batch == f) & atom_keep])
            .mean()
            .detach()
            for f in (0, 2)
        )
        assert torch.allclose(numerator, expected)

    @pytest.mark.parametrize("mixed_dtype", [False, True])
    @pytest.mark.parametrize("ignore_nan", [False, True])
    @pytest.mark.parametrize("level", ["frame", "atom"])
    @pytest.mark.parametrize("func", ["L1Loss", "MSELoss", "HuberLoss"])
    def test_masked_loss_matches_previous_formula(
        self, func, level, ignore_nan, mixed_dtype
    ):
        """Finite targets: bitwise the per-frame formula before the masked-target
        substitution (values and the gradients of selected rows; masked rows get
        zero gradients in both)."""
        pred, ref, batch = _ragged_per_frame_data([2, 3, 4], seed=7)
        if mixed_dtype:
            # model_dtype float32 predictions, default_dtype float64 targets
            pred = pred.detach().to(torch.float32).requires_grad_()
            ref = ref.to(torch.float64)
        mask, atom_keep = _per_frame_test_mask(level, batch)
        loss = PerFrameLoss(func, {"mask_field": "keep", "ignore_nan": ignore_nan})
        pred_d = {self.key: pred}
        ref_d = {self.key: ref, AtomicDataDict.BATCH_KEY: batch, "keep": mask}

        per_frame = loss(pred_d, ref_d, self.key, mean=False)
        numerator, _ = loss.sum_and_count(pred_d, ref_d, self.key)
        numerator.backward()
        legacy_pred = pred.detach().clone().requires_grad_()
        legacy = _legacy_per_frame(loss, legacy_pred, ref, batch, atom_keep)
        legacy.sum().backward()

        assert _same_bits(per_frame, legacy)
        assert _same_bits(numerator, legacy.sum())
        assert _same_bits(pred.grad[atom_keep], legacy_pred.grad[atom_keep])
        assert torch.all(pred.grad[~atom_keep] == 0)
        assert torch.all(legacy_pred.grad[~atom_keep] == 0)


def _per_frame_test_mask(level, batch):
    """Mask field and row selection for frames of 2, 3 and 4 atoms.

    Frame 1 is masked out; at atom level frame 2 also drops its last two atoms.
    """
    frame_keep = torch.as_tensor([True, False, True])
    atom_keep = frame_keep[batch].clone()
    if level == "frame":
        return frame_keep, atom_keep
    atom_keep[batch == 2] = torch.as_tensor([True, True, False, False])
    return atom_keep.unsqueeze(-1), atom_keep


def _legacy_per_frame(loss, prediction, target, batch, keep):
    """Inline copy of ``PerFrameLoss.per_frame`` before masked-out rows had
    their targets replaced (per-atom field, row mask ``keep``)."""
    from nequip.train._loss import _promote

    prediction, target = _promote(prediction, target)
    if loss.ignore_nan:
        valid = target == target
        target = torch.nan_to_num(target, nan=0.0)
    elementwise = loss.func(prediction, target)
    n_rows = elementwise.shape[0]
    elementwise = elementwise.reshape(n_rows, -1)
    if loss.ignore_nan:
        valid = valid.reshape(n_rows, -1)
        elementwise = elementwise * valid
        row_count = valid.sum(dim=1).to(elementwise.dtype)
    else:
        row_count = torch.full((n_rows,), elementwise.shape[1], dtype=elementwise.dtype)
    row_sum = elementwise.sum(dim=1)
    row_sum = torch.where(keep, row_sum, torch.zeros_like(row_sum))
    row_count = torch.where(keep, row_count, torch.zeros_like(row_count))
    n_frames = int(batch.max()) + 1
    frame_sum = torch.zeros(n_frames, dtype=elementwise.dtype)
    frame_sum = frame_sum.index_add(0, batch, row_sum)
    frame_count = torch.zeros(n_frames, dtype=elementwise.dtype)
    frame_count = frame_count.index_add(0, batch, row_count)
    return frame_sum / frame_count.clamp_min(1)


def _same_bits(a, b):
    """Equal dtype, shape and bit pattern (``torch.equal`` treats -0.0 == 0.0)."""
    as_int = {torch.float32: torch.int32, torch.float64: torch.int64}[a.dtype]
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(a.detach().view(as_int), b.detach().view(as_int))
    )


@pytest.mark.parametrize(
    "func", ["MSELoss", "HuberLoss", "L1Loss", "PerAtomMSELoss", "PerFrameMSELoss"]
)
def test_float32_prediction_float64_target(func):
    """``model_dtype: float32`` predictions against ``default_dtype: float64`` targets."""
    key = AtomicDataDict.TOTAL_ENERGY_KEY
    generator = torch.Generator().manual_seed(0)
    pred64 = torch.randn(2, 1, generator=generator, dtype=torch.float64)
    target = torch.randn(2, 1, generator=generator, dtype=torch.float64)
    batch = torch.as_tensor([0, 0, 1, 1, 1])
    pred32 = pred64.to(torch.float32).requires_grad_()
    reference = pred32.detach().to(torch.float64).requires_grad_()
    loss = Loss(coeffs={key: (1.0, func, {"ignore_nan": True})})
    value, _ = loss({key: pred32}, {key: target, AtomicDataDict.BATCH_KEY: batch})
    expected, _ = loss({key: reference}, {key: target, AtomicDataDict.BATCH_KEY: batch})
    # the loss runs in the target dtype; the gradient returns as float32
    assert value.dtype == torch.float64
    assert torch.equal(value, expected)
    value.backward()
    expected.backward()
    assert pred32.grad.dtype == torch.float32
    assert torch.equal(pred32.grad, reference.grad.to(torch.float32))
    numerator, count = loss.funcs[key].sum_and_count(
        {key: pred32}, {key: target, AtomicDataDict.BATCH_KEY: batch}, key
    )
    assert numerator.dtype == torch.float64


@pytest.mark.parametrize("container", ["dict", "batch"])
def test_promoted_view_is_a_read_only_mapping(container):
    from nequip.data import AtomicData
    from nequip.train._loss import _OverlayMapping, _promote_key

    key = AtomicDataDict.FORCE_KEY
    data = AtomicData.from_points(
        pos=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        r_max=2.0,
        forces=torch.zeros(2, 3, dtype=torch.float64),
    )
    ref = AtomicData.to_AtomicDataDict(data) if container == "dict" else data
    pred = {key: torch.ones(2, 3, dtype=torch.float32)}
    pred_view, ref_view = _promote_key(pred, ref, key)
    assert isinstance(pred_view, _OverlayMapping)
    assert isinstance(ref_view, _OverlayMapping)

    # promoted value overlaid, originals untouched and not copied
    assert pred_view[key].dtype == torch.float64
    assert pred[key].dtype == torch.float32
    assert ref_view[key] is ref[key]
    base_keys = set(ref.keys) if container == "batch" else set(ref.keys())
    assert set(ref_view.keys()) == base_keys
    assert len(ref_view) == len(base_keys)
    assert dict(ref_view.items())[AtomicDataDict.POSITIONS_KEY] is ref["pos"]
    assert set(pred_view.keys()) == {key}
    assert list(pred_view.items()) == [(key, pred_view[key])]
    assert len(list(pred_view.values())) == 1
    assert pred_view.get(key) is pred_view[key]

    # missing keys behave like a dict (a Batch alone would return None)
    assert "missing" not in ref_view
    assert ref_view.get("missing", "default") == "default"
    with pytest.raises(KeyError):
        ref_view["missing"]
    # read-only
    with pytest.raises(TypeError):
        ref_view[key] = pred[key]

    # same-dtype pairs are returned as-is
    same = {key: torch.zeros(2, 3, dtype=torch.float64)}
    pred_same, ref_same = _promote_key(same, ref, key)
    assert pred_same is same and ref_same is ref


@pytest.fixture(scope="class")
def loss(request):
    """"""
    d = request.param
    instance = Loss(coeffs=d)
    yield instance


@pytest.fixture(scope="module")
def data(float_tolerance):
    """"""
    pred = {
        AtomicDataDict.FORCE_KEY: torch.rand(10, 3),
        AtomicDataDict.TOTAL_ENERGY_KEY: torch.rand((2, 1)),
        "k": torch.rand((2, 1)),
        AtomicDataDict.ATOM_TYPE_KEY: torch.as_tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
    }
    ref = {
        AtomicDataDict.BATCH_KEY: torch.tensor(
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int
        ),
        AtomicDataDict.FORCE_KEY: torch.rand(10, 3),
        AtomicDataDict.TOTAL_ENERGY_KEY: torch.rand((2, 1)),
        "k": torch.rand((2, 1)),
        AtomicDataDict.ATOM_TYPE_KEY: torch.as_tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
    }
    yield pred, ref


@pytest.fixture(scope="module")
def data_w_NaN(float_tolerance, data):
    """"""
    _pred, _ref = data

    pred = {k: torch.clone(v) for k, v in _pred.items()}
    ref = {k: torch.clone(v) for k, v in _ref.items()}
    ref[AtomicDataDict.FORCE_KEY][-1] = float("nan")
    ref[AtomicDataDict.FORCE_KEY][0] = float("nan")
    ref[AtomicDataDict.TOTAL_ENERGY_KEY][1] = float("nan")

    wo_nan_pred = {k: torch.clone(v) for k, v in _pred.items()}
    wo_nan_ref = {k: torch.clone(v) for k, v in _ref.items()}
    wo_nan_ref[AtomicDataDict.FORCE_KEY] = wo_nan_ref[AtomicDataDict.FORCE_KEY][1:-1]
    wo_nan_ref[AtomicDataDict.TOTAL_ENERGY_KEY] = wo_nan_ref[
        AtomicDataDict.TOTAL_ENERGY_KEY
    ][:1]
    wo_nan_ref[AtomicDataDict.ATOM_TYPE_KEY] = wo_nan_ref[AtomicDataDict.ATOM_TYPE_KEY][
        1:-1
    ]
    wo_nan_ref[AtomicDataDict.BATCH_KEY] = torch.tensor([0, 0, 0], dtype=torch.int)
    wo_nan_pred[AtomicDataDict.FORCE_KEY] = wo_nan_pred[AtomicDataDict.FORCE_KEY][1:-1]
    wo_nan_pred[AtomicDataDict.ATOM_TYPE_KEY] = wo_nan_pred[
        AtomicDataDict.ATOM_TYPE_KEY
    ][1:-1]
    wo_nan_pred[AtomicDataDict.TOTAL_ENERGY_KEY] = wo_nan_pred[
        AtomicDataDict.TOTAL_ENERGY_KEY
    ][:1]

    yield pred, ref, wo_nan_pred, wo_nan_ref
