"""Lower supported source-checkout inference modules to static v1 operations."""

from pathlib import Path
import runpy

from .contract import write_contract


def _activate_checkout(root=None):
    candidates = (
        [Path(root)] if root else [Path(__file__).resolve().parents[2], Path.cwd()]
    )
    for path in candidates:
        guard = path / "_spingnn_source.py"
        if guard.is_file():
            runpy.run_path(str(guard))["activate_source_tree"](path)
            return
    raise ValueError(
        "The exporter requires a SpinGNN2 source checkout (set source_root)"
    )


def _array(value):
    return value.detach().cpu().numpy().copy()


def _affine(layer):
    return {"weight": _array(layer.weight), "bias": _array(layer.bias)}


def _layer(module):
    from .backbone_export import export_backbone, export_mlp
    from .basis_export import export_harmonics, export_radial, export_spin_basis

    op = type(module).__name__
    if not type(module).__module__.startswith(("nequip.nn.", "allegro.nn.")):
        raise ValueError(f"Unsupported model layer {type(module)!r}")

    def attrs(*names):
        return {key: getattr(module, key) for key in names}

    if op in ("OneHotAtomEncoding", "OneHotAtomEncoding_spin"):
        return attrs("num_types")
    if op in (
        "SphericalHarmonicEdgeAttrs",
        "SphericalHarmonicEdgeAttrs_spin_nosoc",
        "SphericalHarmonicEdgeAttrs_spin_soc_simple",
    ):
        return dict(**attrs("out_field"), harmonics=export_harmonics(module))
    if op == "RadialBasisEdgeEncoding":
        return dict(**attrs("out_field"), basis=export_radial(module))
    if op in ("EdgeScalarAddSpin", "RadialBasisEdgeEncoding_spin"):
        return dict(
            **attrs("out_field"),
            field=getattr(module, "field", None),
            basis=export_spin_basis(module),
        )
    if op in ("Allegro_og_Module", "Allegro_spin_Module", "Allegro_born_Module"):
        if module.node_invariant_field != "node_attrs":
            raise ValueError(
                "Unsupported backbone node input " + module.node_invariant_field
            )
        return dict(
            **attrs("field", "edge_invariant_field"), backbone=export_backbone(module)
        )
    if op == "ScalarMLP":
        return dict(
            **attrs("field", "out_field", "add_collect", "cat_collect", "normalize"),
            coeff=module.coeff.item(),
            mlp=export_mlp(module._module),
        )
    if op in ("HB_Energy", "Biquad_Energy", "Spin_JMatrix_Energy", "Spin_ani_Energy"):
        return dict(
            **attrs("field", "out_field", "add_collect"), coeff=module.coeff.item()
        )
    if op == "EdgewiseEnergySum":
        if module.ldelta_spin or module.edgewise_feature_sum:
            raise ValueError("Unsupported delta-spin or feature reduction")
        cutoff = None
        if module.ledge_cutoff:
            if type(module.cutoff).__name__ != "PolynomialCutoff":
                raise ValueError("Unsupported edge energy cutoff")
            cutoff = [module.cutoff._factor, int(module.cutoff.p)]
        return dict(
            factor=module._factor,
            cutoff=cutoff,
            scales=(
                _array(module.per_edge_scales)
                if module.per_edge_species_scale
                else None
            ),
        )
    if op == "EdgewiseReduce":
        if module.reduce not in ("sum", "mean"):
            raise ValueError("Unsupported edge reduction " + module.reduce)
        return dict(**attrs("field", "out_field", "reduce"), factor=module._factor)
    if op == "AtomwiseReduce":
        if (module.field, module.out_field) != ("atomic_energy", "total_energy"):
            raise ValueError(f"Unsupported atomwise reduction {module.out_field}")
        return attrs("reduce", "field", "out_field", "constant")
    if op == "PerSpeciesScaleShift":
        if module.field != "atomic_energy" or module.out_field != module.field:
            raise ValueError("Unsupported per-species field " + module.field)
        return dict(
            type_names=list(module.type_names),
            scales=_array(module.scales) if module.has_scales else None,
            shifts=_array(module.shifts) if module.has_shifts else None,
        )
    if op == "CartesianBECReduce":
        if len(module.converters) != 1 or not module.converters[0].identity:
            raise ValueError("DREAM requires the Cartesian BEC backbone output")
        return dict(
            **attrs("asr", "reduce", "out_field"),
            field=module.converters[0].field,
            factor=module._factor,
            scales=_array(module.scales),
            shifts=_array(module.shifts),
            total=module._frame_mean_field if module._write_total else None,
            pre_asr=module._pre_asr_field if module._write_pre_asr else None,
        )
    if op == "e2IP_nodewise":
        return dict(
            **attrs("prefixes", "spin_force_scale", "soc_f_coeff"),
            Q_e2ip=_array(module.transform),
            heads={
                prefix: dict(
                    matrix_to_irreps=_array(head.matrix_to_irreps),
                    gate=_affine(head.gate_linear),
                    scalar=_affine(head.scalar_linear),
                )
                for prefix, head in module.sigma_heads.items()
            },
        )
    raise ValueError(f"Unsupported model layer {op}")


def export_model(model, stem, *, source_root=None):
    """Export evaluation semantics, preserving the original model and its device."""
    _activate_checkout(source_root)
    from nequip.nn import RescaleOutput, SequentialGraphNetwork
    from nequip.nn._grad_output import ParaStressForceSpinOutput, ParaStressOutput

    rescales = []
    while type(model) is RescaleOutput:
        rescales.append(
            dict(
                scale=model.scale_by.item() if model.has_scale else 1.0,
                shift=model.shift_by.item() if model.has_shift else 0.0,
                scale_keys=list(model.scale_keys),
                shift_keys=list(model.shift_keys),
                square_keys=list(model.square_scale_keys),
            )
        )
        model = model.model
    spin_fv_coeff = 1.0
    if type(model) in (ParaStressOutput, ParaStressForceSpinOutput):
        if isinstance(model, ParaStressForceSpinOutput):
            spin_fv_coeff = model.spin_fv_coeff.item()
        model = model.func
    if type(model) is not SequentialGraphNetwork:
        raise ValueError(f"Unsupported model {type(model).__name__}")
    names = set(model._modules)
    if "born_scale" in names:
        family = "Allegro_born"
    elif "allegro_soc_on" in names or (
        "allegro_onlyr" in names
        and type(model.allegro_onlyr).__name__ == "Allegro_spin_Module"
    ):
        family = "Allegro_soc_e2IP" if "e2IP_nodewise" in names else "Allegro_soc"
    elif "allegro_onlyr" in names:
        family = "Allegro_nosoc"
    elif {"allegro", "e2IP_nodewise"} <= names:
        family = "Allegro_e2IP"
    else:
        raise ValueError("Unsupported model: expected a local SpinGNN++ builder")
    layers = []
    for name, module in model.named_children():
        try:
            layers.append(dict(name=name, op=type(module).__name__, **_layer(module)))
        except (ValueError, NotImplementedError) as exc:
            raise ValueError(f"Cannot export {family}.{name}: {exc}") from exc
    dtype = str(next(model.parameters()).detach().cpu().numpy().dtype)
    if dtype not in ("float64", "float32"):
        raise ValueError(f"Unsupported model precision {dtype}")
    return write_contract(
        stem,
        dict(
            family=family,
            dtype=dtype,
            layers=layers,
            spin_fv_coeff=spin_fv_coeff,
            rescales=rescales[::-1],
        ),
    )


def export_checkpoint(checkpoint, stem, *, config=None, source_root=None):
    """Read the source training state_dict with its saved builder configuration."""
    _activate_checkout(source_root)
    from nequip.train import Trainer
    from nequip.utils import Config
    from nequip.utils._global_options import global_options_context

    path = Path(checkpoint)
    if path.is_dir():
        path = path / "best_model.pth"
    cfg = Config.from_file(str(config or path.with_name("config.yaml")))
    with global_options_context(cfg):
        model, _ = Trainer.load_model_from_training_session(
            path.parent, model_name=path.name, device="cpu", config_dictionary=dict(cfg)
        )
        return export_model(model.eval(), stem, source_root=source_root)
