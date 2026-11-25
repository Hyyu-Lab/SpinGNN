from nequip.nn import PackagedBackbone, SequentialGraphNetwork


def PackagedBackboneModel(config) -> SequentialGraphNetwork:
    """Start a model with a ``torch.package``-d backbone (layer ``packaged_backbone``).

    Config keys (all prefixed ``packaged_backbone_``):
      ``path`` (required), ``taps`` (required), ``stop_after``,
      ``trainable_prefixes`` (default: none, i.e. fully frozen; a prefix matches
      a parameter name relative to the packaged network, with or without its
      leading stage name),
      ``compile_mode`` (default ``eager``), ``model_key`` (default
      ``sole_model``), ``keep_skipped_stages`` (default ``False``),
      ``torch_compile`` (default: off; ``true`` or a mapping with ``enabled``,
      ``backend`` (``inductor``), ``dynamic`` (``true``), ``mode`` and
      ``inductor_options`` to ``torch.compile`` the packaged stages).
    ``type_names`` and ``r_max`` are checked against the package. See
    :class:`nequip.nn.PackagedBackbone` for the semantics. Later builders add
    layers that read the tapped fields.
    """
    prefix = "packaged_backbone_"
    for required in ("path", "taps"):
        if config.get(prefix + required, None) is None:
            raise KeyError(f"PackagedBackboneModel requires `{prefix}{required}`")
    backbone = PackagedBackbone(
        package_path=config[prefix + "path"],
        taps=config[prefix + "taps"],
        stop_after=config.get(prefix + "stop_after", None),
        trainable_prefixes=list(config.get(prefix + "trainable_prefixes", None) or []),
        type_names=config.get("type_names", None),
        r_max=config.get("r_max", None),
        compile_mode=config.get(prefix + "compile_mode", "eager"),
        model_key=config.get(prefix + "model_key", "sole_model"),
        keep_skipped_stages=bool(config.get(prefix + "keep_skipped_stages", False)),
        torch_compile=config.get(prefix + "torch_compile", None),
    )
    return SequentialGraphNetwork({"packaged_backbone": backbone})
