import contextlib

import torch

from nequip.nn import SequentialGraphNetwork
from nequip.utils.misc import dtype_from_name

from allegro.nn import TensorFeatureBECReadout


def TensorFeatureBEC(config, model: SequentialGraphNetwork) -> SequentialGraphNetwork:
    """Append a :class:`allegro.nn.TensorFeatureBECReadout` layer named ``bec_readout``.

    The readout's arguments are read from ``bec_readout_<argument>`` config
    keys (the usual layer-prefix rule), e.g. ``bec_readout_tensor_field``,
    ``bec_readout_num_tensor_features``, ``bec_readout_hidden_channels``,
    ``bec_readout_antisymmetric_channels``, ``bec_readout_asr``.

    ``bec_readout_construct_dtype`` (e.g. ``float64``) builds the readout
    under that default dtype (initial weights are drawn in it) and then casts
    it to the dtype the model is being built in. By default the readout is
    built directly in that dtype.
    """
    return append_layer(config, model, "bec_readout", TensorFeatureBECReadout)


def append_layer(config, model, name, builder, params=None):
    """Append a layer with optional construction dtype and forced parameters."""
    construct_dtype = config.get(f"{name}_construct_dtype", None)
    target_dtype = torch.get_default_dtype()
    context = contextlib.nullcontext()
    if construct_dtype is not None:
        context = _default_dtype(dtype_from_name(construct_dtype))
    with context:
        model.append_from_parameters(
            shared_params=config,
            name=name,
            builder=builder,
            params={} if params is None else params,
        )
    if construct_dtype is not None:
        getattr(model, name).to(dtype=target_dtype)
    return model


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)
