import inspect
from importlib import import_module

from nequip import data
from nequip.data.transforms import TypeMapper
from nequip.data import AtomicDataset, register_fields
from nequip.utils import instantiate, get_w_prefix


def dataset_from_config(config, prefix: str = "dataset") -> AtomicDataset:
    """initialize database based on a config instance

    It needs dataset type name (case insensitive),
    and all the parameters needed in the constructor.

    Examples see tests/data/test_dataset.py TestFromConfig
    and tests/datasets/test_simplest.py

    Args:

    config (dict, nequip.utils.Config): dict/object that store all the parameters
    prefix (str): Optional. The prefix of all dataset parameters

    Return:

    dataset (nequip.data.AtomicDataset)
    """

    config_dataset = config.get(prefix, None)
    if config_dataset is None:
        raise KeyError(f"Dataset with prefix `{prefix}` isn't present in this config!")

    dataset_name = str(config_dataset).lower()
    if inspect.isclass(config_dataset):
        # user define class
        class_name = config_dataset
    else:
        class_name = None
        # A dotted name is an explicit import request. Only an absent module
        # or class falls back to the built-in short-name lookup; errors raised
        # while importing an existing module must remain visible to callers.
        if isinstance(config_dataset, str) and "." in config_dataset:
            module_name, class_attr = config_dataset.rsplit(".", 1)
            try:
                module = import_module(module_name)
            except ModuleNotFoundError as exc:
                # ``import a.b.c`` reports ``a.b`` when the requested module
                # is absent. A missing unrelated dependency has a different
                # name and must not be mistaken for an unknown dataset.
                missing_name = exc.name
                if missing_name is None or not (
                    missing_name == module_name
                    or module_name.startswith(f"{missing_name}.")
                ):
                    raise
            else:
                try:
                    class_name = getattr(module, class_attr)
                except AttributeError:
                    class_name = None

        if class_name is None:
            # Default classes defined in nequip.data or nequip.dataset.
            for k, v in inspect.getmembers(data, inspect.isclass):
                if k.endswith("Dataset"):
                    if k.lower() == dataset_name:
                        class_name = v
                    if k[:-7].lower() == dataset_name:
                        class_name = v
                elif k.lower() == dataset_name:
                    class_name = v

    if class_name is None:
        raise NameError(f"dataset type {dataset_name} does not exists")

    # if dataset r_max is not found, use the universal r_max
    eff_key = "extra_fixed_fields"
    prefixed_eff_key = f"{prefix}_{eff_key}"
    config[prefixed_eff_key] = get_w_prefix(
        eff_key, {}, prefix=prefix, arg_dicts=config
    )
    config[prefixed_eff_key]["r_max"] = get_w_prefix(
        "r_max",
        prefix=prefix,
        arg_dicts=[config[prefixed_eff_key], config],
    )

    # Build a TypeMapper from the config
    type_mapper, _ = instantiate(TypeMapper, prefix=prefix, optional_args=config)

    # Register fields:
    # This might reregister fields, but that's OK:
    instantiate(register_fields, all_args=config)

    instance, _ = instantiate(
        class_name,
        prefix=prefix,
        positional_args={"type_mapper": type_mapper},
        optional_args=config,
    )

    return instance
