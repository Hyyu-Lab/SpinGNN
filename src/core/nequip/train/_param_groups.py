"""Named optimizer parameter groups (``optimizer_param_groups``)."""

import re
from typing import Dict, List, Optional, Tuple, Union

import torch

_RESERVED_KEYS = ("name", "match", "params")
_REGEX_PREFIX = "re:"
_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.\-]+")


def _matcher(pattern: str, where: str):
    if pattern.startswith(_REGEX_PREFIX):
        try:
            compiled = re.compile(pattern[len(_REGEX_PREFIX) :])
        except re.error as error:
            raise ValueError(
                f"{where}: invalid regular expression {pattern!r}"
            ) from error
        return lambda name: compiled.search(name) is not None
    return lambda name: name.startswith(pattern)


def _coerce_option(value):
    # PyYAML reads ``1e-4`` (no decimal point) as a string
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def build_optimizer_param_groups(
    model: torch.nn.Module, specs: List[dict]
) -> Tuple[List[dict], List[Dict[str, Union[str, List[str]]]]]:
    """Split the trainable parameters of ``model`` into named optimizer groups.

    ``specs`` is an ordered list of ``{name, match, <optimizer options>}``:

    - ``name``: unique group name (letters, digits, ``_``, ``-``, ``.``),
      stored in the optimizer param group and logged as ``LR_<name>``;
    - ``match``: list of parameter-name patterns. A plain string matches names
      that start with it; ``"re:<regex>"`` matches with ``re.search``. Groups
      with ``match`` are tried in list order and the first match wins. One
      group may omit ``match``; it is the catch-all that receives every
      remaining trainable parameter, wherever it sits in the list.
    - any other key (e.g. ``lr``, ``weight_decay``) is a per-group optimizer
      option overriding the optimizer defaults.

    Names are those of ``model.named_parameters()``, i.e. the state_dict keys
    of the unwrapped model. Parameters with ``requires_grad=False`` are not
    put in any group. Every trainable parameter must land in a group and every
    group must receive at least one parameter, otherwise ``ValueError`` is
    raised. The list order is the optimizer group index order.

    Returns ``(param_groups, structure)``: the torch param-group dicts and the
    ``[{"name", "params": [parameter names]}]`` layout used to check resumes.
    """
    if not isinstance(specs, (list, tuple)) or len(specs) == 0:
        raise ValueError("optimizer_param_groups must be a non-empty list of groups")

    groups = []
    catch_all: Optional[int] = None
    seen = set()
    for index, spec in enumerate(specs):
        where = f"optimizer_param_groups[{index}]"
        if not isinstance(spec, dict):
            raise ValueError(f"{where} must be a mapping, got {spec!r}")
        name = spec.get("name", None)
        if not isinstance(name, str) or len(name) == 0:
            raise ValueError(f"{where} needs a non-empty string `name`")
        if _NAME_PATTERN.fullmatch(name) is None:
            # names become ``LR_<name>`` columns of the epoch metrics CSV
            raise ValueError(
                f"{where}: group name {name!r} may only contain letters, digits, "
                "'_', '-' and '.'"
            )
        if name in seen:
            raise ValueError(f"{where}: duplicate group name {name!r}")
        seen.add(name)
        if "params" in spec:
            raise ValueError(f"{where} ({name}): `params` is set by `match`")
        match = spec.get("match", None)
        if match is None:
            if catch_all is not None:
                raise ValueError(
                    f"{where} ({name}): only one group may omit `match`; "
                    f"{groups[catch_all][0]!r} is already the catch-all"
                )
            catch_all = index
            matchers = None
        else:
            if isinstance(match, str):
                match = [match]
            if (
                not isinstance(match, (list, tuple))
                or len(match) == 0
                or not all(isinstance(p, str) and len(p) > 0 for p in match)
            ):
                raise ValueError(
                    f"{where} ({name}): `match` must be a non-empty list of "
                    f"non-empty strings, got {match!r}"
                )
            matchers = [_matcher(pattern, where) for pattern in match]
        options = {
            key: _coerce_option(value)
            for key, value in spec.items()
            if key not in _RESERVED_KEYS
        }
        groups.append((name, matchers, options))

    members = [[] for _ in groups]
    member_names = [[] for _ in groups]
    unmatched = []
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = None
        for index, (_, matchers, _) in enumerate(groups):
            if matchers is not None and any(m(parameter_name) for m in matchers):
                target = index
                break
        if target is None:
            target = catch_all
        if target is None:
            unmatched.append(parameter_name)
            continue
        members[target].append(parameter)
        member_names[target].append(parameter_name)

    if unmatched:
        raise ValueError(
            "trainable parameters match no optimizer_param_groups entry (add "
            f"a catch-all group without `match`): {unmatched}"
        )
    empty = [groups[i][0] for i in range(len(groups)) if len(members[i]) == 0]
    if empty:
        raise ValueError(
            f"optimizer_param_groups {empty} match no trainable parameter; check "
            "the patterns against the model's parameter names"
        )

    param_groups = [
        dict(params=members[i], name=groups[i][0], **groups[i][2])
        for i in range(len(groups))
    ]
    structure = [
        dict(name=groups[i][0], params=list(member_names[i]))
        for i in range(len(groups))
    ]
    return param_groups, structure


def describe_structure_mismatch(saved, current) -> Optional[str]:
    """``None`` if two group layouts are equal, else a short description."""
    if saved == current:
        return None

    def summary(structure):
        if structure is None:
            return "no optimizer_param_groups"
        return ", ".join(f"{g['name']}({len(g['params'])})" for g in structure)

    detail = ""
    if saved is not None and current is not None:
        for old, new in zip(saved, current):
            if old != new:
                missing = sorted(set(old["params"]) - set(new["params"]))[:3]
                extra = sorted(set(new["params"]) - set(old["params"]))[:3]
                detail = (
                    f"; first differing group {old['name']!r} -> {new['name']!r}, "
                    f"only in checkpoint {missing}, only in current {extra}"
                )
                break
    return f"checkpoint [{summary(saved)}], current [{summary(current)}]{detail}"


def check_param_group_options(optimizer: torch.optim.Optimizer, specs: List[dict]):
    """Reject per-group options that the optimizer does not know (typos)."""
    known = set(optimizer.defaults)
    for index, spec in enumerate(specs):
        unknown = sorted(set(spec) - set(_RESERVED_KEYS) - known)
        if unknown:
            raise ValueError(
                f"optimizer_param_groups[{index}] ({spec.get('name')}): unknown "
                f"option(s) {unknown} for {type(optimizer).__name__}; known "
                f"options are {sorted(known)}"
            )


def resolve_param_group_index(
    optimizer: torch.optim.Optimizer, selector: Union[int, str]
) -> int:
    """Index of the param group selected by position or by ``name``."""
    groups = optimizer.param_groups
    if isinstance(selector, bool):
        raise ValueError(f"invalid optimizer param group selector {selector!r}")
    if isinstance(selector, int):
        if not -len(groups) <= selector < len(groups):
            raise ValueError(
                f"optimizer param group index {selector} is out of range for "
                f"{len(groups)} group(s)"
            )
        return selector % len(groups)
    if isinstance(selector, str):
        names = [group.get("name", None) for group in groups]
        if selector not in names:
            raise ValueError(
                f"no optimizer param group named {selector!r}; groups are {names}"
            )
        return names.index(selector)
    raise ValueError(f"invalid optimizer param group selector {selector!r}")
