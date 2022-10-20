import sys
import typing
from enum import EnumMeta
from typing import Any
from typing import Type
from typing import TypeVar
from typing import Union

import attr

from avalon.agent.common.params import Params

ParamType = TypeVar("ParamType", bound=Params)


def strip_optional(field: Type) -> Type:
    # Optional is an alias for Union[x, None].
    # Return x if this is indeed an Optional, otherwise return field unchanged.
    origin = typing.get_origin(field)
    args = typing.get_args(field)
    if origin is Union and type(None) in args and len(args) == 2:
        assert args[1] == type(None)
        return typing.cast(Type, args[0])
    return field


def cast_argument_to_paramater_type(key: str, value: str, params: Any) -> Any:
    """Looks up the key in `params` and casts `value` to the type-hinted type of that parameter."""
    fields = attr.fields_dict(type(params))
    assert key in fields, f"could not find {key} in param object {params}."
    dtype = fields[key].type
    if dtype is None:
        raise ValueError("got none dtype - not sure what this means")
    dtype = strip_optional(dtype)
    if type(dtype) == type:
        if dtype == bool:
            if value.lower() == "false":
                return False
                # out[arg] = False
            elif value.lower() == "true":
                return True
                # out[arg] = True
            else:
                raise ValueError(f"could not parse {value} to bool")
        else:
            return dtype(value)
    elif type(dtype) == EnumMeta:
        return dtype[value]
    elif dtype == list[str]:
        # use comma-separated list with no spaces
        return value.split(",")
    elif dtype == tuple[str, ...]:
        # use comma-separated list with no spaces
        return tuple(value.split(","))
    elif dtype == tuple[int, ...]:
        # use comma-separated list with no spaces
        return tuple(value.split(","))
    else:
        assert type(dtype) == str, (key, dtype, type(dtype))
        # We got a forward type declaration, either from an explicit quote-wrapped type, or
        # `from __future__ import annotations`, or by default in future python.
        # In this case dtype will be a string, eg "int" instead of the actual type.
        # In theory attrs.resolve_types or typing.get_type_hints could be used to resolve these, but this breaks
        # if the class has a type defined with a TYPE_CHECKING guard (https://github.com/python/cpython/issues/87629).
        # Using `eval` is one way around this (and suggested in PEP563) but feels unsafe.
        # Probably better to just handle each case manually, eg `if dtype == "str": ...`.
        raise NotImplementedError


def recursive_fill_nested_argument(key: str, value: str, params: Any) -> Any:
    key_parts = key.split(".")
    if len(key_parts) == 1:
        key = key_parts[0]
        value = cast_argument_to_paramater_type(key, value, params)
        return attr.evolve(params, **{key: value})
    else:
        current_key = key_parts[0]
        child_params = params.__getattribute__(current_key)
        updated_params = recursive_fill_nested_argument(".".join(key_parts[1:]), value, child_params)
        return attr.evolve(params, **{current_key: updated_params})


def parse_args(params: ParamType) -> ParamType:
    """Parse args of the form `test.py --arg1 1 --arg2.child_arg value`.

    Args with part1.part2 are interpreted as args into nested params."""
    args = [x.strip() for x in " ".join(sys.argv[1:]).split("--") if x != ""]
    for arg in args:
        key, value = arg.split(" ")
        params = recursive_fill_nested_argument(key, value, params)
    return params
