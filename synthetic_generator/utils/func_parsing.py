"""
File name: func_parsing
Author: Fran Moreno
Last Updated: 9/3/2025
Version: 1.0
Description: TOFILL
"""
from inspect import getmembers, isfunction
from typing import Callable, Optional, Dict
from types import ModuleType


def snake2camelcase(text_snake: str, suffix: str = None) -> Optional[str]:
    """
    Takes a certain string in snake case format (foo_var) and converts it to camel case format (fooBar).
    If a suffix is given, it is interpreted as a certain pattern, so if the string does not start with
    that suffix, an error is raised. If it matches, the suffix is not included in the resulting string.

    :param text_snake: generator function name.
    :param suffix: Used as a pattern. It will be excluded from the resulting string.
    :return: function name in camel case without the initial 'get' word.
    """
    name_parts = text_snake.split('_')

    if name_parts == ['']:
        return None
        # raise ValueError("Cannot convert empty string to Camel-case format.")

    if suffix:
        if name_parts[0] != suffix:
            return None
            # raise ValueError(f"The given text should start with '{suffix}', but starts with '{name_parts[0]}'.")
        name_parts.pop(0)

    if not name_parts or not name_parts[0]:
        return None
        # raise ValueError(f"The given text is not in Snake-case format: '{text_snake}'")

    varname = name_parts.pop(0)

    varname += ''.join([i.capitalize() for i in name_parts])
    return varname


def collect_funcs_from_module(module: ModuleType, suffix: str = None, default: Callable = None) -> dict:
    """
    Inspects the given module to collect all its functions. If a suffix is given, it will be used as a pattern
    to filter the collected functions.

    This function retrieves a dictionary indexing all the collected functions with their names in Camel-case format,
    so it can be used as a function factory.

    Optionally, you can decide to add a default function, by giving it as the `default` parameter. This will ensure
    that, when working with function factories, you always have a fallback function.

    :param module: Loaded module to inspect.
    :param suffix: optional suffix to use as a pattern.
    :param default: optional Callable object to add as a default function to the collection.
    :return: dictionary indexing all the collected functions.
    """
    functions = getmembers(module, isfunction)

    collection: Dict[str, Callable] = {
        camel_name: func
        for (f_name, func) in functions
        if (camel_name := snake2camelcase(f_name, suffix=suffix)) is not None and (suffix is None or suffix in f_name)
    }

    if default:
        collection["__default__"] = default

    return collection


