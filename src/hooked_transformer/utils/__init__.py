import re
from typing import Any


def get_argument_names(f):
    while hasattr(f, "__wrapped__"):
        f = f.__wrapped__
    return f.__code__.co_varnames[: f.__code__.co_argcount]


def get_deep_attr(obj: Any, attr_path: str) -> Any:
    try:
        index_regex = re.compile(r"\[(\d+)\]")
        for attr in attr_path.split("."):
            index_regex_match = index_regex.findall(attr)
            if len(index_regex_match) > 0:
                assert len(index_regex_match) == 1, "attr nested too much"
                index_str = index_regex_match[0]
                obj = getattr(obj, attr[: -(2 + len(index_str))])[int(index_str)]
            elif "[*]" in attr:
                # Handle list-like attributes
                obj = getattr(obj, attr[:-3])[0]
            else:
                obj = getattr(obj, attr)
    except RecursionError:
        raise ValueError(
            f"Recursion error while accessing attribute path: {attr_path}"
        ) from obj
    return obj
