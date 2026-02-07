#!/usr/bin/env python3
"""
validate_str_int_dict — decorator ensuring arguments are dict[str, int].
Uses functools.wraps to preserve the docstring of the decorated function.

Usage:
    @validate_str_int_dict
    def process(data: dict[str, int]) -> int:
        return sum(data.values())
"""

import functools
from typing import Any


def _check_dict_str_int(value: Any, name: str) -> None:
    """Raise TypeError if *value* is not a dict[str, int]."""
    if not isinstance(value, dict):
        raise TypeError(
            f"Parameter '{name}': expected dict[str, int], got {type(value).__name__}"
        )
    for k, v in value.items():
        if not isinstance(k, str):
            raise TypeError(
                f"Parameter '{name}': key {k!r} is {type(k).__name__}, expected str"
            )
        # bool check MUST come before int check because bool is a subclass of int
        if isinstance(v, bool):
            raise TypeError(
                f"Parameter '{name}': value for key {k!r} is bool, expected int"
            )
        if not isinstance(v, int):
            raise TypeError(
                f"Parameter '{name}': value for key {k!r} is {type(v).__name__}, expected int"
            )


def validate_str_int_dict(func):
    """Decorator that validates every argument is a dict[str, int]."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for i, arg in enumerate(args):
            _check_dict_str_int(arg, f"arg[{i}]")
        for name, arg in kwargs.items():
            _check_dict_str_int(arg, name)
        return func(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Quick tests (run: python validate_dict.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    @validate_str_int_dict
    def total(charges: dict[str, int]) -> int:
        return sum(charges.values())

    # --- valid cases ---
    assert total({"a": 1, "b": 2}) == 3
    assert total({}) == 0
    assert total({"x": -5}) == -5

    # --- invalid: not a dict ---
    try:
        total([1, 2])
        assert False, "should have raised"
    except TypeError:
        pass

    # --- invalid: non-str key ---
    try:
        total({1: 10})
        assert False, "should have raised"
    except TypeError:
        pass

    # --- invalid: non-int value ---
    try:
        total({"a": "hello"})
        assert False, "should have raised"
    except TypeError:
        pass

    # --- invalid: float value ---
    try:
        total({"a": 1.5})
        assert False, "should have raised"
    except TypeError:
        pass

    # --- invalid: bool value (bool is subclass of int) ---
    try:
        total({"a": True})
        assert False, "should have raised"
    except TypeError:
        pass

    # --- kwargs are validated too ---
    @validate_str_int_dict
    def multi(a: dict[str, int], b: dict[str, int]) -> int:
        return sum(a.values()) + sum(b.values())

    assert multi({"x": 1}, b={"y": 2}) == 3

    try:
        multi({"x": 1}, b={"y": "bad"})
        assert False, "should have raised"
    except TypeError:
        pass

    print("All tests passed.")
