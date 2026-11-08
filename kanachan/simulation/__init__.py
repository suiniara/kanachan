#!/usr/bin/env python3

def __getattr__(name):
    modules = {"Tool": ".tool", "TestModel": ".test_model",
               "test": "._simulation"}
    if name in modules:
        from importlib import import_module

        module = import_module(modules[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def simulate(*args, seed=None, **kwargs):
    """Simulate games; a uint64 seed fixes walls and seat arrangements.

    Deterministic policies are required for repeatable decisions. Sampling
    policies also depend on their PyTorch RNG and the inference schedule.
    """
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
        raise ValueError("seed must be an integer in [0, 2**64), or None.")
    from ._simulation import simulate as native_simulate

    return native_simulate(*args, seed=seed, **kwargs)


__all__ = ["Tool", "TestModel", "simulate", "test"]
