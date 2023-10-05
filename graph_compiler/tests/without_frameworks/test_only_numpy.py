import ivy
import numpy as np
from graph_compiler import compile
from transpiler.transpiler import transpile


def _numpy_fn(x):
    return np.add(x, x)


def test_compile_only_numpy():
    ivy.set_numpy_backend()
    x = np.array([1.0, 2.0, 3.0])
    graph = compile(_numpy_fn, args=(x,))
    graph(x)


def test_transpile_only_numpy():
    x = np.array([1.0, 2.0, 3.0])
    graph = transpile(_numpy_fn, source="numpy", to="numpy", args=(x,))
    graph(x)
