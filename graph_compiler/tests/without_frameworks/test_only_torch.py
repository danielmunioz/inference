import ivy
import torch
from graph_compiler import compile
from transpiler.transpiler import transpile


def _torch_fn(x):
    return torch.add(x, x)


def test_compile_only_torch():
    ivy.set_torch_backend()
    x = torch.tensor([1.0, 2.0, 3.0])
    graph = compile(_torch_fn, args=(x,))
    graph(x)


def test_transpile_only_torch():
    x = torch.tensor([1.0, 2.0, 3.0])
    graph = transpile(_torch_fn, source="torch", to="torch", args=(x,))
    graph(x)
