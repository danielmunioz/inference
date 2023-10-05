"""
Reloading tests to be run directly (not using PyTest).
"""

import torch
import importlib
from functools import partial
import ivy
import numpy as np

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from graph_compiler import compile


def add_fn(x):
    y = ivy.add(x, x)
    return x + y


def test_compile_add(framework):
    ivy.set_backend(framework)

    x = ivy.array([1.0, 2.0, 3.0])
    graph = compile(add_fn, args=(x,))
    original_ret = add_fn(x)
    comp_ret = graph(x)
    assert ivy.allclose(original_ret, comp_ret)


class TestTorchLayer(torch.nn.Module):
    def __init__(self):
        self.act = torch.nn.functional.gelu

    def forward(self, x):
        return self.act(x)


def test_torch_layer():
    ivy.set_torch_backend()
    model = TestTorchLayer()
    x = torch.rand(size=(1, 3, 2, 2), dtype=torch.float32)
    graph = compile(model.forward, args=(x,))
    original = ivy.to_numpy(model.forward(x))
    compiled_ret = ivy.to_numpy(graph(x))
    assert np.allclose(original, compiled_ret)


test_compile_add("jax")
test_compile_add("torch")
test_compile_add("numpy")
test_compile_add("tensorflow")

test_torch_layer()
