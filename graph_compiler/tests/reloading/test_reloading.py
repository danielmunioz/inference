"""Simple tests for reloader.py"""


import torch
import importlib
from functools import partial
import pytest
import ivy

from graph_compiler.reloader import apply_and_reload
from graph_compiler import compile
import graph_compiler
import simple_module

import helper


partial_add = partial(torch.add)


def with_reloading(fn):
    def new_fn(framework, *args, **kwargs):
        use_reloader = graph_compiler.globals.use_reloader
        graph_compiler.globals.use_reloader = True
        ret = fn(framework, *args, **kwargs)
        graph_compiler.globals.use_reloader = use_reloader
        return ret

    return new_fn


def test_reload_module():
    a = torch.tensor(0)
    b = torch.tensor(1)

    simple_module.exported_add(a, b)
    assert not helper.logged

    reloaded_add = apply_and_reload(
        simple_module.exported_add, to_apply=helper.add_logging
    )
    helper.pause_logging = False

    reloaded_add(a, b)
    assert helper.logged

    importlib.reload(torch)
    importlib.reload(simple_module)
    helper.reset()


def inscript_add(*args):
    return partial_add(*args)


def test_reload_script():
    a = torch.tensor(0)
    b = torch.tensor(1)

    inscript_add(a, b)
    assert not helper.logged

    reloaded_add = apply_and_reload(inscript_add, to_apply=helper.add_logging)
    helper.pause_logging = False

    reloaded_add(a, b)

    assert helper.logged

    importlib.reload(torch)
    importlib.reload(simple_module)
    helper.reset()


def test_reload_module_callable():
    a = torch.tensor(0)
    b = torch.tensor(1)
    c = simple_module.TestCallable()

    c(a, b)
    assert not helper.logged

    reloaded_add = apply_and_reload(c, to_apply=helper.add_logging)
    helper.pause_logging = False
    reloaded_add(a, b)
    assert helper.logged

    importlib.reload(torch)
    importlib.reload(simple_module)
    helper.reset()


def test_reload_bound_method():
    a = torch.tensor(0)
    b = torch.tensor(1)
    c = simple_module.TestCallable()

    c.test_method(a, b)
    assert not helper.logged

    reloaded_add = apply_and_reload(c.test_method, to_apply=helper.add_logging)
    helper.pause_logging = False
    reloaded_add(a, b)
    assert helper.logged

    importlib.reload(torch)
    importlib.reload(simple_module)
    helper.reset()


def add_fn(x):
    y = ivy.add(x, x)
    return x + y


@pytest.mark.parametrize("framework", ["torch", "jax", "numpy", "tensorflow"])
@with_reloading
def test_compile_add(framework):
    ivy.set_backend(framework)

    x = ivy.array([1.0, 2.0, 3.0])
    graph = compile(add_fn, x)
    original_ret = add_fn(x)
    comp_ret = graph(x)
    assert ivy.allclose(original_ret, comp_ret)


@pytest.mark.parametrize("framework", ["torch"])
@with_reloading
def test_compile_callable(framework):
    ivy.set_backend(framework)

    x = ivy.array([1.0, 2.0])
    y = ivy.array([2.0, 3.0])

    c = simple_module.TestCallable()

    graph = compile(c, x, y)
    original_ret = c(x, y)
    comp_ret = graph(x, y)
    assert ivy.allclose(original_ret, comp_ret)


@pytest.mark.parametrize("framework", ["torch"])
@with_reloading
def test_compile_method(framework):
    ivy.set_backend(framework)

    x = ivy.array([1.0, 2.0])
    y = ivy.array([2.0, 3.0])

    c = simple_module.TestCallable()

    graph = compile(c.test_method, x, y)
    original_ret = c.test_method(x, y)
    comp_ret = graph(x, y)
    assert ivy.allclose(original_ret, comp_ret)
