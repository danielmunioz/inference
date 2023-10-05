import pytest
import gast
import ast
import inspect
import torch
import ivy
import numpy as np
import jax

import control_flow_experimental.autograph_ivy.core.api as cf
import control_flow_experimental.autograph_ivy.pyct.parser as parser
import test_helpers as helpers

import control_flow as oldcf


def fn_to_tree(fn):
    parsed = parser.parse_entity(fn, future_features=[])[0]
    tree = gast.gast_to_ast(parsed)
    return tree


# global variables

glob = 0


def incr_global():
    global glob

    glob += 1


def ret_global():
    return glob


def test_incr_global():
    global glob
    glob = 0

    original = incr_global
    recompiled = cf.to_functional_form(incr_global)

    assert glob == 0
    original()
    assert glob == 1
    recompiled()
    assert glob == 2


def test_ret_global():
    global glob
    glob = 0

    original = ret_global
    recompiled = cf.to_functional_form(ret_global)

    assert glob == 0

    glob = 1

    assert original() == 1

    assert recompiled() == 1


# inside closures


def inside_nonlocal():
    a = 4

    def inner1():
        nonlocal a
        a += 1

    inner1()

    return a


def test_inside_nonlocal():
    original = inside_nonlocal
    recompiled = cf.to_functional_form(original)

    assert original() == 5
    assert recompiled() == 5


# outside closures


def outside_nonlocal():
    a = 0

    def inner():
        nonlocal a
        a += 1
        return a

    return inner


def outside_multi_nonlocal():
    a = 0
    b = 0

    def inner():
        nonlocal a, b
        a += 1
        b = b + 1
        return a + b

    return inner


def read_closure():
    a = 0

    def inner():
        return a

    return inner


def modify_closure():
    a = 12

    def inner1():
        nonlocal a
        a += 1

    def inner2():
        return a

    return inner1, inner2


def test_outside_nonlocal():
    original = outside_nonlocal()
    recompiled = cf.to_functional_form(original)

    assert original() == 1
    assert original() == 2
    assert recompiled() == 3


def test_outside_multi_nonlocal():
    original = outside_multi_nonlocal()
    recompiled = cf.to_functional_form(original)

    assert original() == 2
    assert original() == 4
    assert recompiled() == 6


def test_read_closure():
    original = read_closure()
    recompiled = cf.to_functional_form(original)

    assert original() == recompiled()


def test_modify_closure():
    original_incr, original_ret = modify_closure()
    recompiled_incr = cf.to_functional_form(original_incr)
    recompiled_ret = cf.to_functional_form(original_ret)

    assert original_ret() == 12
    assert recompiled_ret() == 12

    recompiled_incr()

    assert original_ret() == 13
    assert recompiled_ret() == 13

    original_incr()

    assert original_ret() == 14
    assert recompiled_ret() == 14


# nested closures


def read_nested_closure():
    a = 23

    def outer():
        def inner():
            return a

        return inner()

    return outer


def nested_nonlocal():
    a = 0

    def outer():
        def inner():
            nonlocal a
            a += 1

        inner()
        return a

    return outer


def nested_multi_nonlocal():
    a = 0
    b = 0

    def outer():
        def inner():
            nonlocal a, b
            a += 1
            b += 1

        inner()
        return a + b

    return outer


def test_read_nested_closure():
    original = read_nested_closure()
    recompiled = cf.to_functional_form(original)

    assert original() == 23
    assert recompiled() == 23


def test_nested_nonlocal():
    original = nested_nonlocal()
    recompiled = cf.to_functional_form(original)

    assert original() == 1
    assert recompiled() == 2


def test_nested_multi_nonlocal():
    original = nested_multi_nonlocal()
    recompiled = cf.to_functional_form(original)

    assert original() == 2
    assert recompiled() == 4


# decorated function


def decorator(fn):
    def dotwice(x):
        return fn(fn(x))

    return dotwice


@decorator
def add_two(x):
    return x + 1


def test_decorator_add_two():
    original = add_two
    recompiled = cf.to_functional_form(original)

    assert original(0) == 2

    assert recompiled(0) == 2


# non-global decorator


def ret_decorated_function():
    def local_decorator(fn):
        def dotwice(x):
            return fn(fn(x))

        return dotwice

    @local_decorator
    def local_add_two(x):
        return x + 1

    return local_add_two


def test_local_decorator():
    original = ret_decorated_function()
    recompiled = cf.to_functional_form(original)

    assert original(0) == 2

    assert recompiled(0) == 2


# closure decorator


def closure_decorated_function():
    def local_decorator(fn):
        def dotwice(x):
            return fn(fn(x))

        return dotwice

    def inner():
        @local_decorator
        def local_add_two(x):
            return x + 1

        return local_add_two

    return inner()


def test_closure_decorator():
    original = closure_decorated_function()
    recompiled = cf.to_functional_form(original)

    assert original(0) == 2

    assert recompiled(0) == 2


# if_else


def if_basic(x):
    if x > 0:
        global glob
        glob = 1
    else:
        pass
    return glob


def if_local(x):
    y = 0
    if x > 0:
        y = 1
    else:
        pass
    return y


def if_closure():
    x = 0

    def inner(y):
        if y > x:
            global glob
            glob = True
        else:
            pass

    return inner


def if_return(x):
    if x > 0:
        return 1
    return 0


def if_else_return(x):
    y = 0
    if x > 0:
        return 1
    else:
        y += 2
    return y


def if_else_return_nested(x, y):
    z = 0
    if x > 0:
        if y < 0:
            return 0
        else:
            z += 3
        return z
    else:
        z += 2
    return z


def test_if_basic():
    original = if_basic
    recompiled = cf.to_functional_form(original)

    assert not helpers.has_scf(fn_to_tree(cf.to_functional_form(original)))

    global glob
    glob = 0

    assert original(0) == 0
    assert original(1) == 1

    glob = 0

    assert recompiled(0) == 0
    assert recompiled(1) == 1


def test_if_local():
    original = if_local
    recompiled = cf.to_functional_form(original)

    assert not helpers.has_scf(fn_to_tree(cf.to_functional_form(original)))

    assert original(0) == 0
    assert original(1) == 1

    assert recompiled(0) == 0
    assert recompiled(1) == 1


def test_if_closure():
    original = if_closure()
    recompiled = cf.to_functional_form(original)

    assert not helpers.has_scf(fn_to_tree(cf.to_functional_form(original)))

    global glob
    glob = False

    original(1)
    assert glob == True

    glob = False

    recompiled(1)
    assert glob == True

    glob = False

    original(0)
    assert glob == False

    recompiled(0)
    assert glob == False


def test_if_return():
    original = if_return
    recompiled = cf.to_functional_form(original)

    assert not helpers.has_scf(fn_to_tree(cf.to_functional_form(original)))

    assert original(1) == 1
    assert original(0) == 0

    assert recompiled(1) == 1
    assert recompiled(0) == 0


def test_if_else_return():
    original = if_else_return
    recompiled = cf.to_functional_form(original)

    assert not helpers.has_scf(fn_to_tree(cf.to_functional_form(original)))

    assert original(1) == 1
    assert original(0) == 2

    assert recompiled(1) == 1
    assert recompiled(0) == 2


def test_if_else_return_nested():
    original = if_else_return_nested
    recompiled = cf.to_functional_form(original)

    assert not helpers.has_scf(fn_to_tree(cf.to_functional_form(original)))

    assert original(1, -1) == 0
    assert original(1, 1) == 3
    assert original(-1, 0) == 2

    assert recompiled(1, -1) == 0
    assert recompiled(1, 1) == 3
    assert recompiled(-1, 0) == 2


# multiple if_else


def if_multiple_return(n):
    if n > 0:
        return 1
    if n > 0:
        return 2
    return 3


def test_if_multiple_return():
    original = if_multiple_return
    recompiled = cf.to_functional_form(original)

    assert not helpers.has_scf(fn_to_tree(cf.to_functional_form(original)))

    assert original(1) == 1

    assert recompiled(1) == 1


# complex tests


def test_torch_convnext():
    from transformers import ConvNextModel

    model = ConvNextModel.from_pretrained("facebook/convnext-tiny-224")
    functional = cf.to_functional_form(model.__call__)
    x1 = torch.rand(size=(1, 3, 224, 224), dtype=torch.float32)

    converted_ret = ivy.to_numpy(functional(x1).last_hidden_state)
    non_converted_ret = ivy.to_numpy(model(x1).last_hidden_state)
    assert np.allclose(converted_ret, non_converted_ret)


def test_flax_BEiT():
    from transformers import FlaxBeitModel

    ivy.set_backend("jax")

    model = FlaxBeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")

    key1 = jax.numpy.array([2, 4], dtype="uint32")
    key2 = jax.numpy.array([90, 10], dtype="uint32")
    x1 = jax.random.uniform(key1, shape=(1, 3, 224, 224), dtype="float32")
    x2 = jax.random.uniform(key2, shape=(1, 3, 224, 224), dtype="float32")

    converted_fn = cf.to_functional_form(model)

    non_converted_ret_1 = model(x1).last_hidden_state
    non_converted_ret_2 = model(x2).last_hidden_state

    converted_ret_1 = converted_fn(x1).last_hidden_state
    converted_ret_2 = converted_fn(x2).last_hidden_state

    assert np.allclose(converted_ret_1, non_converted_ret_1)
    assert np.allclose(converted_ret_2, non_converted_ret_2)
    assert not np.allclose(converted_ret_1, converted_ret_2)


def test_kornia():
    import matplotlib.pyplot as plt
    from demos.kornia.functions import dilate_edges, img_np, img

    img = torch.tensor(img_np)
    converted_dilate_edges = cf.to_functional_form(dilate_edges)

    original = np.array(dilate_edges(img))
    converted_ret = np.array(converted_dilate_edges(img))

    assert np.allclose(original, converted_ret)
