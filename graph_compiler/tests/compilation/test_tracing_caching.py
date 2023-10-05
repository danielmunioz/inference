# global
import os
import time
import urllib
import ivy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch

# local
from graph_compiler import compile

import simple_math_in_ivy as simple_math_in_ivy

# Tests #
# ------#

test_func = """
import ivy

def double_nested(x):
    z = x + 2
    return z


def nested(x):
    w = x**2
    x = double_nested(w)
    return x


def ivy_fn(x, y):
    x = nested(x)
    return ivy.add(x, y)


"""

test_func_2 = """
import ivy


def double_nested(x):
    z = x + 2
    return z


def nested(x):
    w = x**2
    x = double_nested(w)
    return x


def ivy_fn(x, y):
    print('hello')
    x = nested(x)

    return ivy.add(x, y)


"""


def test_compile_with_caching():
    x = ivy.array([1, 3])
    y = ivy.array([1, 2])

    f = open("caching_test.py", "w")
    f.write(test_func)
    f.close()

    from caching_test import ivy_fn

    comp_fn_orig = compile(ivy_fn, args=(x, y), graph_caching=True)

    comp_fn_orig(x, y)
    os.remove("caching_test.py")

    f = open("caching_test.py", "w")
    f.write(test_func)
    f.close()

    from caching_test import ivy_fn

    comp_fn_new = compile(ivy_fn, args=(x, y), graph_caching=True)
    os.remove("caching_test.py")

    c_orig = ivy.to_numpy(comp_fn_orig(x, y))
    c_new = ivy.to_numpy(comp_fn_new(x, y))
    assert np.allclose(c_orig, c_new)


def _functional_fn(x):
    y = ivy.mean(x, keepdims=True)
    z = ivy.mean(x, keepdims=True)
    f = ivy.mean(y, keepdims=True)
    k = ivy.cos(z)
    m = ivy.sin(f)
    o = ivy.tan(y)
    return ivy.concat([k, m, o], axis=-1)


@pytest.mark.parametrize("x_raw", [[1]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_ivy_caching(x_raw, dtype, dev, verbose=False):
    ivy.set_backend("torch")
    x = ivy.array(x_raw, dtype=dtype, device=dev)

    start_uncached = time.time()
    graph_uncached = compile(_functional_fn, args=(x,), graph_caching=True)
    end_uncached = time.time()
    runtime_uncached = end_uncached - start_uncached

    start_cached = time.time()
    graph_cached = compile(_functional_fn, args=(x,), graph_caching=True)
    end_cached = time.time()
    runtime_cached = end_cached - start_cached

    if verbose:
        print(type(graph_uncached))
        print(graph_uncached(x))
        print(type(graph_cached))
        print(graph_cached(x))
        print(f"Compilation time for first run: {runtime_uncached}")
        print(f"Compilation time for second (cached) run: {runtime_cached}")

    uncached_return = ivy.to_numpy(graph_uncached(x))
    cached_return = ivy.to_numpy(graph_cached(x))

    assert np.allclose(uncached_return, cached_return)


def test_kornia_caching(verbose=False):
    import torch
    from demos.kornia.functions import dilate_edges, img

    ivy.set_backend("torch")
    img_input = torch.randn_like(img)

    start_uncached = time.time()
    graph_uncached = compile(dilate_edges, args=(img_input,), graph_caching=True)
    end_uncached = time.time()

    start_cached = time.time()
    graph_cached = compile(dilate_edges, args=(img_input,), graph_caching=True)
    end_cached = time.time()

    if verbose:
        print(f"Compilation time for first run: {end_uncached-start_uncached}")
        print(f"Compilation time for second (cached) run: {end_cached-start_cached}")

    uncached_return = ivy.to_numpy(graph_uncached(img_input))
    cached_return = ivy.to_numpy(graph_cached(img_input))

    assert np.allclose(uncached_return, cached_return)


def _detach_div_fn(x):
    return x + (ivy.array([1.0]) / ivy.array([2.0]))


@pytest.mark.parametrize("x_raw", [[1]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_compile_w_detached_divide_caching(x_raw, dtype, dev):
    # smoke test
    x = ivy.array(x_raw, dtype=dtype, device=dev)
    graph = compile(_detach_div_fn, args=(x,), graph_caching=True)
    graph_cached = compile(_detach_div_fn, args=(x,), graph_caching=True)
    # value test
    nc_return = ivy.to_numpy(_detach_div_fn(x))
    c_return = ivy.to_numpy(graph(x))
    c_c_return = ivy.to_numpy(graph_cached(x))
    assert np.allclose(nc_return, c_return)
    assert np.allclose(c_return, c_c_return)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("image_dims", [[224, 224]])
def test_resnet_18_imagenet_caching(
    batch_size,
    image_dims,
    dev,
):
    ivy.set_backend("torch")
    with ivy.DefaultDevice(dev):
        try:
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", "resnet18", pretrained=True
            )
        except urllib.error.URLError:
            pytest.skip()
        x0 = ivy.random_uniform(
            low=0.0,
            high=1.0,
            shape=[batch_size] + [3] + image_dims,
            dtype=torch.float32,
            device=dev,
        )
        x1 = ivy.random_uniform(
            low=0.0,
            high=1.0,
            shape=[batch_size] + [3] + image_dims,
            dtype=torch.float32,
            device=dev,
        )
        ret0_nc = model(x0)
        ret1_nc = model(x1)
        assert not np.allclose(ivy.to_numpy(ret0_nc), ivy.to_numpy(ret1_nc))
        comp_network = compile(model, args=(x0,), graph_caching=True)
        comp_network.show(
            # fname="resnet_18_imagenet.html",  # uncomment this to save the graph locally
        )
        ret0_c = comp_network(x0)
        ret1_c = comp_network(x1)
        assert not np.allclose(ivy.to_numpy(ret0_c), ivy.to_numpy(ret1_c))
        assert np.allclose(ivy.to_numpy(ret0_nc), ivy.to_numpy(ret0_c))
        assert np.allclose(ivy.to_numpy(ret1_nc), ivy.to_numpy(ret1_c))


def test_torch_RoFormer_caching():
    from transformers import RoFormerModel

    ivy.set_backend("torch")

    model = RoFormerModel.from_pretrained("junnyu/roformer_chinese_base")
    inputs = {
        "input_ids": torch.tensor(
            [[101, 47582, 117, 6052, 35735, 5992, 35712, 48564, 102]]
        ),
        "token_type_ids": torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]]),
    }
    non_compiled_ret = model(**inputs).last_hidden_state.detach()

    compiled_graph = compile(model, kwargs=inputs, graph_caching=True)
    compiled_ret = compiled_graph(**inputs).last_hidden_state.detach()
    assert np.allclose(compiled_ret, non_compiled_ret)

    compiled_graph_cached = compile(model, kwargs=inputs, graph_caching=True)
    compiled_ret_cached = compiled_graph_cached(**inputs).last_hidden_state.detach()
    assert np.allclose(compiled_ret_cached, non_compiled_ret)


def test_torch_convnext_caching():
    from transformers import ConvNextModel

    model = ConvNextModel.from_pretrained("facebook/convnext-tiny-224")
    x1 = torch.rand(size=(1, 3, 224, 224), dtype=torch.float32)
    ivy.set_backend("torch")

    non_compiled_ret = ivy.to_numpy(model(x1).last_hidden_state)

    compiled_graph = compile(model.__call__, args=(x1,), graph_caching=True)
    compiled_ret = ivy.to_numpy(compiled_graph(x1).last_hidden_state)
    assert np.allclose(compiled_ret, non_compiled_ret)

    compiled_graph_cached = compile(model.__call__, args=(x1,), graph_caching=True)
    compiled_ret_cached = ivy.to_numpy(compiled_graph_cached(x1).last_hidden_state)
    assert np.allclose(compiled_ret_cached, non_compiled_ret)


def test_flax_BEiT_caching():
    from transformers import FlaxBeitModel
    import jax

    ivy.set_backend("jax")

    model = FlaxBeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")

    key1 = jax.numpy.array([2, 4], dtype="uint32")
    key2 = jax.numpy.array([90, 10], dtype="uint32")
    x1 = jax.random.uniform(key1, shape=(1, 3, 224, 224), dtype="float32")

    compiled_fn = compile(model, args=(x1,), graph_caching=True)
    compiled_fn_cached = compile(model, args=(x1,), graph_caching=True)

    non_compiled_ret = model(x1).last_hidden_state
    compiled_ret = compiled_fn(x1).last_hidden_state
    compiled_cached_ret = compiled_fn_cached(x1).last_hidden_state

    assert np.allclose(non_compiled_ret, compiled_ret)
    assert np.allclose(compiled_cached_ret, non_compiled_ret)


class TestTorchLayer(torch.nn.Module):
    def __init__(self):
        self.act = torch.nn.functional.gelu

    def forward(self, x):
        return self.act(x)


def test_torch_layer_caching():
    ivy.set_torch_backend()
    if ivy.current_backend_str() != "torch":
        pytest.skip()
    model = TestTorchLayer()
    x = torch.rand(size=(1, 3, 2, 2), dtype=torch.float32)
    graph = compile(model.forward, args=(x,), graph_caching=True)
    original = ivy.to_numpy(model.forward(x))
    compiled_ret = ivy.to_numpy(graph(x))
    assert np.allclose(original, compiled_ret)


def _fn_w_copyto(x, y):
    z = np.add(y, y)
    np.copyto(x, z)
    return x


def test_numpy_copyto_caching():
    if ivy.current_backend_str() != "numpy":
        pytest.skip()
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    graph = compile(_fn_w_copyto, args=(x, y), graph_caching=True)
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    graph_cached = compile(_fn_w_copyto, args=(x, y), graph_caching=True)
    x = np.array([1.0, 2.0])
    compiled_ret = graph(x, y)
    x = np.array([1.0, 2.0])
    compiled_ret_cached = graph_cached(x, y)
    x = np.array([1.0, 2.0])
    non_compiled_ret = _fn_w_copyto(x, y)
    assert np.allclose(compiled_ret, non_compiled_ret)
    assert np.allclose(compiled_ret_cached, non_compiled_ret)


def _tf_numpy_function(x):
    w = np.square(x)
    y = tf.convert_to_tensor(w)
    z = tf.add(y, x)
    return z.numpy() ** 2


def test_compile_tf_with_numpy_caching():
    if ivy.current_backend_str() != "tensorflow":
        pytest.skip()
    x = np.array([1.0, 2.0])
    graph = compile(_tf_numpy_function, args=(x,), with_numpy=True, graph_caching=True)
    graph_cached = compile(
        _tf_numpy_function, args=(x,), with_numpy=True, graph_caching=True
    )
    y = np.array([2.0, 3.0])
    non_compiled_ret = _tf_numpy_function(y)
    compiled_ret = graph(y)
    compiled_ret_cached = graph_cached(y)
    assert np.allclose(non_compiled_ret, compiled_ret)
    assert np.allclose(non_compiled_ret, compiled_ret_cached)


@jax.jit
def _jitted_fn(x):
    return jnp.add(x, x)


def _some_jax_fn(x):
    y = _jitted_fn(x)
    return jnp.sin(y)


def test_compiling_with_jax_jit_caching():
    if ivy.current_backend_str() != "jax":
        pytest.skip()
    x = ivy.native_array([1.0, 2.0])
    y = ivy.native_array([2.0, 3.0])
    non_compiled_ret = ivy.to_numpy(_some_jax_fn(y))
    graph = compile(_some_jax_fn, args=(x,), graph_caching=True)
    compiled_ret = ivy.to_numpy(graph(y))
    graph_cached = compile(_some_jax_fn, args=(x,), graph_caching=True)
    compiled_ret_cached = ivy.to_numpy(graph_cached(y))
    assert np.allclose(compiled_ret, non_compiled_ret)
    assert np.allclose(compiled_ret_cached, non_compiled_ret)


class StatefulInArg:
    def __init__(self, dev):
        self._state = ivy.native_array([0.0], device=dev)

    def add_one(self):
        self._state += 1

    @property
    def state(self):
        return self._state


def _stateful_in_arg_method(x, sia):
    x = x + 1
    sia.add_one()
    return x


@pytest.mark.parametrize("x_raw", [([0])])
@pytest.mark.parametrize("dtype", ["float32"])
def test_compile_w_stateful_in_args_caching(x_raw, dtype, dev):
    # as tensors
    x = ivy.native_array(x_raw, dtype=dtype, device=dev)
    sia = StatefulInArg(dev)
    # compile
    fname = "w_stateful_in_args"
    graph = compile(
        _stateful_in_arg_method,
        arg_stateful_idxs=[[1]],
        args=(x, sia),
        graph_caching=True,
    )

    # value test
    ret = graph(x, sia)
    assert ret == 1

    graph_cached = compile(
        _stateful_in_arg_method,
        arg_stateful_idxs=[[1]],
        args=(x, sia),
        graph_caching=True,
    )

    # value test
    ret_cached = graph_cached(x, sia)
    assert ret_cached == 1
