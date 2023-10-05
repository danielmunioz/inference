import torch
import pytest
import numpy as np
import jax
import gc

import ivy
from graph_compiler import compile
import simple_module

# helpers


def _get_wrapped_fns():
    gc.collect()
    ret = {}
    all_objects = gc.get_objects()
    for obj in all_objects:
        if not callable(obj):
            continue
        if "wrapped_for_compiling" in dir(obj):
            ret[obj] = True
    return ret


def check_fns_unwrapped(test):
    def new_test(*args, **kwargs):
        gc.freeze()
        test(*args, **kwargs)
        assert len(_get_wrapped_fns()) <= 1
        gc.unfreeze()

    return new_test


class TestTorchLayer(torch.nn.Module):
    def __init__(self):
        self.act = torch.nn.functional.gelu

    def forward(self, x):
        return self.act(x)


def test_torch_layer():
    ivy.set_torch_backend()
    if ivy.current_backend_str() != "torch":
        pytest.skip()
    model = TestTorchLayer()
    x = torch.rand(size=(1, 3, 2, 2), dtype=torch.float32)
    graph = compile(model.forward, args=(x,))
    original = ivy.to_numpy(model.forward(x))
    compiled_ret = ivy.to_numpy(graph(x))
    assert np.allclose(original, compiled_ret)


def test_torch_convnext():
    from transformers import ConvNextModel

    model = ConvNextModel.from_pretrained("facebook/convnext-tiny-224")
    x1 = torch.rand(size=(1, 3, 224, 224), dtype=torch.float32)
    ivy.set_backend("torch")
    compiled_graph = compile(model.__call__, args=(x1,))

    compiled_ret = ivy.to_numpy(compiled_graph(x1).last_hidden_state)
    non_compiled_ret = ivy.to_numpy(model(x1).last_hidden_state)
    assert np.allclose(compiled_ret, non_compiled_ret)


def test_flax_BEiT():
    from transformers import FlaxBeitModel

    ivy.set_backend("jax")

    model = FlaxBeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")

    key1 = jax.numpy.array([2, 4], dtype="uint32")
    key2 = jax.numpy.array([90, 10], dtype="uint32")
    x1 = jax.random.uniform(key1, shape=(1, 3, 224, 224), dtype="float32")
    x2 = jax.random.uniform(key2, shape=(1, 3, 224, 224), dtype="float32")

    compiled_fn = compile(model, args=(x1,))

    non_compiled_ret_1 = model(x1).last_hidden_state
    non_compiled_ret_2 = model(x2).last_hidden_state

    compiled_ret_1 = compiled_fn(x1).last_hidden_state
    compiled_ret_2 = compiled_fn(x2).last_hidden_state

    assert np.allclose(compiled_ret_1, non_compiled_ret_1)
    assert np.allclose(compiled_ret_2, non_compiled_ret_2)
    assert not np.allclose(compiled_ret_1, compiled_ret_2)


@check_fns_unwrapped
def test_keras_nasnet():
    import tensorflow as tf

    model = tf.keras.applications.NASNetMobile(
        input_shape=None,
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=1000,
    )

    ivy.set_backend("tensorflow")
    x = tf.random.uniform(shape=(1, 224, 224, 3))
    graph = compile(model, args=(x,))

    assert tf.nn.avg_pool in graph.list_function_frequencies(return_raw=True)


def test_flax_BERT():
    from transformers import BertTokenizer, FlaxBertModel

    ivy.set_backend("jax")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = FlaxBertModel.from_pretrained("bert-base-uncased")
    inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

    compiled_graph = compile(model, kwargs=inputs)
    compiled_ret = compiled_graph(**inputs).last_hidden_state
    non_compiled_ret = model(**inputs).last_hidden_state
    assert np.allclose(compiled_ret, non_compiled_ret)


@check_fns_unwrapped
def test_torch_kornia():
    from demos.kornia.functions import dilate_edges, img

    ivy.set_backend("torch")
    dilated_edges = dilate_edges(img)
    compiled_graph = compile(dilate_edges, args=(torch.randn_like(img),))
    dilated_edges_comp = compiled_graph(img)
    assert np.allclose(dilated_edges_comp, dilated_edges)


@check_fns_unwrapped
def test_tf_convnext():
    from transformers import TFConvNextModel
    import tensorflow as tf

    model = TFConvNextModel.from_pretrained("facebook/convnext-tiny-224")
    ivy.set_backend("tensorflow")
    x = tf.random.uniform(shape=(1, 3, 224, 224), dtype=tf.float32)
    graph = compile(model.__call__, args=(x,))

    assert tf.identity in graph.list_function_frequencies(return_raw=True)


@check_fns_unwrapped
def test_w_partial():
    x = torch.tensor([1, 1])
    y = torch.tensor([2, 2])
    ivy.set_backend("torch")
    graph = compile(simple_module.exported_add, args=(x, y))

    assert torch.add in graph.list_function_frequencies(return_raw=True)


def wrapper_func(argument_function):
    def func_to_compile(x, y):
        return argument_function(x, y)

    return func_to_compile


@check_fns_unwrapped
def test_w_closure_func():
    ivy.set_backend("torch")

    x = torch.rand((2, 3, 10, 10), dtype=torch.float32)
    y = torch.rand((2, 3, 10, 10), dtype=torch.float32)

    loss_fn = wrapper_func(torch.nn.functional.mse_loss)
    ret = loss_fn(x, y)

    graph = compile(wrapper_func(torch.nn.functional.mse_loss), args=(x, y))
    comp_ret = graph(x, y)

    assert np.allclose(ret.numpy(), comp_ret.numpy())
    assert torch.nn.functional.mse_loss in graph.list_function_frequencies(
        return_raw=True
    )
