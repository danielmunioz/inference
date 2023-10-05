import pytest
import torch
import urllib
import numpy as np
import tensorflow as tf
import jax
import jaxlib
import psutil
import os
import skimage

import ivy
import graph_compiler.globals as glob
from graph_compiler import compile
from graph_compiler.exchange import _convert_dict_to_graph, _convert_graph_to_dict
from transpiler.transpiler import transpile, unify
import graph_compiler.globals as glob

glob.use_reloader = False

# this dict overrides any config parameters passed to the deepmind perceiver IO tests
perceiver_config = {
    "num_blocks": 2,
    "num_self_attend_heads": 2,
    "num_self_attends_per_block": 2,
}


def test_tensorflow_DeiT():
    from transformers import TFDeiTModel

    ivy.set_backend("tensorflow")

    model = TFDeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
    x1 = tf.random.uniform((1, 3, 224, 224), dtype=tf.float32)
    x2 = tf.random.uniform((1, 3, 224, 224), dtype=tf.float32)
    non_compiled_ret_1 = model(x1).last_hidden_state
    non_compiled_ret_2 = model(x2).last_hidden_state

    compiled_graph = compile(model, args=(x1,))
    compiled_ret_1 = compiled_graph(x1).last_hidden_state
    compiled_ret_2 = compiled_graph(x2).last_hidden_state

    assert np.allclose(compiled_ret_1, non_compiled_ret_1)
    assert np.allclose(compiled_ret_2, non_compiled_ret_2)
    assert not np.allclose(compiled_ret_1, compiled_ret_2)


def test_tensorflow_mobile_ViT():
    from transformers import TFMobileViTModel

    ivy.set_backend("tensorflow")

    tf.config.run_functions_eagerly(True)

    model = TFMobileViTModel.from_pretrained("apple/mobilevit-small")
    x1 = tf.random.uniform((1, 3, 256, 256), dtype=tf.float32)
    x2 = tf.random.uniform((1, 3, 256, 256), dtype=tf.float32)
    non_compiled_ret_1 = model(x1).last_hidden_state
    non_compiled_ret_2 = model(x2).last_hidden_state

    tf.config.run_functions_eagerly(False)

    compiled_graph = compile(model, args=(x1,))
    compiled_ret_1 = compiled_graph(x1).last_hidden_state
    compiled_ret_2 = compiled_graph(x2).last_hidden_state

    assert np.allclose(compiled_ret_1, non_compiled_ret_1)
    assert np.allclose(compiled_ret_2, non_compiled_ret_2)
    assert not np.allclose(compiled_ret_1, compiled_ret_2)


def test_torch_RoFormer():
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

    compiled_graph = compile(model, kwargs=inputs)
    compiled_ret = compiled_graph(**inputs).last_hidden_state.detach()
    assert np.allclose(compiled_ret, non_compiled_ret)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("image_dims", [[224, 224]])
def test_resnet_18_imagenet(
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
        comp_network = compile(model, args=(x0,))
        comp_network.show(
            # fname="resnet_18_imagenet.html",  # uncomment this to save the graph locally
        )
        ret0_c = comp_network(x0)
        ret1_c = comp_network(x1)
        assert not np.allclose(ivy.to_numpy(ret0_c), ivy.to_numpy(ret1_c))
        assert np.allclose(ivy.to_numpy(ret0_nc), ivy.to_numpy(ret0_c))
        assert np.allclose(ivy.to_numpy(ret1_nc), ivy.to_numpy(ret1_c))


def test_perceiver_io_deepmind_exchange():
    from demos.perceiver_io.perceiver_io_deepmind.model import (
        imagenet_classify,
        img,
        noise,
    )

    ivy.set_backend("jax")
    # compile jax model
    orig_graph = compile(
        imagenet_classify,
        args=(noise,),
        kwargs={"override_encoder_config": perceiver_config},
    )

    graph_dict = _convert_graph_to_dict(orig_graph)
    new_graph = _convert_dict_to_graph(graph_dict)

    # perform inference on same image with the exchanged model
    logits, _ = imagenet_classify(img, override_encoder_config=perceiver_config)
    logits_new, _ = new_graph(img, override_encoder_config=perceiver_config)
    # test equality
    assert np.allclose(logits, logits_new, atol=0.0001)


def test_unify_deepmind_perceiver_io():
    from demos.perceiver_io.perceiver_io_deepmind.model import (
        imagenet_classify,
        img,
        noise,
    )

    ivy.set_backend("jax")

    # unify (compile to framework agnostic ivy)
    ivy_graph = unify(
        imagenet_classify,
        source="jax",
        args=(noise,),
        kwargs={"override_encoder_config": perceiver_config},
    )

    # perform inference on same image with each backend
    original_logits, _ = imagenet_classify(
        img, override_encoder_config=perceiver_config
    )
    jax_logits, _ = ivy_graph(img, override_encoder_config=perceiver_config)
    ivy.set_backend("torch")
    torch_logits, _ = ivy_graph(img, override_encoder_config=perceiver_config)
    ivy.set_backend("tensorflow")
    tf_logits, _ = ivy_graph(img, override_encoder_config=perceiver_config)
    ivy.set_backend("numpy")
    np_logits, _ = ivy_graph(img, override_encoder_config=perceiver_config)

    # test equality for all backends
    assert np.allclose(original_logits, jax_logits.data, atol=0.0001)
    assert isinstance(jax_logits.data, jax.Array)
    assert np.allclose(original_logits, torch_logits.data, atol=0.0001)
    assert isinstance(torch_logits.data, torch.Tensor)
    assert np.allclose(original_logits, tf_logits.data, atol=0.0001)
    assert isinstance(tf_logits.data, tf.Tensor)
    assert np.allclose(original_logits, np_logits.data, atol=0.0001)
    assert isinstance(np_logits.data, np.ndarray)


# @pytest.mark.parametrize("backend", ["jax", "numpy", "tensorflow", "torch"])
# def test_ivy_perceiver_io_to_ivy(backend):
#     from demos.perceiver_io.perceiver_io_ivy import PerceiverIO
#     from demos.perceiver_io.specs import PerceiverIOSpec

#     ivy.set_backend(backend)

#     spec = PerceiverIOSpec(
#         input_dim=3,
#         num_input_axes=2,
#         output_dim=10,
#         max_fourier_freq=12,
#         query_shape=[1],
#         queries_dim=256,
#         network_depth=2,
#         latents_dim=256,
#         num_self_attention_heads=2,
#         num_lat_att_per_layer=2,
#     )
#     x0 = ivy.random_uniform(shape=[1, 1, 32, 32, 3])
#     x1 = ivy.random_uniform(shape=[1, 1, 32, 32, 3])

#     net = PerceiverIO(spec=spec)
#     ivy_network = compile(net, to="ivy", include_generators=False, args=(x0,))
#     ivy_network.show()

#     original_ret = ivy.to_numpy(net(x1))
#     ivy.set_backend("tensorflow")
#     tf_ret = ivy.to_numpy(ivy_network(x1))
#     ivy.set_backend("torch")
#     torch_ret = ivy.to_numpy(ivy_network(x1))
#     ivy.set_backend("jax")
#     jax_ret = ivy.to_numpy(ivy_network(x1))
#     ivy.set_backend("numpy")
#     np_ret = ivy.to_numpy(ivy_network(x1))

#     # test equality for all backends
#     atol = 0.00001
#     assert np.allclose(original_ret, jax_ret, atol=atol)
#     assert np.allclose(original_ret, torch_ret, atol=atol)
#     assert np.allclose(original_ret, tf_ret, atol=atol)
#     assert np.allclose(original_ret, np_ret, atol=atol)


# @pytest.mark.parametrize("backend", ["jax", "numpy", "tensorflow", "torch"])
# @pytest.mark.parametrize("batch_size", [1])
# @pytest.mark.parametrize("time_dim", [1])
# @pytest.mark.parametrize("image_dims", [[32, 32]])
# @pytest.mark.parametrize("num_classes", [10])
# def test_perceiver_io_ivy(
#     backend,
#     batch_size,
#     time_dim,
#     image_dims,
#     num_classes,
# ):
#     from demos.perceiver_io.perceiver_io_ivy import PerceiverIO
#     from demos.perceiver_io.specs import PerceiverIOSpec

#     ivy.set_backend(backend)

#     spec = PerceiverIOSpec(
#         input_dim=3,
#         num_input_axes=2,
#         output_dim=num_classes,
#         max_fourier_freq=12,
#         query_shape=[1],
#         queries_dim=256,
#         network_depth=2,
#         latents_dim=256,
#         num_self_attention_heads=2,
#         num_lat_att_per_layer=2,
#     )
#     x0 = ivy.random_uniform(
#         low=0,
#         high=1,
#         shape=[batch_size, time_dim] + image_dims + [3],
#         device=ivy.default_device(),
#     )
#     x1 = ivy.random_uniform(
#         low=0,
#         high=1,
#         shape=[batch_size, time_dim] + image_dims + [3],
#         device=ivy.default_device(),
#     )
#     net = PerceiverIO(spec=spec)
#     ret0_nc = ivy.to_numpy(net(x0))
#     ret1_nc = ivy.to_numpy(net(x1))
#     assert not np.allclose(ret0_nc, ret1_nc)
#     comp_network = compile(net, args=(x0,))
#     comp_network.show(
#         # fname="perceiver_io.html",  # uncomment this to save the graph locally
#     )
#     ret0_c = ivy.to_numpy(comp_network(x0))
#     ret1_c = ivy.to_numpy(comp_network(x1))
#     assert not np.allclose(ret0_c, ret1_c)
#     assert np.allclose(ret0_nc, ret0_c)
#     assert np.allclose(ret1_nc, ret1_c)


@pytest.mark.parametrize("to", ["tensorflow", "numpy", "torch", "jax"])
def test_perceiver_io_deepmind(to):
    from demos.perceiver_io.perceiver_io_deepmind.model import (
        imagenet_classify,
        img,
        noise,
    )

    ivy.set_backend("jax")
    # transpile to new backend
    transpiled_graph = transpile(
        imagenet_classify,
        source="jax",
        to=to,
        args=(noise,),
        kwargs={"override_encoder_config": perceiver_config},
    )
    # perform inference on same image with each backend
    logits, _ = imagenet_classify(img, override_encoder_config=perceiver_config)
    ivy.set_backend(to)
    img = ivy.native_array(np.array(img))
    ivy.previous_backend()
    transpiled_logits, _ = transpiled_graph(
        img, override_encoder_config=perceiver_config
    )
    # test equality for all backends
    assert np.allclose(logits, transpiled_logits, atol=0.0001)

    # check memory consumption is as expected
    process = psutil.Process(os.getpid())
    GB_used = process.memory_info().rss / 1e9
    # ToDo: figure out why memory is compounding each test, and reduce this
    assert GB_used < 10

    # check globals are reset properly
    assert glob.logging_paused == True
    assert glob.logging_stack == list()
    assert glob.iterator_chain == list()
    assert glob.raw_id_to_weakref == dict()
    assert glob.raw_id_to_unique_id == dict()
    assert glob.dependent_ids == set()


def test_compile_numpy_with_scipy():
    ivy.set_numpy_backend()
    x = np.random.uniform(size=(256, 256))
    y = np.ones((5, 5))
    z = 0.1
    graph = compile(skimage.restoration.deconvolution.wiener, args=(x, y, z))

    original_ret = skimage.restoration.deconvolution.wiener(x, y, z)
    compiled_ret = graph(x, y, z)

    assert np.allclose(original_ret, compiled_ret)


def test_unify_vit():
    from transformers import ViTModel

    glob.use_reloader = True

    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    x = torch.rand((1, 3, 224, 224))
    ivy_graph = unify(model.__call__, source="torch", args=(x,))

    y = torch.rand((2, 3, 224, 224))
    original_ret = model(y).last_hidden_state.detach().numpy()
    ivy.set_backend("torch")
    unified_ret = ivy_graph(y).last_hidden_state.detach().numpy()
    assert np.allclose(original_ret, unified_ret, atol=1e-4)
