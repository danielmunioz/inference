# global
import sys
import pytest
import flax
import haiku as hk
from haiku._src.module import ModuleMetaclass
import jax.numpy as jnp
import paddle
import tensorflow as tf
import torch
import numpy as np
import jax

# local
import ivy
from transpiler.transpiler import transpile, unify
from graph_compiler.conversion import (
    nest_array_to_new_backend,
    native_array_to_frontend,
    _to_ivy_dtype,
)
from graph_compiler.compiler import compile
from graph_compiler.graph import Graph, LazyGraph
import graph_compiler.globals as glob
import graph_compiler.tracked_var_proxy as tvp

glob.use_reloader = False

# overwrite ivy compiler functions as we want to test with the current compiler code,
# not with an old compiled version from whichever commit ivy_repo is pinned to
ivy.compile = compile

# test classes
from tests.module.native_modules import NATIVE_MODULES, TestIvyModule

# tests


@pytest.mark.parametrize("x_raw", [[1, 2]])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_module_show_graph(
    x_raw,
    dtype,
    array_caching,
    dev,
):
    x = ivy.array(x_raw, dtype=dtype, device=dev)
    fname = "module_show_graph_{}_{}".format(ivy.current_backend_str(), array_caching)
    ivy_module = TestIvyModule(2, 2)
    ivy_module.compile(array_caching=array_caching, args=(x,))
    ivy_module.show_graph(
        output_connected_only=True,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )


@pytest.mark.parametrize("lazy_transpile", [True, False])
@pytest.mark.parametrize("target", ["tensorflow", "jax", "torch", "paddle"])
@pytest.mark.parametrize("jax_module_type", ["haiku", "flax"])
def test_module_unify(lazy_transpile, target, jax_module_type):
    source = ivy.current_backend_str()
    # smoke test
    if source in ["numpy"]:
        # numpy does not support modules natively
        pytest.skip()
    if source != "jax" and jax_module_type == "flax":
        # only jax has variant, no need to test twice for the other backend
        pytest.skip()

    # TODO: remove this clause once the paddle frontends have been created in ivy
    if "paddle" in [source, target]:
        pytest.skip()

    raw = [[1.0, 2.0], [3.0, 4.0]]
    dummy_input = ivy.native_array(raw)

    args = None if lazy_transpile else (dummy_input,)
    native_module = NATIVE_MODULES[source]
    params_v = None
    if source == "jax":
        # jax requires parameters to be passed as arguments
        native_module = native_module[jax_module_type]
        params_v = (
            native_module.init(0, dummy_input)
            if jax_module_type == "haiku"
            else native_module.init(jax.random.PRNGKey(0), dummy_input)
        )
        ivy_module = unify(native_module, args=args, params_v=params_v)
    else:
        ivy_module = unify(native_module, args=args)

    # TODO: for now, do transpile here, later should be done in/before
    # the execute_with_gradient step
    if lazy_transpile:
        ivy_module(dummy_input)

    assert ivy.current_backend_str() == source

    # verify ivy module can be run with any backend
    ivy.set_backend(target)

    # TODO: This should happen automatically?
    # convert the weights to target
    ivy_module.v = nest_array_to_new_backend(ivy_module.v, to_ignore=tvp._to_ignore)

    test_input = ivy.native_array([[5.0, 6.0], [7.0, 8.0]])
    test_target = ivy.native_array([[9.0, 10.0], [11.0, 12.0]])

    def loss_fn(weights):
        out = ivy_module(test_input, v=weights)
        return ivy.mean((out - test_target) ** 2)

    # train
    loss_tm1 = 1e12
    lr = 1e-4

    for _ in range(5):
        loss, grads = ivy.execute_with_gradients(loss_fn, ivy_module.v)
        ivy_module.v = ivy_module.v - lr * grads
        assert loss < loss_tm1
        loss_tm1 = loss

    ivy_module.compile(args=(test_input,))

    for _ in range(5):
        loss, grads = ivy.execute_with_gradients(loss_fn, ivy_module.v)
        ivy_module.v -= lr * grads
        assert loss < loss_tm1
        loss_tm1 = loss


@pytest.mark.parametrize("array_caching", [True, False])
@pytest.mark.parametrize("lazy_compile", [True, False])
def test_module_compile(array_caching, lazy_compile, dev):
    x0 = ivy.array([[1.0, 2.0], [3.0, 4.0]])
    x1 = ivy.array([[2.0, 3.0], [4.0, 5.0]])

    ivy_module = TestIvyModule(2, 2, device=dev)
    non_compiled_return_0 = ivy.to_numpy(ivy_module(x0))
    non_compiled_return_1 = ivy.to_numpy(ivy_module(x1))

    args = None if lazy_compile else (x0,)
    ivy_module = ivy.compile(ivy_module, array_caching=array_caching, args=args)

    compiled_return_0 = ivy.to_numpy(ivy_module(x0))
    compiled_return_1 = ivy.to_numpy(ivy_module(x1))

    assert isinstance(ivy_module._module_graph, (Graph, LazyGraph))
    assert np.allclose(non_compiled_return_0, compiled_return_0)
    assert np.allclose(non_compiled_return_1, compiled_return_1)


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("lazy_transpile", [True, False])
@pytest.mark.parametrize("jax_module_type", ["haiku", "flax"])
def test_transpile_module_to_haiku(
    dtype,
    lazy_transpile,
    jax_module_type,
    dev,
):
    source = ivy.current_backend_str()
    if source == "numpy":
        # numpy does not support modules natively
        pytest.skip()

    # TODO: remove this clause once the paddle frontends have been created in ivy
    if source == "paddle":
        pytest.skip()
    if source != "jax" and jax_module_type == "flax":
        # only jax has variant, no need to test twice for the other backend
        pytest.skip()

    dummy_input = ivy.native_array([[0.1, 0.2]], dtype=dtype, device=dev)
    source_module = NATIVE_MODULES[source]
    params_v = None
    if source == "jax":
        # jax requires parameters to be passed as arguments
        source_module = source_module[jax_module_type]
        params_v = (
            source_module.init(0, dummy_input)
            if jax_module_type == "haiku"
            else source_module.init(jax.random.PRNGKey(0), dummy_input)
        )

    if lazy_transpile:
        target_module = transpile(source_module, to="haiku", params_v=params_v)
    else:
        target_module = transpile(
            source_module, to="haiku", params_v=params_v, args=(dummy_input,)
        )

    # set backend since we are now working purely in jax
    ivy.set_backend("jax")

    dummy_input = ivy.native_array([[0.1, 0.2]], dtype=dtype, device=dev)
    test_input = ivy.native_array([[0.3, 0.4]], dtype=dtype, device=dev)
    test_target = ivy.native_array([[0.5, 0.6]], dtype=dtype, device=dev)

    def forward_fn(*a, **kw):
        model = target_module()
        return model(*a, **kw)

    transpiled_module = hk.transform(forward_fn)
    params_jax = transpiled_module.init(0, dummy_input)

    def loss_func(weights, input_data, target):
        preds = transpiled_module.apply(weights, 0, input_data)
        return jnp.mean((target - preds) ** 2)

    # training loop
    def UpdateWeights(weights, gradients):
        return weights - lr * gradients

    lr = 0.001
    loss_tm1 = 1e12

    for _ in range(10):
        loss, param_grads = jax.value_and_grad(loss_func)(
            params_jax, test_input, test_target
        )
        params_jax = jax.tree_map(UpdateWeights, params_jax, param_grads)
        assert loss < loss_tm1
        loss_tm1 = loss


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("lazy_transpile", [True, False])
@pytest.mark.parametrize("jax_module_type", ["haiku", "flax"])
def test_transpile_module_to_flax(
    dtype,
    lazy_transpile,
    jax_module_type,
    dev,
):
    source = ivy.current_backend_str()
    if source == "numpy":
        # numpy does not support modules natively
        pytest.skip()
    if source != "jax" and jax_module_type == "flax":
        # only jax has variant, no need to test twice for the other backend
        pytest.skip()

    # TODO: remove this clause once the paddle frontends have been created in ivy
    if source == "paddle":
        pytest.skip()

    dummy_input = ivy.native_array([[0.1, 0.2]], dtype=dtype, device=dev)
    source_module = NATIVE_MODULES[source]
    params_v = None
    if source == "jax":
        # jax requires parameters to be passed as arguments
        source_module = source_module[jax_module_type]
        params_v = (
            source_module.init(0, dummy_input)
            if jax_module_type == "haiku"
            else source_module.init(jax.random.PRNGKey(0), dummy_input)
        )

    args = None if lazy_transpile else (dummy_input,)
    target_module = transpile(source_module, to="flax", params_v=params_v, args=args)
    dummy_input = ivy.to_numpy(dummy_input)
    # set backend since we are now working purely in jax
    ivy.set_backend("jax")

    params_jax = target_module.init(
        jax.random.PRNGKey(0), ivy.native_array(dummy_input)
    )

    test_input = ivy.native_array([[0.3, 0.4]], dtype=dtype, device=dev)
    test_target = ivy.native_array([[0.5, 0.6]], dtype=dtype, device=dev)

    def loss_func(weights, input_data, target):
        preds = target_module.apply(
            weights, input_data
        )  # , mutable=list(weights.keys()))
        return jnp.mean((target - preds) ** 2)

    # training loop
    def UpdateWeights(weights, gradients):
        return weights - lr * gradients

    lr = 0.001
    loss_tm1 = 1e12

    for _ in range(10):
        loss, param_grads = jax.value_and_grad(loss_func)(
            params_jax, test_input, test_target
        )
        params_jax = jax.tree_map(UpdateWeights, params_jax, param_grads)
        assert loss < loss_tm1
        loss_tm1 = loss


@pytest.mark.parametrize("lazy_transpile", [True, False])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("jax_module_type", ["haiku", "flax"])
def test_transpile_module_to_keras(lazy_transpile, dtype, jax_module_type, dev):
    source = ivy.current_backend_str()
    if source == "numpy":
        # numpy does not support modules natively
        pytest.skip()
    if source != "jax" and jax_module_type == "flax":
        # only jax has variant, no need to test twice for the other backend
        pytest.skip()

    # TODO: remove this clause once the paddle frontends have been created in ivy
    if source == "paddle":
        pytest.skip()

    dummy_input = ivy.native_array([[0.1, 0.2]], dtype=dtype, device=dev)
    source_module = NATIVE_MODULES[source]
    params_v = None
    if source == "jax":
        # jax requires parameters to be passed as arguments
        source_module = source_module[jax_module_type]
        params_v = (
            source_module.init(0, dummy_input)
            if jax_module_type == "haiku"
            else source_module.init(jax.random.PRNGKey(0), dummy_input)
        )
    args = None if lazy_transpile else (dummy_input,)
    target_module = transpile(
        source_module, to="tensorflow", params_v=params_v, args=args
    )

    # set backend since we are now working purely in tensorflow
    ivy.set_backend("tensorflow")

    # TODO: for now transpile here, later try to do it in forward of torch
    dummy_input = ivy.native_array([[0.1, 0.2]], dtype=dtype, device=dev)
    target_module(dummy_input)

    test_input = ivy.native_array([[0.3, 0.4]], dtype=dtype, device=dev)

    # train
    loss_tm1 = 1e12
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

    for _ in range(5):
        with tf.GradientTape() as tape:
            output = target_module(test_input, training=True)
            loss = tf.reduce_mean(output)

        grads = tape.gradient(loss, target_module.trainable_weights)
        optimizer.apply_gradients(zip(grads, target_module.trainable_weights))
        assert loss < loss_tm1
        loss_tm1 = loss


@pytest.mark.parametrize("lazy_transpile", [True, False])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("jax_module_type", ["haiku", "flax"])
def test_transpile_module_to_torch(lazy_transpile, dtype, jax_module_type, dev):
    source = ivy.current_backend_str()
    if source == "numpy":
        # numpy does not support modules natively
        pytest.skip()
    if source != "jax" and jax_module_type == "flax":
        # only jax has variant, no need to test twice for the other backend
        pytest.skip()

    # TODO: remove this clause once the paddle frontends have been created in ivy
    if source == "paddle":
        pytest.skip()

    dummy_input = ivy.native_array([[0.1, 0.2]], dtype=dtype, device=dev)
    source_module = NATIVE_MODULES[source]
    params_v = None
    if source == "jax":
        # jax requires parameters to be passed as arguments
        source_module = source_module[jax_module_type]
        params_v = (
            source_module.init(0, dummy_input)
            if jax_module_type == "haiku"
            else source_module.init(jax.random.PRNGKey(0), dummy_input)
        )
    args = None if lazy_transpile else (dummy_input,)
    target_module = transpile(source_module, to="torch", params_v=params_v, args=args)

    # set backend since we are now working purely in torch
    ivy.set_backend("torch")

    test_input = ivy.native_array([[0.3, 0.4]], dtype=dtype, device=dev)
    test_target = ivy.native_array([[0.5, 0.6]], dtype=dtype, device=dev)

    # TODO: for now transpile here, later try to do it in forward of torch
    dummy_input = ivy.native_array([[0.1, 0.2]], dtype=dtype, device=dev)
    target_module(dummy_input)

    # train
    loss_tm1 = 1e12
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(target_module.parameters(), lr=1e-3)
    for _ in range(5):
        optimizer.zero_grad()
        ret = target_module(test_input)
        loss = loss_fn(ret, test_target)
        loss.backward()
        optimizer.step()
        assert loss.item() < loss_tm1
        loss_tm1 = loss.item()


@pytest.mark.parametrize("bs_ic_oc", [([2, 3], 10, 5)])
def test_to_torch_module(bs_ic_oc):
    if ivy.current_backend_str() != "torch":
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        "float32",
    )
    y = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), output_channels),
        "float32",
    )
    x_in = ivy.to_native(x)
    target = ivy.to_native(y)
    ivy_model = TestIvyModule(input_channels, output_channels)
    torch_model = transpile(ivy_model, to="torch", args=(x,))
    optimizer = torch.optim.SGD(torch_model.parameters(), lr=1e-3)
    mae = torch.nn.L1Loss()
    loss_tm1 = 1e12

    def loss_fn():
        preds = torch_model(x_in)
        return mae(target, preds)

    for _ in range(10):
        loss = loss_fn()
        loss.backward()
        optimizer.step()

        assert loss < loss_tm1
        loss_tm1 = loss


@pytest.mark.parametrize("bs_ic_oc", [([2, 3], 10, 5)])
def test_to_keras_module(bs_ic_oc):
    if ivy.current_backend_str() != "tensorflow":
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        "float32",
    )
    y = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), output_channels),
        "float32",
    )
    ivy_model = TestIvyModule(input_channels, output_channels)
    tf_model = transpile(ivy_model, to="tensorflow", args=(x,))
    x_in = ivy.to_native(x)
    target = ivy.to_native(y)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    mae = tf.keras.losses.MeanAbsoluteError()
    loss_tm1 = 1e12

    def loss_fn():
        preds = tf_model(x_in)
        return mae(target, preds)

    for epoch in range(10):
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            tape.watch(tf_model.trainable_weights)
            loss = loss_fn()
        grads = tape.gradient(loss, tf_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, tf_model.trainable_weights))

        assert loss < loss_tm1
        loss_tm1 = loss


@pytest.mark.parametrize("bs_ic_oc", [([2, 3], 10, 5)])
@pytest.mark.parametrize("jax_module_type", ["haiku", "flax"])
def test_to_jax_module(bs_ic_oc, jax_module_type):
    if ivy.current_backend_str() != "jax":
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        "float32",
    )
    y = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), output_channels),
        "float32",
    )
    x_in = ivy.to_native(x)
    target = ivy.to_native(y)
    loss_tm1 = 1e12

    ivy_model = TestIvyModule(input_channels, output_channels)
    jax_model = transpile(ivy_model, to=jax_module_type, args=(x,))
    rng = jax.random.PRNGKey(42)
    lr = 0.001

    if jax_module_type == "haiku":

        def forward_fn(*a, **kw):
            model = jax_model()
            return model(*a, **kw)

        def MAELoss(weights, input_data, target):
            preds = model.apply(weights, rng, input_data)
            return jnp.mean(jnp.abs(target - preds))

        model = hk.transform(forward_fn)

    else:
        model = jax_model

        def MAELoss(weights, input_data, target):
            preds = model.apply(weights, input_data)
            return jnp.mean(jnp.abs(target - preds))

    params = model.init(rng, x_in)

    def UpdateWeights(weights, gradients):
        return weights - lr * gradients

    for epoch in range(10):
        loss, param_grads = jax.value_and_grad(MAELoss)(params, x_in, target)
        params = jax.tree_map(UpdateWeights, params, param_grads)

        assert loss < loss_tm1
        loss_tm1 = loss


@pytest.mark.parametrize("bs_ic_oc", [([2, 3], 10, 5)])
def test_to_paddle_module(bs_ic_oc):
    if ivy.current_backend_str() != "paddle":
        pytest.skip()
    else:  # TODO: remove this 'else' statement once the paddle frontends have been created
        pytest.skip()
    batch_shape, input_channels, output_channels = bs_ic_oc
    x = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), input_channels),
        "float32",
    )
    y = ivy.astype(
        ivy.linspace(ivy.zeros(batch_shape), ivy.ones(batch_shape), output_channels),
        "float32",
    )
    x_in = ivy.to_native(x)
    target = ivy.to_native(y)
    ivy_model = TestIvyModule(input_channels, output_channels)
    paddle_model = transpile(ivy_model, to="paddle", args=(x,))
    optimizer = paddle.optimizer.SGD(
        learning_rate=1e-3, parameters=paddle_model.parameters()
    )
    mae = paddle.nn.L1Loss()
    loss_tm1 = 1e12

    def loss_fn():
        preds = paddle_model(x_in)
        return mae(target, preds)

    for _ in range(10):
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        assert loss < loss_tm1
        loss_tm1 = loss


@pytest.mark.parametrize("lazy_transpile", [True, False])
@pytest.mark.parametrize("dtype", ["float32"])
def test_transpile_module_to_paddle(lazy_transpile, dtype, dev):
    source = ivy.current_backend_str()
    if source == "numpy":
        # numpy does not support modules natively
        pytest.skip()
    else:  # TODO: remove this 'else' statement once the paddle frontends have been created
        pytest.skip()

    dummy_input = ivy.native_array([[0.1, 0.2]], dtype=dtype, device=dev)
    source_module = NATIVE_MODULES[source]
    params_v = None
    if source == "jax":
        # jax requires parameters to be passed as arguments
        params_v = source_module.init(0, dummy_input)
    args = None if lazy_transpile else (dummy_input,)
    target_module = transpile(source_module, to="paddle", params_v=params_v, args=args)

    # set backend since we are now working purely in paddle
    ivy.set_backend("paddle")

    test_input = ivy.native_array([[0.3, 0.4]], dtype=dtype, device=dev)
    test_target = ivy.native_array([[0.5, 0.6]], dtype=dtype, device=dev)

    dummy_input = ivy.native_array([[0.1, 0.2]], dtype=dtype, device=dev)
    target_module(dummy_input)

    # train
    loss_tm1 = 1e12
    loss_fn = paddle.nn.MSELoss()
    optimizer = paddle.optimizer.SGD(
        learning_rate=1e-3, parameters=target_module.parameters()
    )
    for _ in range(5):
        optimizer.clear_grad()
        ret = target_module(test_input)
        loss = loss_fn(ret, test_target)
        loss.backward()
        optimizer.step()
        assert loss.numpy() < loss_tm1
        loss_tm1 = loss.numpy()


@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("jax_module_type", ["haiku", "flax"])
def test_transpile_module_array_types(dtype, jax_module_type, dev):
    source = ivy.current_backend_str()
    if source == "numpy":
        # numpy does not support modules natively
        pytest.skip()
    if source != "jax" and jax_module_type == "flax":
        # only jax has variants, no need to test twice on other backends
        pytest.skip()

    # TODO: remove this clause (and add paddle to this test) once the paddle frontends have been created
    if source == "paddle":
        pytest.skip()

    expected_array = source if source != "jax" else "jaxlib"
    kc = {
        "torch": "fc1/bias",
        "tensorflow": "dense/bias:0",
        "jax": {
            "haiku": "test_haiku_module|~|test_haiku_linear_1|~|linear/b",
            "flax": "params/_linear0/_linear/bias",
        },
    }[source]
    frontend_array = {
        "torch": "ivy.functional.frontends.torch.tensor.Tensor",
        "tensorflow": "ivy.functional.frontends.tensorflow.tensor.EagerTensor",
        "jax": "ivy.functional.frontends.jax.devicearray.DeviceArray",
    }[source]

    dummy_input = ivy.native_array([[0.1, 0.2]], dtype=dtype, device=dev)
    params_v = None

    source_module = NATIVE_MODULES[source]
    if source == "jax":
        source_module = source_module[jax_module_type]
        kc = kc[jax_module_type]

        # jax requires parameters to be passed as arguments
        params_v = (
            source_module.init(0, dummy_input)
            if jax_module_type == "haiku"
            else source_module.init(jax.random.PRNGKey(0), dummy_input)
        )
    args = (dummy_input,)
    kwargs = {}

    # check that v has the corresponding types at each stage of transpilation

    BACKEND_TO_MODULE_FROM_BACKEND = {
        "torch": ivy.ModuleConverters.from_torch_module,
        "jax": {
            "haiku": ivy.ModuleConverters.from_haiku_module,
            "flax": ivy.ModuleConverters.from_flax_module,
        },
        "tensorflow": ivy.ModuleConverters.from_keras_module,
    }
    fw_kwargs = {}
    if params_v is not None:
        params_key = "params_hk" if jax_module_type == "haiku" else "params_fx"
        fw_kwargs[params_key] = params_v

    module_converter = BACKEND_TO_MODULE_FROM_BACKEND[source]
    if source == "jax":
        module_converter = module_converter[jax_module_type]
    ivy_module = module_converter(
        source_module,
        instance_args=args,
        instance_kwargs=kwargs,
        **fw_kwargs,
    )

    # simulate transpilation function with intermediate asserts
    kwargs["v"] = ivy_module.v
    assert str(type(kwargs["v"][kc])).split("'")[1].split(".")[0] == expected_array
    ivy.set_backend(source)
    graph = compile(ivy_module._call, args=args, kwargs=kwargs)
    assert str(type(kwargs["v"][kc])).split("'")[1].split(".")[0] == expected_array
    kwargs["source"] = source
    kwargs["target"] = "torch"
    graph.reload_sourcecode(frontend=source)
    ivy.set_backend("torch")
    args = ivy.nested_map(
        args,
        native_array_to_frontend,
        include_derived={dict: True},
    )
    kwargs = ivy.nested_map(
        kwargs,
        native_array_to_frontend,
        include_derived={dict: True},
    )
    graph.constants = ivy.nested_map(graph.constants, native_array_to_frontend)
    graph.constants = ivy.nested_map(graph.constants, _to_ivy_dtype)
    assert str(type(kwargs["v"][kc])).split("'")[1] == frontend_array
    graph = compile(graph, args=args, kwargs=kwargs)
    assert str(type(kwargs["v"][kc])).split("'")[1] == frontend_array


class ModelRNG(hk.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        b = hk.get_parameter("b", shape=[x.shape[-1]], dtype=x.dtype, init=jnp.ones)
        x = hk.dropout(hk.next_rng_key(), 0, x)
        return x + b


def test_haiku_model_with_rng():
    if ivy.current_backend_str() != "jax":
        pytest.skip()

    def forward(images):
        model = ModelRNG()
        logits = model(images)
        return logits

    model = hk.transform(forward)
    key = jax.random.PRNGKey(42)
    dummy_input = jax.random.uniform(key, shape=(1, 512))
    params = model.init(rng=key, images=dummy_input)
    transpiled = transpile(model, to="torch", params_v=params, args=(dummy_input,))
    assert isinstance(transpiled, torch.nn.Module)


def test_transpile_module_with_converted_args_kwargs():
    if ivy.current_backend_str() != "torch":
        # minimally test torch -> tensorflow only
        pytest.skip()

    torch_args = (ivy.native_array([[0.1, 0.2]]),)
    torch_kwargs = {}
    torch_module = NATIVE_MODULES["torch"]
    tensorflow_module = transpile(
        torch_module, to="tensorflow", args=torch_args, kwargs=torch_kwargs
    )

    ivy.set_backend("tensorflow")
    tensorflow_args = nest_array_to_new_backend(torch_args, native=True)
    tensorflow_kwargs = nest_array_to_new_backend(torch_kwargs, native=True)

    ret = tensorflow_module(*tensorflow_args, **tensorflow_kwargs)
    assert isinstance(ret, tf.Tensor)


def test_transpile_flax_module_explicitly():
    if ivy.current_backend_str() != "jax":
        pytest.skip()

    model = NATIVE_MODULES["jax"]["flax"]
    key = jax.random.PRNGKey(0)
    args = (jax.random.uniform(key, shape=(1, 2)),)
    params_v = model.init(key, *args)
    transpiled = transpile(
        model, source="flax", to="torch", params_v=params_v, args=args
    )
    assert isinstance(transpiled, torch.nn.Module)


def test_transpile_to_jax_flexibility(monkeypatch):
    if ivy.current_backend_str() not in ("torch", "tensorflow"):
        pytest.skip()
    args = (ivy.native_array([[0.1, 0.2]]),)
    source_module = NATIVE_MODULES[ivy.current_backend_str()]
    haiku_module = transpile(source_module, to="jax", args=args)

    # case where haiku is not installed
    monkeypatch.setitem(sys.modules, 'haiku', None)
    flax_module = transpile(source_module, to="jax", args=args)

    # case where both haiku and flax are not installed
    monkeypatch.setitem(sys.modules, 'flax', None)
    message = "Couldn't find haiku or flax installed in the system !"
    with pytest.raises(ModuleNotFoundError, match=message):
        flax_module = transpile(source_module, to="jax", args=args)

    assert isinstance(haiku_module, ModuleMetaclass)
    assert isinstance(flax_module, flax.linen.Module)


def test_transpile_from_jax_explicitly():
    if ivy.current_backend_str() != "jax":
        pytest.skip()

    def transpile_from_jax_source(model, key, *args):
        params = model.init(key, *args)
        transpiled = transpile(
            model, source="jax", to="torch", params_v=params, args=args
        )
        return transpiled

    key = jax.random.PRNGKey(0)
    args = (jax.random.uniform(key, shape=(1, 2)),)
    # transpile from haiku
    haiku_model = NATIVE_MODULES["jax"]["haiku"]
    haiku_torch_module = transpile_from_jax_source(haiku_model, key, *args)
    # transpile from flax
    flax_model = NATIVE_MODULES["jax"]["flax"]
    flax_torch_module = transpile_from_jax_source(flax_model, key, *args)

    assert isinstance(haiku_torch_module, torch.nn.Module)
    assert isinstance(flax_torch_module, torch.nn.Module)
