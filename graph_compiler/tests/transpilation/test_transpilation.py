# global
import jax

jax.config.update("jax_enable_x64", True)
import jaxlib
import torch
import pytest
import numpy as np
import jax.numpy as jnp
import tensorflow as tf
from packaging import version
import re

# local
import ivy
import simple_math_in_torch as simple_math_in_torch
from graph_compiler.conversion import _to_ivy_device
from graph_compiler.graph import LazyGraph
from transpiler.transpiler import transpile, unify
import graph_compiler.globals as glob

try:
    import functorch
except:
    functorch = None

glob.use_reloader = False

# Helpers #
# ------- #


def jax_fn(x):
    a = jnp.dot(x, x)
    b = jnp.mean(x)
    return x * a + b


# Tests #
# ----- #


@pytest.mark.parametrize("to", ["jax", "torch", "tensorflow", "numpy", "paddle"])
def test_transpile_to_native(to):
    x = jnp.array([1.0, 2.0, 3.0], dtype="float32")
    transpiled_graph = transpile(jax_fn, source="jax", to=to, args=(x,))

    original_ret = jax_fn(x)
    ivy.set_backend(to)
    x = ivy.native_array([1.0, 2.0, 3.0])
    transpiled_ret = ivy.to_numpy(transpiled_graph(x))
    ivy.previous_backend()
    assert np.allclose(original_ret, transpiled_ret)


@pytest.mark.parametrize("to", ["jax", "torch", "tensorflow", "numpy", "paddle"])
def test_transpile_lazy_with_target_args(to):
    transpiled_graph = transpile(jax_fn, source="jax", to=to)

    x = jnp.array([1.0, 2.0, 3.0])
    original_ret = jax_fn(x)

    ivy.set_backend(to)
    x = ivy.native_array([1.0, 2.0, 3.0])
    transpiled_ret = ivy.to_numpy(transpiled_graph(x))
    ivy.previous_backend()

    assert np.allclose(original_ret, transpiled_ret)


def test_transpile_to_ivy():
    ivy.set_backend("jax")

    x = jnp.array([1.0, 2.0, 3.0])
    ivy_graph = transpile(jax_fn, source="jax", to="ivy", args=(x,))

    JaxArray = (
        jax.Array
        if version.parse(jax.__version__) >= version.parse("0.4.1")
        else jaxlib.xla_extension.DeviceArray
    )
    JaxArray = (
        (JaxArray, jax.interpreters.xla._DeviceArray)
        if version.parse(jax.__version__) <= version.parse("0.4.8")
        else (JaxArray,)
    )

    original_ret = jax_fn(x)
    jax_ret = ivy_graph(x)
    assert isinstance(jax_ret.data, JaxArray)

    ivy.set_backend("torch")
    torch_ret = ivy_graph(x)
    assert isinstance(torch_ret.data, torch.Tensor)
    torch_ret = ivy.to_numpy(torch_ret)

    ivy.set_backend("tensorflow")
    tf_ret = ivy_graph(x)
    assert isinstance(tf_ret.data, tf.Tensor)

    ivy.set_backend("numpy")
    np_ret = ivy_graph(x)
    assert isinstance(np_ret.data, np.ndarray)

    assert np.allclose(original_ret, jax_ret.data)
    assert np.allclose(original_ret, torch_ret)
    assert np.allclose(original_ret, tf_ret.data)
    assert np.allclose(original_ret, np_ret.data)


@pytest.mark.parametrize("to", ["jax", "tensorflow", "numpy", "torch", "paddle", "ivy"])
def test_transpile_torch_module(to):
    ivy.set_backend("torch")

    x = ivy.native_array([1.0])
    y = ivy.native_array([2.0])
    original_add_result = ivy.to_numpy(simple_math_in_torch.add(x, y))
    original_sub_result = ivy.to_numpy(simple_math_in_torch.sub(x, y))
    original_mul_result = ivy.to_numpy(simple_math_in_torch.mul(x, y))
    original_div_result = ivy.to_numpy(simple_math_in_torch.div(x, y))

    transpiled_simple_math = transpile(simple_math_in_torch, source="torch", to=to)
    if to != "ivy":
        ivy.set_backend(to)
    transpiled_add_result = ivy.to_numpy(transpiled_simple_math.add(x, y))
    transpiled_sub_result = ivy.to_numpy(transpiled_simple_math.sub(x, y))
    transpiled_mul_result = ivy.to_numpy(transpiled_simple_math.mul(x, y))
    transpiled_div_result = ivy.to_numpy(transpiled_simple_math.div(x, y))

    assert np.allclose(original_add_result, transpiled_add_result)
    assert np.allclose(original_sub_result, transpiled_sub_result)
    assert np.allclose(original_mul_result, transpiled_mul_result)
    assert np.allclose(original_div_result, transpiled_div_result)


def getitem(x):
    a = x[2]
    return a


@pytest.mark.parametrize("x_raw", [[1.0, 2.0, 3.0]])
def test_transpile_getitem(x_raw):
    ivy.set_backend("jax")
    # jax
    x = jnp.array(x_raw)
    jax_result = getitem(x)

    # tf
    transpiled_graph = transpile(getitem, source="jax", to="tensorflow")
    tf_result = transpiled_graph(x)
    assert np.allclose(jax_result, tf_result)


def at_set(x):
    a = x.at[1].set(200)
    return a


@pytest.mark.parametrize("target_fw", ["jax", "torch", "tensorflow", "numpy", "paddle"])
@pytest.mark.parametrize("x_raw", [[1.0, 2.0, 3.0]])
def test_transpile_at_set(target_fw, x_raw):
    ivy.set_backend("jax")
    # jax
    x = jnp.array(x_raw)
    jax_result = at_set(x)

    # transpiled
    transpiled_graph = transpile(at_set, source="jax", to=target_fw)
    transpiled_result = transpiled_graph(x)
    ivy.set_backend(target_fw)
    assert np.allclose(jax_result, ivy.to_numpy(transpiled_result))


def _input_in_output(x, y):
    z = x + y
    return z, y


@pytest.mark.parametrize("from_", ["tensorflow", "jax", "torch", "numpy", "paddle"])
@pytest.mark.parametrize("to", ["jax", "torch", "tensorflow", "numpy", "paddle"])
@pytest.mark.parametrize("x_raw", [[1.0]])
def test_transpile_input_in_output(from_, to, x_raw):
    if (
        from_ == "paddle"
    ):  # TODO: remove this once the paddle frontends have been created
        pytest.skip()

    ivy.set_backend(from_)
    # origin
    x = ivy.array(x_raw)
    og_a, og_b = _input_in_output(x, x * 2)
    og_a = ivy.to_numpy(og_a)
    og_b = ivy.to_numpy(og_b)

    # transpiled
    transpiled_graph = transpile(_input_in_output, source=from_, to=to)
    transpiled_result = transpiled_graph(x, x * 2)
    ivy.set_backend(to)
    assert np.allclose(og_a, ivy.to_numpy(transpiled_result[0]))
    assert np.allclose(og_b, ivy.to_numpy(transpiled_result[1]))


# ToDo: Test rest of transpilation methods
def _arbitrary_variable_tracking_int(x, int1):
    res = torch.mul(x, int1)
    int2 = int1 + int1 * 2
    return res, int1, int2


@pytest.mark.parametrize("to", ["tensorflow", "jax", "torch", "numpy", "paddle"])
def test_transpile_arbitrary_variable_tracking(to):
    ivy.set_backend("torch")
    # raw values
    x = ivy.native_array([1.0])
    c_int, nc_int = 10, 20
    # transpiled
    transpiled_graph = transpile(
        _arbitrary_variable_tracking_int,
        source="torch",
        to=to,
        args=(x, nc_int),
    )
    # value test
    nc_ret = _arbitrary_variable_tracking_int(x, c_int)
    np_nc_ret = ivy.to_numpy(nc_ret[0])
    ivy.set_backend(to)
    x = ivy.native_array([1.0])
    transpiled_result = transpiled_graph(x, c_int)
    np_trans_result = ivy.to_numpy(transpiled_result[0])

    assert np.allclose(np_nc_ret, np_trans_result)
    assert transpiled_result[1] != nc_int
    assert nc_ret[1] == transpiled_result[1]
    assert nc_ret[2] == transpiled_result[2]
    assert type(transpiled_result[2]) == int


def _op_with_scalar(x):
    z = x + 2.0
    return z


@pytest.mark.parametrize("from_", ["jax", "torch", "numpy", "tensorflow", "paddle"])
@pytest.mark.parametrize("to", ["jax", "torch", "numpy", "tensorflow", "paddle"])
def test_transpile_op_with_scalar(from_, to):
    if (
        from_ == "paddle"
    ):  # TODO: remove this once the paddle frontends have been created
        pytest.skip()

    ivy.set_backend(from_)
    # origin
    x_og = ivy.native_array([1.0])
    og_result = ivy.to_numpy(_op_with_scalar(x_og))

    # transpiled
    transpiled_graph = transpile(_op_with_scalar, source=from_, to=to)
    transpiled_result = transpiled_graph(x_og)
    ivy.set_backend(to)
    assert np.allclose(og_result, ivy.to_numpy(transpiled_result))


def _tf_dtypes(x, target_dtype):
    y = tf.cast(x, target_dtype)
    return y


@pytest.mark.parametrize("target_dtype", [tf.float32])
@pytest.mark.parametrize("to", ["torch"])
def test_transpile_dtypes_tf(target_dtype, to):
    ivy.set_backend("tensorflow")
    # raw values
    x = tf.constant([1.0])
    # transpiled
    transpiled_graph = transpile(
        _tf_dtypes,
        source="tensorflow",
        to=to,
        args=(x, target_dtype),
    )
    # value test
    nc_ret = _tf_dtypes(x, target_dtype)
    target_dtype_in_ivy = ivy.as_ivy_dtype(target_dtype)
    ivy.set_backend(to)
    x = ivy.native_array([1.0])
    transpiled_result = transpiled_graph(x, target_dtype)
    target_dtype_in_to = ivy.as_native_dtype(target_dtype_in_ivy)
    assert np.allclose(nc_ret, ivy.to_numpy(transpiled_result))
    assert transpiled_result.dtype == target_dtype_in_to


def _np_dtypes(x, target_dtype):
    y = np.array(x, dtype=target_dtype)
    return y


@pytest.mark.parametrize("target_dtype", [np.float32])
@pytest.mark.parametrize("to", ["tensorflow"])
def test_transpile_dtypes_np(target_dtype, to):
    ivy.set_backend("numpy")
    # raw values
    x = np.array([1.0])
    # transpiled
    transpiled_graph = transpile(
        _np_dtypes,
        source="numpy",
        to=to,
        args=(x, target_dtype),
    )
    transpiled_result = transpiled_graph(x, target_dtype)
    # value test
    nc_ret = _np_dtypes(x, target_dtype)
    target_dtype_in_ivy = ivy.as_ivy_dtype(target_dtype)
    ivy.set_backend(to)
    target_dtype_in_to = ivy.as_native_dtype(target_dtype_in_ivy)
    assert np.allclose(nc_ret, ivy.to_numpy(transpiled_result))
    assert transpiled_result.dtype == target_dtype_in_to


def _jax_dtypes(x, target_dtype):
    y = jnp.array(x, target_dtype)
    return y


@pytest.mark.parametrize("target_dtype", [jnp.float32])
@pytest.mark.parametrize("to", ["torch"])
def test_transpile_dtypes_jax(target_dtype, to):
    ivy.set_backend("jax")
    # raw values
    x = ivy.native_array([1.0])
    # transpiled
    transpiled_graph = transpile(
        _jax_dtypes,
        source="jax",
        to=to,
        args=(x, target_dtype),
    )
    # value test
    nc_ret = _jax_dtypes(x, target_dtype)
    target_dtype_in_ivy = ivy.as_ivy_dtype(target_dtype)
    ivy.set_backend(to)
    x = ivy.native_array([1.0])
    transpiled_result = transpiled_graph(x, target_dtype)
    target_dtype_in_to = ivy.as_native_dtype(target_dtype_in_ivy)
    assert np.allclose(nc_ret, ivy.to_numpy(transpiled_result))
    assert transpiled_result.dtype == target_dtype_in_to


def _torch_dtypes(x, target_dtype):
    y = x.to(dtype=target_dtype)
    return y


@pytest.mark.parametrize("target_dtype", [torch.float32])
@pytest.mark.parametrize("to", ["numpy"])
def test_transpile_dtypes_torch(target_dtype, to):
    ivy.set_backend("torch")
    # raw values
    x = ivy.native_array([1.0])
    # transpiled
    transpiled_graph = transpile(
        _torch_dtypes,
        source="torch",
        to=to,
        args=(x, target_dtype),
    )
    # value test
    nc_ret = ivy.to_numpy(_torch_dtypes(x, target_dtype))
    target_dtype_in_ivy = ivy.as_ivy_dtype(target_dtype)
    ivy.set_backend(to)
    x = ivy.native_array([1.0])
    transpiled_result = transpiled_graph(x, target_dtype)
    target_dtype_in_to = ivy.as_native_dtype(target_dtype_in_ivy)
    assert np.allclose(nc_ret, ivy.to_numpy(transpiled_result))
    assert transpiled_result.dtype == target_dtype_in_to


def _jax_numpy_function(x):
    w = np.square(x)
    y = jax.numpy.asarray(w)
    z = jax.lax.add(y, y)
    return z**2


@pytest.mark.parametrize("to", ["tensorflow", "jax", "torch", "numpy", "paddle"])
def test_transpile_jax_with_numpy(to):
    x = np.array([1.0, 2.0])
    graph = transpile(
        _jax_numpy_function, source="jax", to=to, args=(x,), with_numpy=True
    )
    y = np.array([2.0, 3.0])
    source_ret = _jax_numpy_function(y)
    ivy.set_backend(to)
    y = ivy.native_array([2.0, 3.0])
    to_ret = ivy.to_numpy(graph(y))
    assert np.allclose(source_ret, to_ret)


def _torch_numpy_function(x):
    y = np.square(x) + x
    z = torch.from_numpy(y)
    return (z + z).numpy() ** 2


@pytest.mark.parametrize("to", ["tensorflow", "jax", "torch", "numpy", "paddle"])
def test_transpile_torch_with_numpy(to):
    x = np.array([1.0, 2.0])
    graph = transpile(
        _torch_numpy_function, source="torch", to=to, args=(x,), with_numpy=True
    )
    ivy.set_backend(to)
    y = ivy.native_array([2.0, 3.0], dtype="float64")
    source_ret = _torch_numpy_function(ivy.to_numpy(y))
    to_ret = graph(y)
    assert np.allclose(source_ret, to_ret)


def _tf_numpy_function(x):
    w = np.square(x)
    y = tf.convert_to_tensor(w)
    z = tf.add(y, x)
    return z.numpy() ** 2


@pytest.mark.parametrize("to", ["tensorflow", "jax", "torch", "numpy", "paddle"])
def test_transpile_tf_with_numpy(to):
    x = np.array([1.0, 2.0])
    graph = transpile(
        _tf_numpy_function, source="tensorflow", to=to, args=(x,), with_numpy=True
    )
    y = np.array([2.0, 3.0])
    source_ret = _tf_numpy_function(y)
    ivy.set_backend(to)
    y = ivy.native_array([2.0, 3.0], dtype="float64")
    to_ret = ivy.to_numpy(graph(y))
    assert np.allclose(source_ret, to_ret)


def _torch_device(x):
    # y = x.to(device="cpu")
    y = x.to(device=torch.device("cpu"))
    return y


@pytest.mark.parametrize("to", ["tensorflow", "jax", "torch", "numpy", "paddle"])
def test_transpile_torch_device(to):
    ivy.set_backend("torch")
    # raw values
    x = ivy.native_array([1.0])
    # transpiled
    transpiled_graph = transpile(
        _torch_device,
        source="torch",
        to=to,
        args=(x,),
    )
    target_ivy_device = ivy.as_ivy_dev(torch.device("cpu"))

    # value test
    nc_ret = _torch_device(x)
    ivy.set_backend(to)
    x = ivy.native_array([1.0])
    transpiled_result = transpiled_graph(x)
    if to != "numpy":
        if to == "jax":
            transpiled_device = transpiled_result.device()
        elif to == "paddle":
            transpiled_device = transpiled_result.place
        else:
            transpiled_device = transpiled_result.device
        transpiled_ivy_device = _to_ivy_device(transpiled_device)
        assert transpiled_ivy_device == target_ivy_device

    assert np.allclose(nc_ret, ivy.to_numpy(transpiled_result))


def _tf_device(x, target_device) -> tf.Tensor:
    with tf.device(target_device):  # type: ignore
        # Didn't find any other method to transfer variable from dev to another
        # TODO: Find a better method to copy to device
        y = tf.convert_to_tensor(x.numpy())
        return y


def _jax_device(x, target_device) -> jax.Array:
    return jax.device_put(x, target_device)


@pytest.mark.parametrize("target_device", jax.devices())
@pytest.mark.parametrize("to", ["tensorflow", "jax", "torch", "numpy", "paddle"])
def test_transpile_jax_device(to, target_device):
    ivy.set_backend("jax")
    # raw values
    x = ivy.native_array([1.0])
    # transpiled
    transpiled_graph = transpile(
        _jax_device,
        source="jax",
        to=to,
        args=(x, target_device),
    )
    target_ivy_device = ivy.as_ivy_dev(target_device)

    # value test
    nc_ret = _jax_device(x, target_device)
    ivy.set_backend(to)
    x = ivy.native_array([1.0])
    transpiled_result = transpiled_graph(x, target_device)
    if to != "numpy":
        if to == "jax":
            transpiled_device = transpiled_result.device()
        elif to == "paddle":
            transpiled_device = transpiled_result.place
        else:
            transpiled_device = transpiled_result.device
        transpiled_ivy_device = _to_ivy_device(transpiled_device)
        assert transpiled_ivy_device == target_ivy_device

    assert np.allclose(nc_ret, ivy.to_numpy(transpiled_result))


def _torch_view(x):
    y = x.view([1, 1, 1, 2])
    y *= 2
    return y


@pytest.mark.parametrize("to", ["tensorflow", "jax", "torch", "numpy", "paddle"])
def test_transpile_torch_view(to):
    ivy.set_backend("torch")
    # raw values
    x1 = ivy.native_array([1.0, 2.0])
    x2 = ivy.native_array([3.0, 4.0])
    # transpiled
    transpiled_graph = transpile(
        _torch_view,
        source="torch",
        to=to,
        args=(x1,),
    )
    # value test
    nc_ret = _torch_view(x2)
    ivy.set_backend(to)
    x2 = ivy.native_array([3.0, 4.0])
    transpiled_ret = transpiled_graph(x2)
    assert np.allclose(nc_ret, ivy.to_numpy(transpiled_ret))


def test_transpile_library():
    import kornia

    # transpile library lazily
    jax_kornia = transpile(kornia, source="torch", to="jax")
    # check that submodules are transpiled correctly
    assert isinstance(jax_kornia.filters.canny, LazyGraph)
    assert isinstance(jax_kornia.utils.image.image_to_tensor, LazyGraph)


def _matmul_shape(x):
    ndim = x.dim()
    if ndim < 2:
        return ()
    return (x.shape[-1],)


def _ivy_shape_transpile(x):
    y = torch.ones(*_matmul_shape(x))
    x = torch.matmul(x, y)
    x = torch.flatten(x)
    return x


def test_transpile_ivy_shape():
    ivy.set_backend("torch")

    nc_x = torch.randn((20, 20, 100))
    c_x = torch.randn((2, 2, 10))
    graph = transpile(
        _ivy_shape_transpile,
        source="torch",
        to="ivy",
        args=(nc_x,),
    )

    original_ret = _ivy_shape_transpile(c_x)

    torch_ret = graph(c_x)
    assert np.allclose(original_ret, torch_ret)
    assert np.allclose(original_ret.shape, torch_ret.shape)

    ivy.set_backend("jax")
    jax_ret = graph(c_x)
    assert np.allclose(original_ret, jax_ret)
    assert np.allclose(original_ret.shape, jax_ret.shape)

    ivy.set_backend("tensorflow")
    tf_ret = graph(c_x)
    assert np.allclose(original_ret, tf_ret)
    assert np.allclose(original_ret.shape, tf_ret.shape)

    ivy.set_backend("numpy")
    np_ret = graph(c_x)
    assert np.allclose(original_ret, np_ret)
    assert np.allclose(original_ret.shape, np_ret.shape)


def _transpile_shape(x):
    batch_size = x.shape[0]
    return batch_size


def test_transpile_shape():
    rng = jax.random.PRNGKey(0)
    nc_x = jax.random.uniform(rng, shape=(1, 224, 224, 3))
    c_x_jax = jax.random.uniform(rng, shape=(2, 224, 224, 3))

    graph = transpile(_transpile_shape, source="jax", to="torch", args=(nc_x,))

    c_x_torch = torch.rand((2, 224, 224, 3))

    original_ret = _transpile_shape(c_x_jax)
    torch_ret = graph(c_x_torch)
    assert original_ret == torch_ret


def contains_method_from_parent_dataclass(x):
    mean = np.mean(x)
    std = np.std(x)
    sub = np.subtract(x, mean)
    return np.divide(sub, std)


def test_transpile_method_from_ivy_parent_dataclass():
    # `astype` gets converted to `ivy._ArrayWithDataTypes.astype`
    # so need to account for this to change source to
    # `ivy.Array.astype` for all methods from parent dataclasses.

    x = np.random.uniform(size=10).astype(np.float32)
    graph = unify(contains_method_from_parent_dataclass, source="numpy")
    ret = contains_method_from_parent_dataclass(x)

    # numpy
    ivy.set_backend("numpy")
    np_ret = graph(x)

    # jax
    x_ = jnp.array(x)
    ivy.set_backend("jax")
    jax_ret = graph(x_)

    # torch
    x_ = torch.from_numpy(x)
    ivy.set_backend("torch")
    torch_ret = graph(x_)

    # tensorflow
    x_ = tf.constant(x)
    ivy.set_backend("tensorflow")
    tf_ret = graph(x_)

    assert np.allclose(ret, np_ret, atol=1e-4)
    assert np.allclose(ret, jax_ret, atol=1e-4)
    assert np.allclose(ret, torch_ret, atol=1e-4)
    assert np.allclose(ret, tf_ret, atol=1e-4)


def fn_with_numpy_constants(omega):
    omega_hat = torch.zeros(3, 3).type(omega.dtype).to(omega.device)
    omega_hat[0, 1] = -omega[2]
    return omega_hat


def test_transpile_with_numpy_constants():
    ivy.set_torch_backend()

    x = torch.tensor([0.02, 0.96, 0.72])
    graph = transpile(fn_with_numpy_constants, source="torch", to="numpy")

    ret = fn_with_numpy_constants(x)
    np_ret = graph(x)

    assert np.allclose(ret.numpy(), ivy.to_numpy(np_ret))


# Vmap Transpilation Tests (JAX)
def vmap_composite_scalar_jax(x, y):
    def scalar_fn(x, y):
        a = jnp.sum(x * y)
        b = jnp.mean(a)
        return jnp.cos(b) + jnp.sin(a)

    vmap_fn = jax.vmap(scalar_fn, in_axes=(0, 0))
    z = vmap_fn(x, y)
    y = jnp.mean(x) + z
    return y


def nested_vmaps_jax(x, y):
    def scalar_func1(y):
        return jax.vmap(jnp.exp)(y)

    def scalar_func2(z):
        z = jnp.expand_dims(z, axis=-1)
        return jax.vmap(scalar_func1)(jnp.sin(z))

    x = jnp.expand_dims(x, axis=-1)
    return jax.vmap(scalar_func2)(x)[:, :, 0]


@pytest.mark.parametrize(
    "fn",
    (
        vmap_composite_scalar_jax,
        nested_vmaps_jax,
    ),
)
@pytest.mark.parametrize("target_fw", ["tensorflow", "torch", "jax", "numpy"])
def test_jax_vmap_transpile(fn, target_fw):
    # initial inputs
    x = jax.random.normal(jax.random.PRNGKey(0), (5, 4))
    y = jax.random.normal(jax.random.PRNGKey(1), (5, 4))

    transpiled_graph = transpile(fn, source="jax", to=target_fw, args=(x, y))

    # ToDo: Test with inputs (5,5,4) once we start tracking shapes (ivy.Shape)
    # new test inputs
    x2 = jax.random.normal(jax.random.PRNGKey(10), (5, 4))
    y2 = jax.random.normal(jax.random.PRNGKey(15), (5, 4))
    original_ret = fn(x2, y2)

    ivy.set_backend(target_fw)
    x2 = ivy.native_array(np.array(x2))
    y2 = ivy.native_array(np.array(y2))
    transpiled_ret = transpiled_graph(x2, y2)
    tol = 1e-3
    assert np.allclose(
        original_ret, ivy.to_numpy(transpiled_ret), equal_nan=True, rtol=tol
    )


# Vmap Transpilation (PyTorch)
def vmap_composite_scalar_torch(x, y):
    def scalar_fn(x, y):
        a = torch.sum(x * y)
        b = torch.mean(a)
        return torch.cos(b) + torch.sin(a)

    try:
        vmap_fn = torch.vmap(scalar_fn, in_dims=(0, 0))
    except:
        vmap_fn = functorch.vmap(scalar_fn, in_dims=(0, 0))

    z = vmap_fn(x, y)
    y = torch.mean(x) + z
    return y


def nested_vmaps_torch(x, y):
    def scalar_func1(y):
        try:
            y = torch.vmap(torch.exp)(y)
        except:
            y = functorch.vmap(torch.exp)(y)
        return y

    def scalar_func2(z):
        z = torch.unsqueeze(z, dim=-1)
        try:
            z = torch.vmap(scalar_func1)(torch.sin(z))
        except:
            z = functorch.vmap(scalar_func1)(torch.sin(z))
        return z

    x = torch.unsqueeze(x, dim=-1)
    try:
        x = torch.vmap(scalar_func2)(x)[:, :, 0]
    except:
        x = functorch.vmap(scalar_func2)(x)[:, :, 0]
    return x


@pytest.mark.parametrize(
    "fn",
    (
        vmap_composite_scalar_torch,
        nested_vmaps_torch,
    ),
)
@pytest.mark.parametrize("target_fw", ["tensorflow", "torch", "jax", "numpy"])
def test_torch_vmap_transpile(fn, target_fw):
    ivy.set_backend("torch")
    # initial inputs
    x = torch.normal(0, 1, size=(5, 4))
    y = torch.normal(0, 1, size=(5, 4))

    transpiled_graph = transpile(fn, source="torch", to=target_fw, args=(x, y))

    # ToDo: Test with inputs (5,5,4) once we start tracking shapes (ivy.Shape)
    # new test inputs
    x2 = torch.normal(0, 1, size=(5, 4))
    y2 = torch.normal(0, 1, size=(5, 4))

    original_ret = fn(x2, y2)

    ivy.set_backend(target_fw)
    x2 = ivy.native_array(x2.numpy())
    y2 = ivy.native_array(y2.numpy())
    transpiled_ret = transpiled_graph(x2, y2)
    tol = 1e-3
    assert np.allclose(
        original_ret, ivy.to_numpy(transpiled_ret), equal_nan=True, rtol=tol
    )


# def _with_cached_native_fn(x):
#     return jax.lax.reduce_window(
#         x,
#         0.0,
#         jax.lax.add,
#         window_dimensions=(3,),
#         window_strides=(1,),
#         padding="SAME",
#         window_dilation=(3,),
#     )


# @pytest.mark.parametrize("target", ["tensorflow", "torch", "jax", "numpy", "paddle"])
# def test_transpile_with_cached_native_fn(target):
#     ivy.set_jax_backend()
#     x = jnp.array([1.0, 2.0, 3.0])
#     y = jnp.array([2.0, 3.0, 4.0])
#     original_ret = ivy.to_numpy(_with_cached_native_fn(y))
#     transpiled_graph = transpile(
#         _with_cached_native_fn, source="jax", to=target, args=(x,)
#     )
#     transpiled_ret = transpiled_graph(y)
#     assert np.allclose(ivy.to_numpy(transpiled_ret), original_ret)


def test_transpile_errors():
    message = re.escape(
        "source must be one of 'torch', 'jax', 'tensorflow', 'numpy', "
        "'paddle', 'flax', 'haiku' or 'keras'."
    )
    with pytest.raises(ivy.exceptions.IvyException, match=message):
        transpile(jax_fn, source="incorrect_source", to="tensorflow", args=(1,))

    message = re.escape(
        "to must be one of 'ivy', 'torch', 'jax', 'tensorflow', 'numpy', "
        "'paddle', 'flax', 'haiku' or 'keras'."
    )
    with pytest.raises(ivy.exceptions.IvyException, match=message):
        transpile(jax_fn, source="jax", to="incorrect_target", args=(1,))
