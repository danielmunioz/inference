import enum
import pytest
import numpy as np
import jax.numpy as jnp
import tensorflow as tf
import torch

import ivy
from graph_compiler import compile
import graph_compiler
import graph_compiler.globals as glob

glob.use_reloader = False

# arbitrary variable tracking
# ToDo: Test kwargs


def _tvp_int(start, end, int1):
    # backend op with tvp as arg
    lns = ivy.linspace(start, end, int1)
    # op with tvp
    int2 = int1 + int1 * 2
    return lns, int1, int2


@pytest.mark.parametrize("fw", ["tensorflow", "jax", "torch", "numpy"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_int(
    fw,
    array_caching,
):
    ivy.set_backend(fw)
    # raw values
    start = ivy.array(0, dtype="float32")
    end = ivy.array(100, dtype="float32")
    c_int, nc_int = 10, 20
    # compile
    fname = "arbitrary_variable_tracking_{}".format(array_caching)
    ## int
    graph = compile(_tvp_int, array_caching=array_caching, args=(start, end, nc_int))
    # value test
    nc_ret = _tvp_int(start, end, c_int)
    c_ret = graph(start, end, c_int)
    assert len(nc_ret[0]) == c_int
    assert len(c_ret[0]) == c_int
    assert np.allclose(nc_ret[0].data, c_ret[0])
    assert c_ret[1] != nc_int
    assert nc_ret[1] == c_ret[1]
    assert nc_ret[2] == c_ret[2]
    assert type(c_ret[2]) == int


def _tvp_int2(a, x):
    return a[x, ..., 1:x:2]


@pytest.mark.parametrize("fw", ["tensorflow", "jax", "torch", "numpy"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_slices(
    fw,
    array_caching,
):
    ivy.set_backend(fw)
    # raw values
    start = ivy.array(0, dtype="float32")
    end = ivy.array(100, dtype="float32")
    c_int, nc_int = 10, 20
    # compile
    fname = "arbitrary_variable_tracking_{}".format(array_caching)
    ## int in slice
    a = ivy.random_normal(shape=(25, 25))
    graph = compile(_tvp_int2, array_caching=array_caching, args=(a, nc_int))
    # value test
    nc_ret = _tvp_int2(a, c_int)
    c_ret = graph(a, c_int)
    assert len(nc_ret) == len(c_ret)
    assert np.allclose(nc_ret.data, c_ret)


def _tvp_float(float1):
    # also input in output
    return float1


@pytest.mark.parametrize("fw", ["tensorflow", "jax", "torch", "numpy"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_float(
    fw,
    array_caching,
):
    ivy.set_backend(fw)
    # raw values
    c_float, nc_float = 1.5, 2.5
    # compile
    fname = "arbitrary_variable_tracking_{}".format(array_caching)
    ## float
    graph = compile(_tvp_float, array_caching=array_caching, args=(nc_float,))
    # value test
    nc_ret = _tvp_float(c_float)
    c_ret = graph(c_float)
    assert nc_ret == c_ret
    assert type(c_ret) == float


def _tvp_list(list1):
    # inplace op
    list1.append(3)
    # indexing and setting
    list1[1] = list1[0]
    return list1


@pytest.mark.parametrize("fw", ["tensorflow", "jax", "torch", "numpy"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_list(
    fw,
    array_caching,
):
    ivy.set_backend(fw)
    # raw values
    c_list1, nc_list = [1, 2], [3, 4]
    c_list2 = [1, 2]
    # compile
    fname = "arbitrary_variable_tracking_{}".format(array_caching)
    ## list
    graph = compile(_tvp_list, array_caching=array_caching, args=(nc_list,))
    assert nc_list == [3, 4]
    # value test
    nc_ret = _tvp_list(c_list1)
    c_ret = graph(c_list2)
    assert nc_ret == c_ret
    assert c_list1 == c_list2
    assert type(c_ret) == list


def _tvp_list_unpacking(x):
    # simple unpacking using __iter__
    l, m, n = x

    # asterisk unpacking
    y = ivy.native_array([l, m, n])
    z = ivy.native_array([*x])
    return y, z


@pytest.mark.parametrize("fw", ["tensorflow", "jax", "torch", "numpy"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_list_unpacking(
    fw,
    array_caching,
):
    ivy.set_backend(fw)
    # config
    x1 = [3.4, 2.2, 5.6]
    x2 = [1.2, 0.6, 9.2]
    # compile
    fname = "list_unpacking_{}".format(array_caching)
    graph = compile(_tvp_list_unpacking, array_caching=array_caching, args=(x1,))
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # type test
    assert callable(graph)
    # value test
    nc_ret1, nc_ret2 = _tvp_list_unpacking(x2)
    c_ret1, c_ret2 = graph(x2)
    assert np.allclose(nc_ret1, c_ret1)
    assert np.allclose(nc_ret2, c_ret2)


def _tvp_raw_methods(list1, float1):
    a = graph_compiler.len(list1)
    b = graph_compiler.int(float1)
    return a, b


@pytest.mark.parametrize("fw", ["tensorflow", "jax", "torch", "numpy"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_raw_methods_replacement(
    fw,
    array_caching,
):
    ivy.set_backend(fw)
    # raw values
    c_float, nc_float = 1.5, 2.5
    c_list1, nc_list = [1, 2], [3, 4]
    c_list2 = [1, 2]
    # compile
    fname = "arbitrary_variable_tracking_{}".format(array_caching)
    # list2
    graph = compile(
        _tvp_raw_methods, array_caching=array_caching, args=(nc_list, nc_float)
    )
    assert nc_list == [3, 4]
    # value test
    nc_ret = _tvp_raw_methods(c_list1, c_float)
    c_ret = graph(c_list2, c_float)
    assert nc_ret == tuple(c_ret)
    assert type(c_ret[1]) == int
    assert c_ret[1] == int(c_float)


def _tvp_tuple(tuple1):
    # getting
    t1 = tuple1[0]
    return t1


@pytest.mark.parametrize("fw", ["tensorflow", "jax", "torch", "numpy"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_tuple(
    fw,
    array_caching,
):
    ivy.set_backend(fw)
    # raw values
    c_tuple, nc_tuple = (1, 2), (3, 4)
    # compile
    fname = "arbitrary_variable_tracking_{}".format(array_caching)
    ## tuple
    graph = compile(_tvp_tuple, array_caching=array_caching, args=(nc_tuple,))
    # value test
    nc_ret = _tvp_tuple(c_tuple)
    c_ret = graph(c_tuple)
    assert nc_ret == c_ret
    assert type(c_ret) == int


def _tvp_tuple_unpacking(x):
    shape = x.shape

    # simple unpacking using __iter__
    i, j, k = shape

    # asterisk unpacking
    x = ivy.ones(shape=shape)
    y = ivy.ones(shape=[*shape])
    z = ivy.ones(shape=(i, j, k))
    return x, y, z


@pytest.mark.parametrize("fw", ["tensorflow", "jax", "torch", "numpy"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_tuple_unpacking(
    fw,
    array_caching,
):
    ivy.set_backend(fw)
    # config
    x1 = ivy.array([[[1.0], [2.0]], [[1.0], [2.0]]])
    x2 = ivy.array([[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]])
    # compile
    fname = "tuple_unpacking_{}".format(array_caching)
    graph = compile(
        _tvp_tuple_unpacking,
        array_caching=array_caching,
        args=(x1,),
        to="ivy",
    )
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # type test
    assert callable(graph)
    # value test
    nc_ret1, nc_ret2, nc_ret3 = _tvp_tuple_unpacking(x2)
    c_ret1, c_ret2, c_ret3 = graph(x2)
    assert np.allclose(nc_ret1, c_ret1)
    assert np.allclose(nc_ret2, c_ret2)
    assert np.allclose(nc_ret3, c_ret3)


def _tvp_str_unpacking(string):
    # simple unpacking using __iter__
    i, j, k, l, m = string

    # asterisk unpacking
    x = [*string]
    y = i + j + k + l + m
    z = [char for char in string]

    return x, y, z


@pytest.mark.parametrize("fw", ["tensorflow", "jax", "torch", "numpy"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_str_unpacking(
    fw,
    array_caching,
):
    ivy.set_backend(fw)
    # config
    x1 = "unify"
    x2 = "unite"
    # compile
    fname = "str_unpacking_{}".format(array_caching)
    graph = compile(_tvp_str_unpacking, array_caching=array_caching, args=(x1,))
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # type test
    assert callable(graph)
    # value test
    nc_ret1, nc_ret2, nc_ret3 = _tvp_str_unpacking(x2)
    c_ret1, c_ret2, c_ret3 = graph(x2)
    assert nc_ret1 == c_ret1
    assert nc_ret2 == c_ret2
    assert nc_ret3 == c_ret3


def _tvp_dict(str1, dict1):
    # setting and terminal function without return
    dict1["test"] = str1.upper()
    return None


@pytest.mark.parametrize("fw", ["tensorflow", "jax", "torch", "numpy"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_dict(
    fw,
    array_caching,
):
    ivy.set_backend(fw)
    # raw values
    c_str, nc_str = "comp", "not_comp"
    c_dict1, nc_dict = {"a": 1}, {"b": 2}
    c_dict2 = {"a": 1}
    # compile
    fname = "arbitrary_variable_tracking_{}".format(array_caching)
    ## dict
    graph = compile(_tvp_dict, array_caching=array_caching, args=(nc_str, nc_dict))
    # value test
    nc_ret = _tvp_dict(c_str, c_dict1)
    c_ret = graph(c_str, c_dict2)
    assert c_dict2["test"] == c_str.upper()
    assert c_dict1 == c_dict2


def _sum(y, z):
    return y + z


def _tvp_dict_unpacking_asterisk(d):
    # single asterisk
    keys = [*d]

    # double asterisk unpacking
    a = {"a": 1, "b": 2, **d}
    d_sum = _sum(**d)

    return a, keys, d_sum


@pytest.mark.parametrize("fw", ["tensorflow", "jax", "torch", "numpy"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_dict_unpacking_asterisk(
    fw,
    array_caching,
):
    ivy.set_backend(fw)
    # config
    d1 = {"y": 1, "z": 2}
    d2 = {"y": 10, "z": 20}
    # compile
    fname = "dict_unpacking_{}".format(array_caching)
    graph = compile(
        _tvp_dict_unpacking_asterisk, array_caching=array_caching, args=(d1,)
    )
    graph.show(
        output_connected_only=True,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # type test
    assert callable(graph)
    # value test
    nc_ret, nc_keys, nc_sum = _tvp_dict_unpacking_asterisk(d2)
    c_ret, c_keys, c_sum = graph(d2)
    assert nc_ret == c_ret
    assert nc_keys == c_keys
    assert nc_sum == c_sum


def _tvp_dict_keys(d):
    # simple unpacking using __iter__
    key1, key2 = d
    keys1 = [key1, key2]

    # keys
    keys2 = list(d.keys())

    return keys1, keys2


@pytest.mark.parametrize("fw", ["tensorflow", "jax", "torch", "numpy"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_dict_keys(
    fw,
    array_caching,
):
    ivy.set_backend(fw)
    # config
    d1 = {"y": 1, "z": 2}
    d2 = {"a": 1, "b": 2}
    # compile
    fname = "dict_unpacking_{}".format(array_caching)
    graph = compile(_tvp_dict_keys, array_caching=array_caching, args=(d1,))
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # type test
    assert callable(graph)
    # value test
    nc_keys1, nc_keys2 = _tvp_dict_keys(d2)
    c_keys1, c_keys2 = graph(d2)
    assert nc_keys1 == c_keys1
    assert nc_keys2 == c_keys2


def _tvp_dict_values(d):
    # values
    values = list(d.values())
    total = 0
    [total := total + v for v in values]

    return values, total


@pytest.mark.parametrize("fw", ["tensorflow", "jax", "torch", "numpy"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_dict_values(
    fw,
    array_caching,
):
    ivy.set_backend(fw)
    # config
    d1 = {"y": 1, "z": 2}
    d2 = {"y": 10, "z": 20}
    # compile
    fname = "dict_unpacking_{}".format(array_caching)
    graph = compile(_tvp_dict_values, array_caching=array_caching, args=(d1,))
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # type test
    assert callable(graph)
    # value test
    nc_values, nc_total = _tvp_dict_values(d2)
    c_values, c_total = graph(d2)
    assert nc_values == c_values
    assert nc_total == c_total


def _tvp_dict_items(d):
    # items
    c = {k: v for k, v in d.items()}
    return c


@pytest.mark.parametrize("fw", ["tensorflow", "jax", "torch", "numpy"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_dict_items(
    fw,
    array_caching,
):
    ivy.set_backend(fw)
    # config
    d1 = {"y": 1, "z": 2}
    d2 = {"y": 10, "z": 20}
    # compile
    fname = "dict_unpacking_{}".format(array_caching)
    graph = compile(_tvp_dict_items, array_caching=array_caching, args=(d1,))
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # type test
    assert callable(graph)
    # value test
    nc_ret = _tvp_dict_items(d2)
    c_ret = graph(d2)
    assert nc_ret == c_ret


def _tvp_int_enum(intenum1, intenum2):
    # __int__ methods
    newnum = intenum1 + 2 * intenum2
    newnum2 = 3 * intenum1 - intenum2
    # Also input in output
    return intenum1, intenum2, newnum, newnum2


@pytest.mark.parametrize("fw", ["tensorflow", "jax", "torch", "numpy"])
@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_int_enum(
    fw,
    array_caching,
):
    ivy.set_backend(fw)

    # raw values
    nc_enum = enum.IntEnum("Numbers", [("TEN", 10), ("TWENTY", 20), ("THIRTY", 30)])
    c_enum = enum.IntEnum("Numbers", [("THREE", 3), ("FOUR", 4), ("FIVE", 5)])

    # compile
    fname = f"test_compile_enum_{array_caching}_{fw}"

    ## enum
    graph = compile(
        _tvp_int_enum,
        array_caching=array_caching,
        args=(nc_enum.TEN, nc_enum.TWENTY),
    )

    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )

    # value test
    nc_inp_1, nc_inp_2, nc_val1, nc_val2 = _tvp_int_enum(c_enum.THREE, c_enum.FIVE)
    c_inp_1, c_inp_2, c_val1, c_val2 = graph(c_enum.THREE, c_enum.FIVE)

    assert nc_inp_1 == c_inp_1
    assert nc_inp_2 == c_inp_2
    assert nc_val1 == c_val1 == 13
    assert nc_val2 == c_val2 == 4


def _jax_shape(x):
    a = x.shape
    b = jnp.shape(x)
    return ivy.arange(a[0] + b[0])


@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_shape_jax(array_caching):
    ivy.set_backend("jax")
    # config
    x1 = ivy.native_array([0.0, 1.0, 2.0])
    x2 = ivy.native_array([[0.0, 1.0], [2.0, 3.0]])
    # compile
    graph = compile(
        _jax_shape,
        array_caching=array_caching,
        args=(x1,),
    )
    graph.show()
    # type test
    assert callable(graph)
    # value test
    nc_ret = _jax_shape(x2)
    c_ret = graph(x2)
    assert np.allclose(nc_ret, c_ret)


def _np_shape(x):
    a = x.shape
    b = np.shape(x)
    return ivy.arange(a[0] + b[0])


@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_shape_numpy(array_caching):
    ivy.set_backend("numpy")
    # config
    x1 = ivy.native_array([0.0, 1.0, 2.0])
    x2 = ivy.native_array([[0.0, 1.0], [2.0, 3.0]])
    # compile
    fname = "np_shape_{}".format(array_caching)
    graph = compile(
        _np_shape,
        array_caching=array_caching,
        args=(x1,),
    )
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # type test
    assert callable(graph)
    # value test
    nc_ret = _np_shape(x2)
    c_ret = graph(x2)
    assert np.allclose(nc_ret, c_ret)


def _torch_size(x):
    a = x.size()
    b = x.shape
    return ivy.arange(a[0] + b[0])


@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_size_torch(array_caching):
    ivy.set_backend("torch")
    # config
    x1 = ivy.native_array([0.0, 1.0, 2.0])
    x2 = ivy.native_array([[0.0, 1.0], [2.0, 3.0]])
    # compile
    fname = "torch_size_{}".format(array_caching)
    graph = compile(
        _torch_size,
        array_caching=array_caching,
        args=(x1,),
    )
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # type test
    assert callable(graph)
    # value test
    nc_ret = _torch_size(x2)
    c_ret = graph(x2)
    assert np.allclose(nc_ret.data, c_ret)


# tf shape
# ToDo: transpilation


def test_tf_shape_metaclass():
    import tensorflow as tf
    import graph_compiler.tracked_var_proxy as tvp

    # unset any previously set backends
    ivy.unset_backend()

    # unset the tf tensor shape override if it has already been overridden before
    tvp.unset_tf_tensor_shape_override()

    cls = tvp.type_to_proxy()["TensorShape"]

    assert tvp._tf_tensor_shape_override == False
    assert issubclass(cls, tf.TensorShape) == False
    assert issubclass(cls, tvp.TrackedVarProxy) == True

    # set backend to tensorflow
    ivy.set_backend("tensorflow")

    cls = tvp.type_to_proxy()["TensorShape"]
    assert tvp._tf_tensor_shape_override == True
    assert issubclass(cls, tf.TensorShape) == True
    assert issubclass(cls, tvp.TrackedVarProxy) == True


def _tf_shape(x):
    b = x.shape
    c = b[0]
    return ivy.arange(c)


@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_shape_tf(array_caching):
    ivy.set_backend("tensorflow")
    # config
    x1 = ivy.native_array([0.0, 1.0, 2.0])
    x2 = ivy.native_array([[0.0, 1.0], [2.0, 3.0]])
    # compile
    fname = "tf_shape_{}".format(array_caching)
    graph = compile(
        _tf_shape,
        array_caching=array_caching,
        args=(x1,),
    )
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # type test
    assert callable(graph)
    # value test
    nc_ret = _tf_shape(x2)
    c_ret = graph(x2)
    assert np.allclose(nc_ret.data, c_ret)


# ivy shape


def _ivy_shape(x):
    a = x.shape
    b = ivy.shape(x)
    return ivy.arange(a[0] + b[0])


@pytest.mark.parametrize("array_caching", [True, False])
def test_compile_shape_ivy(array_caching):
    ivy.set_backend("torch")
    # config
    x1 = ivy.array([0.0, 1.0, 2.0])
    x2 = ivy.array([[0.0, 1.0], [2.0, 3.0]])
    # compile
    fname = "ivy_shape_{}".format(array_caching)
    graph = compile(
        _ivy_shape,
        array_caching=array_caching,
        args=(x1,),
        to="ivy",
    )
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # type test
    assert callable(graph)
    # value test
    nc_ret = ivy.to_numpy(_ivy_shape(x2))
    c_ret = ivy.to_numpy(graph(x2))
    # fws test

    ivy.set_backend("tensorflow")
    x2 = ivy.native_array([[0.0, 1.0], [2.0, 3.0]])
    c_ret_tf = ivy.to_numpy(graph(x2))

    ivy.set_backend("torch")
    x2 = ivy.native_array([[0.0, 1.0], [2.0, 3.0]])
    c_ret_torch = ivy.to_numpy(graph(x2))

    ivy.set_backend("jax")
    x2 = ivy.native_array([[0.0, 1.0], [2.0, 3.0]])
    c_ret_jax = ivy.to_numpy(graph(x2))

    ivy.set_backend("numpy")
    x2 = ivy.native_array([[0.0, 1.0], [2.0, 3.0]])
    c_ret_numpy = ivy.to_numpy(graph(x2))

    assert np.allclose(nc_ret, c_ret)
    assert np.allclose(c_ret, c_ret_tf)
    assert np.allclose(c_ret, c_ret_torch)
    assert np.allclose(c_ret, c_ret_jax)
    assert np.allclose(c_ret, c_ret_numpy)


def _batch_keras_model(input_shape):
    input_layer = tf.keras.Input(shape=input_shape)
    X = tf.keras.layers.Flatten()(input_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=X)
    return model


@pytest.mark.parametrize("array_caching", [True, False])
def test_batch_keras(array_caching):
    ivy.set_backend("tensorflow")
    # config
    batch1, batch2 = 1, 5
    input_shape = (100, 100)
    x1 = tf.ones((batch1, *input_shape))
    x2 = tf.ones((batch2, *input_shape))
    model = _batch_keras_model(input_shape)
    # compile
    fname = "batch_keras_{}".format(array_caching)
    graph = compile(
        model,
        array_caching=array_caching,
        args=(x1,),
    )
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
    # value test
    nc_ret = model(x2)
    c_ret = graph(x2)
    assert np.allclose(nc_ret, c_ret)


def _fn_with_extra_ops(x):
    if x.shape == tf.TensorShape(None):
        pass
    return tf.add(x, x)


def test_registered_ops():
    ivy.set_backend("tensorflow")
    x = tf.constant([1.0])
    graph = compile(_fn_with_extra_ops, args=(x,))
    assert len(graph._functions) == 1


def _fn_with_inplace_tracked_var(x):
    # check __mul__ isnt flagged as inplace (as 0 is
    # in args and output which always has same id)
    a = torch.tensor([1.0, 2.0, 3.0])
    b = (a.shape[0] - 1) * 0
    if torch.__version__ < "1.11.0":
        b = torch.tensor(b)
    return torch.multiply(b, x)


def test_inplace_tracked_var():
    ivy.set_torch_backend()
    x = torch.tensor([2.0, 3.0])
    compile(_fn_with_inplace_tracked_var, args=(x,))


# ToDo: add test for show
"""
    graph.show(
        output_connected_only=False,
        # fname=fname + ".html",  # uncomment this to save the graph locally
    )
"""
