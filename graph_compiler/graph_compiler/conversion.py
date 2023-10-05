import numpy as np
import copy
import enum
from typing import Union, Tuple, Iterable
from collections import UserDict

import ivy
from graph_compiler.numpy_proxy import (
    NewNDArray,
    NUMPY_TO_CUSTOM,
    CUSTOM_TO_NUMPY,
)
from graph_compiler.tracked_var_proxy import (
    TrackedVarProxy,
    type_to_proxy,
    _to_ignore,
    PROXY_TO_BUILTIN_TYPES,
    PROXY_ITERATOR_TO_TYPES,
)


# Checks #
# ------ #

frontend_arrays = []


def is_frontend_array(x) -> bool:
    if not frontend_arrays:
        import ivy.functional.frontends.jax as jax_frontend
        import ivy.functional.frontends.torch as torch_frontend
        import ivy.functional.frontends.tensorflow as tf_frontend
        import ivy.functional.frontends.numpy as np_frontend

        frontend_arrays.append(torch_frontend.Tensor)
        frontend_arrays.append(np_frontend.ndarray)
        frontend_arrays.append(tf_frontend.EagerTensor)
        frontend_arrays.append(jax_frontend.DeviceArray)

    return isinstance(x, tuple(frontend_arrays))


def is_array(x, with_numpy: bool = False):
    is_from_numpy = isinstance(x, NewNDArray) or (
        isinstance(x, ivy.Array) and isinstance(x.data, NewNDArray)
    )
    return ivy.is_array(x) or is_frontend_array(x) or (is_from_numpy and with_numpy)


# Native arrays #
# ------------- #


def _to_native(x, inplace: bool = False, to_ignore: tuple = None):
    to_ignore = ivy.default(to_ignore, ())
    if isinstance(x, to_ignore):
        return x
    if isinstance(x, ivy.Array):
        return x.data
    elif is_frontend_array(x):
        return x.ivy_array.data
    elif isinstance(x, ivy.Container):
        return x.cont_map(
            lambda x_, _: _to_native(x_, inplace=inplace), inplace=inplace
        )
    return x


def to_native(x, cont_inplace: bool = False, to_ignore: tuple = None):
    return ivy.nested_map(
        x,
        lambda x: _to_native(x, inplace=cont_inplace, to_ignore=to_ignore),
        shallow=False,
    )


# Numpy proxies #
# ------------- #


def _to_ND(x):
    """Converts numpy ndarrays/scalars to instances of our custom subclasses
    ``NewNDArray``, ``NewFloat64`` etc. It does this even if the ndarray is contained
    within an ivy or frontend array.
    """
    if type(x) in NUMPY_TO_CUSTOM:
        return NUMPY_TO_CUSTOM[type(x)](x)
    elif isinstance(x, ivy.Array) and type(x.data) in NUMPY_TO_CUSTOM:
        return ivy.Array(NUMPY_TO_CUSTOM[type(x.data)](x.data))
    elif is_frontend_array(x) and type(x.ivy_array.data) in NUMPY_TO_CUSTOM:
        x.ivy_array.data = NUMPY_TO_CUSTOM[type(x.ivy_array.data)](x.ivy_array.data)
        return x
    return x


def _from_ND(x):
    """Converts instances of our custom subclasses ``NewNDArray``,
    ``NewFloat64`` etc. to numpy ndarrays/scalars. It does this even if
    the NewNDArray is contained within an ivy or frontend array.
    """
    if type(x) in CUSTOM_TO_NUMPY:
        return x.data
    elif isinstance(x, ivy.Array) and type(x.data) in CUSTOM_TO_NUMPY:
        return ivy.Array(CUSTOM_TO_NUMPY[type(x.data)](x.data))
    elif is_frontend_array(x) and type(x.ivy_array.data) in CUSTOM_TO_NUMPY:
        x.ivy_array.data = CUSTOM_TO_NUMPY[type(x.ivy_array.data)](x.ivy_array.data)
        return x
    return x


def to_custom_numpy_type(x, with_numpy):
    """
    This is used to convert any ``numpy.ndarray`` or scalar input arguments
    to the graph to our custom numpy types. It is also used in the op logging
    wrapped function to ensure that all numpy creation functions
    (``linspace``, ``random.uniform``,...) return custom types.

    We do this to ensure that during op logging all the parameters in
    the graph are our custom classes, which we need to do in order to be
    able to track instance methods (necessary because ``numpy.ndarray``
    methods can't be inplace updated with our wrapped function, whereas
    the subclasses' methods can be).
    """
    if ivy.current_backend_str() == "numpy" or with_numpy:
        return ivy.nested_map(x, _to_ND, include_derived={dict: True})
    return x


def _custom_to_numpy(args, kwargs):
    """Converts our custom subclass of numpy ndarrays
    back to numpy ndarrays/scalars.
    """
    args = ivy.nested_map(
        args,
        lambda x: np.array(x) if isinstance(x, NewNDArray) else x,
        shallow=False,
        to_ignore=_to_ignore,
    )
    kwargs = ivy.nested_map(
        kwargs,
        lambda x: np.array(x) if isinstance(x, NewNDArray) else x,
        shallow=False,
        to_ignore=_to_ignore,
    )
    return args, kwargs


# Between backends #
# ---------------- #

# ToDo: map ResourceVariable's to our own frontend variable
NATIVE_ARRAY_TO_FRONTEND = {
    # "DeviceArray": ivy.functional.frontends.jax.numpy.array,
    # "Array": ivy.functional.frontends.jax.numpy.array,
    # "ArrayImpl": ivy.functional.frontends.jax.numpy.array,
    # "Tensor": ivy.functional.frontends.torch.tensor,
    # "Parameter": ivy.functional.frontends.torch.tensor,
    # "EagerTensor": ivy.functional.frontends.tensorflow.constant,
    # "ResourceVariable": ivy.functional.frontends.tensorflow.constant,
    # "ndarray": ivy.functional.frontends.numpy.array,
}

ARRAY_TO_BACKEND = {
    "ndarray": "numpy",
    "NewNDArray": "numpy",
    "Tensor": "torch",
    "Parameter": "torch",
    "EagerTensor": "tensorflow",
    "ResourceVariable": "tensorflow",
    "DeviceArray": "jax",
    "Array": "jax",
    "ArrayImpl": "jax",
}


def native_array_to_frontend(x: Union[ivy.Array, ivy.NativeArray]):
    """Converts a native/ivy array into the corresponding ivy frontend
    array. Also the array contained within the frontend array will be
    from the globally set ivy backend.
    """
    if not NATIVE_ARRAY_TO_FRONTEND:
        import ivy.functional.frontends.jax as jax_frontend
        import ivy.functional.frontends.torch as torch_frontend
        import ivy.functional.frontends.tensorflow as tf_frontend
        import ivy.functional.frontends.numpy as np_frontend

        NATIVE_ARRAY_TO_FRONTEND["DeviceArray"] = jax_frontend.numpy.array
        NATIVE_ARRAY_TO_FRONTEND["Array"] = jax_frontend.numpy.array
        NATIVE_ARRAY_TO_FRONTEND["ArrayImpl"] = jax_frontend.numpy.array
        NATIVE_ARRAY_TO_FRONTEND["Tensor"] = torch_frontend.tensor
        NATIVE_ARRAY_TO_FRONTEND["Parameter"] = torch_frontend.tensor
        NATIVE_ARRAY_TO_FRONTEND["EagerTensor"] = tf_frontend.constant
        NATIVE_ARRAY_TO_FRONTEND["ResourceVariable"] = tf_frontend.constant
        NATIVE_ARRAY_TO_FRONTEND["ndarray"] = np_frontend.array
    x = x.data if isinstance(x, ivy.Array) else x
    if is_frontend_array(x):
        return x
    x_type = type(x).__name__
    if x_type in ARRAY_TO_BACKEND:
        try:
            x = x.detach().cpu() if x_type in ["Parameter", "Tensor"] else x
            return NATIVE_ARRAY_TO_FRONTEND[x_type](ivy.array(np.array(x)))
        except:
            return x
    return x


def array_to_new_backend(
    x: Union[ivy.Array, ivy.NativeArray],
    native: bool = False,
    with_numpy: bool = False,
) -> Union[ivy.Array, ivy.NativeArray]:
    if with_numpy and isinstance(x, np.ndarray):
        return x
    native_x = x.data if isinstance(x, ivy.Array) else x
    native_x_type = type(native_x).__name__

    # Modify native_type here since @tf.function converts tf.EagerTensor into
    # tf.Tensor when running @tf.function on a transpiled graph
    if ivy.current_backend_str() == "tensorflow":
        import tensorflow as tf

        native_x_type = (
            "EagerTensor"
            if not tf.executing_eagerly() and isinstance(native_x, tf.Tensor)
            else native_x_type
        )

    # Check for paddle first, as it shares the 'Tensor' native_x_type with torch
    if "paddle" in str(native_x.__class__) and ivy.current_backend_str() == "paddle":
        if native:
            return native_x
        else:
            return x

    # Check if the other possible backends match with the native data type
    if (
        native_x_type in ARRAY_TO_BACKEND
        and ARRAY_TO_BACKEND[native_x_type] == ivy.current_backend_str()
    ):
        if ivy.current_backend_str() == "torch":
            # torch and paddle both use 'Tensor', so we need to check that this is a torch tensor
            if "torch" in str(native_x.__class__):
                return x
            else:  # if it's actually a paddle tensor, convert to an ivy array
                ret = ivy.array(native_x.numpy())
                return ret.data if native else ret
        return x

    if is_frontend_array(x):
        return x
    if native_x_type not in ARRAY_TO_BACKEND:
        return x
    native_x = (
        native_x.detach().cpu()
        if native_x_type in ["Parameter", "Tensor"]
        else native_x
    )
    np_intermediary = np.array(native_x)
    ret = ivy.array(np_intermediary)
    return ret.data if native else ret


def nest_array_to_new_backend(
    nest, with_numpy=False, native=True, to_ignore=None, shallow=True
):
    return ivy.nested_map(
        nest,
        lambda x: array_to_new_backend(x, native=native, with_numpy=with_numpy),
        include_derived=True,
        to_ignore=to_ignore,
        shallow=shallow,
    )


# Dtypes #
# ------ #


def _to_ivy_dtype(dtype):
    if ivy.is_native_array(dtype):
        return dtype
    if isinstance(dtype, (int, float, complex, bool)):
        return dtype
    if ivy.is_native_dtype(dtype) or any(
        dtype is t for t in (bool, float, int, complex)
    ):
        return ivy.as_ivy_dtype(dtype)
    return dtype


def _to_ivy_device(device):
    # source is always present because it's the framework it was compiled on
    backend_str = ivy.current_backend_str()
    if isinstance(device, ivy.NativeDevice):
        # TODO: Numpy shouldn't have devices ideally, need to check it out
        if backend_str == "numpy" and device != "cpu":
            return device
        ivy_dev = ivy.as_ivy_dev(device)
        return ivy_dev
    return device


def _dtype_and_dev_to_ivy(x):
    _temp = _to_ivy_device(x)
    _temp = _to_ivy_dtype(_temp)
    return _temp


def _batched_tracer_to_array(obj):
    if hasattr(obj, "batch_dim"):
        return obj.val
    return obj


def _convert_to_ivy_dtype(args, kwargs, to_ivy):
    if to_ivy:
        args = ivy.nested_map(
            args,
            lambda x: ivy.as_ivy_dtype(x) if isinstance(x, ivy.NativeDtype) else x,
            shallow=False,
            to_ignore=_to_ignore,
        )
        kwargs = ivy.nested_map(
            kwargs,
            lambda x: ivy.as_ivy_dtype(x) if isinstance(x, ivy.NativeDtype) else x,
            shallow=False,
            to_ignore=_to_ignore,
        )
    return args, kwargs


# Tracked Variable Proxy #
# ---------------------- #


# ToDo: Dynamic control flow
def track(
    var: Union[int, float, bool, str, list, tuple, dict, Iterable],
    with_numpy: bool = True,
    stateful_classes: Tuple = (),
    _deepcopy=True,
) -> TrackedVarProxy:
    """Recursively wraps an arbitrary variable or an iterable of abitrary variables
    in order to track its usage and effect on a compiled function.
    Note_1: While the value of the variable will be tracked, the dynamic control flow of the compiled
    function will not be re-evaluated.
    Note_2: When wrapping bool variables, the not keyword will break tracking.

    Parameters
    ----------
    var
        Variable to track.
    with_numpy
        Whether we are compiling the graph with numpy
    stateful_classes
        Classes to be considered stateful during compilation
    _deepcopy
        Whether to perform deepcopy of var before tracking

    Returns
    -------
    Derived class of TrackedVarProxy that mirrors the behaviour of var and will be tracked during compilation.
    """
    from graph_compiler.helpers import _is_tracked_variable, _is_untrackable

    if _is_tracked_variable(var):
        return var

    ret = None
    _cls = type(var)
    if ivy.exists(var) and isinstance(var, (list, tuple)):
        # Make sure to track only those aribitrary vars which themselves don't
        # contain other wrapped or stateful classes we'll already be logging
        ret = [
            track(
                v,
                with_numpy=with_numpy,
                stateful_classes=stateful_classes,
                _deepcopy=_deepcopy,
            )
            if not _is_untrackable(
                v, with_numpy=with_numpy, stateful_classes=stateful_classes
            )
            else v
            for v in var
        ]

    if ivy.exists(var) and isinstance(var, (dict, UserDict)):
        ret = dict(
            track(
                (k, v),
                with_numpy=with_numpy,
                stateful_classes=stateful_classes,
                _deepcopy=_deepcopy,
            )
            if not _is_untrackable(
                (k, v), with_numpy=with_numpy, stateful_classes=stateful_classes
            )
            else (k, v)
            for k, v in var.items()
        )

    if ivy.exists(ret) and not _is_untrackable(
        ret, with_numpy=with_numpy, stateful_classes=stateful_classes
    ):
        return _track(ret, _cls=_cls, _deepcopy=_deepcopy)
    elif ret:
        return _cls(ret)

    return _track(var, _deepcopy=_deepcopy)


def _track(
    var: Union[int, float, bool, str, list, tuple, dict],
    _cls=None,
    _deepcopy=True,
) -> TrackedVarProxy:
    var = _cls(var) if _cls else var
    type_str = type(var).__name__

    # Need to track enums since they are subclasses of int but their type_str is not enum.Enum
    if isinstance(var, enum.IntEnum):
        type_str = "IntEnum"
    elif isinstance(var, enum.Enum):
        type_str = "Enum"

    # Retreive the type_to_proxy dict
    _type_to_proxy = type_to_proxy()

    if type_str in _type_to_proxy:
        if _deepcopy:
            var = copy.deepcopy(var)
        return _type_to_proxy[type_str](var)
    else:
        return var
        # ToDo: Raise warning? Maybe do this with an isinstance check


def untrack(var: Union[TrackedVarProxy, Iterable[TrackedVarProxy]]):
    """
    Recursively untracks tracked variables or iterable of tracked variables.
    Parameters
    ----------
    var
        Variable or an iterable of variables to recursively untrack.

    Returns
    -------
    Untracked var or iterable of vars
    """
    from graph_compiler.helpers import _is_tracked_variable

    _cls = type(var)
    _typ = None

    # If input is a TrackedVarProxy class
    if _cls in PROXY_TO_BUILTIN_TYPES:
        _typ = PROXY_TO_BUILTIN_TYPES[_cls]

    # elif input is a TrackedVarIteratorProxy class
    elif _cls in PROXY_ITERATOR_TO_TYPES:
        _typ = (
            PROXY_ITERATOR_TO_TYPES[_cls] if _cls in PROXY_ITERATOR_TO_TYPES else _typ
        )
        _typ = "ivy." + _typ if _typ and _typ == "Shape" else _typ

    cls = eval(_typ) if _typ else _cls

    var = var.get_var() if _is_tracked_variable(var) else var

    if isinstance(var, (list, tuple)):
        if type(var).__name__ not in PROXY_TO_BUILTIN_TYPES.values():
            return var
        ret = [untrack(v) for v in var]
        return (
            cls(ret)
            if not hasattr(var, "_fields")
            else cls(**dict(zip(var._fields, ret)))
        )

    if isinstance(var, (dict, UserDict)):
        if type(var).__name__ not in PROXY_TO_BUILTIN_TYPES.values():
            return var
        ret = [untrack((k, v)) for (k, v) in var.items()]
        return cls(ret)

    return var
