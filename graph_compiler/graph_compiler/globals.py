import sys

logging_paused = True
use_reloader = True
logging_stack = list()
iterator_chain = list()
raw_id_to_weakref = dict()
raw_id_to_unique_id = dict()
dependent_ids = set()
returns_cache = list()
wrapped_fns = dict()
# for finding the problematic node when the frontend and
# original graphs aren't equivalent during transpiling
check_frontend_vs_original_differences = False


# versioning #
# ---------- #


def get_version(fw):
    val = sys.modules[fw].__version__ if sys.modules.get(fw, None) else None
    # do some preprocessing, like check for a + since torch adds that
    if "+" in val:
        val = val.split("+")[0]
    return val


def version_resolver(backend_version, dic):
    if isinstance(backend_version, tuple):
        backend_version = backend_version[0]
    version_parts = backend_version.split(".")
    version_tuple = []
    for part in version_parts:
        if ".dev" in part:
            part = part.split(".dev")[0]
        if ".post" in part:
            part = part.split(".post")[0]
        version_tuple.append(int(part))
    version_tuple = tuple(version_tuple)

    for key in dic.keys():
        kl = key.split(" ")
        k1 = kl[0]
        if ".dev" in k1:
            k1 = k1.split(".dev")[0]
        k1 = tuple(map(int, k1.split(".")))

        if "above" in key and k1 <= version_tuple:
            return dic[key]

        if "below" in key and k1 >= version_tuple:
            return dic[key]

        if "to" in key:
            k2 = kl[2]
            if ".dev" in k2:
                k2 = k2.split(".dev")[0]
            if ".post" in k2:
                k2 = k2.split(".post")[0]
            k2 = tuple(map(int, k2.split(".")))
            if k1 <= version_tuple <= k2:
                return dic[key]


# Wrapping #
# -------- #

# tf has many functions available in multiple modules, so the modules are ordered from most to least obscure
# to ensure the most simple path is wrapped last, and hence the one we store
MODULES_TO_WRAP = {
    "numpy": [
        "numpy.lib",
        "numpy",
        "numpy.linalg",
        "numpy.random",
        "numpy.fft",
        "numpy.polynomial",
        "scipy.fft",
        "scipy.linalg",
        "scipy.signal",
        "scipy.stats",
    ],
    "paddle": ["paddle", "paddle.linalg", "paddle.nn.functional", "paddle.fft"],
    "jax": [
        "jax",
        "jax.nn",
        "jax.lax",
        "jax.numpy",
        "jax.numpy.linalg",
        "jax.numpy.linalg.fft",
        "jax.random",
        "jax.scipy.special",
        "jax.scipy.linalg",
        "jax.scipy.signal",
        "jax.scipy.stats",
    ],
    "tensorflow": [
        "tensorflow.compat.v2.compat.v1",
        "tensorflow.compat.v1",
        "tensorflow.compat.v1.nn",
        "tensorflow.compat.v1.linalg",
        "tensorflow.compat.v1.math",
        "tensorflow.compat.v2",
        "tensorflow.compat.v2.nn",
        "tensorflow.compat.v2.linalg",
        "tensorflow.compat.v2.math",
        "tensorflow.experimental.numpy",
        "tensorflow.keras.activations",
        "tensorflow.keras.metrics",
        "tensorflow.keras.layers",
        "tensorflow.keras.losses",
        "tensorflow",
        "tensorflow.linalg",
        "tensorflow.random",
        "tensorflow.nn",
        "tensorflow.math",
        "tensorflow.signal",
        "tensorflow.image",
    ],
    "torch": [
        "torch",
        "torch.nn.functional",
        "torch.fft",
        "torch.linalg",
        "torch.signal",
        "torch.special",
        "torch.nn.utils.rnn",
    ],
    "ivy": ["ivy"],
}

CLASSES_TO_WRAP = {
    "numpy": [("numpy", "ndarray")],
    "paddle": [("paddle", "Tensor")],
    "jax": [
        ("jaxlib.xla_extension", "DeviceArray"),
        ("jaxlib.xla_extension", "ArrayImpl"),
    ],
    "tensorflow": [
        ("tensorflow._api.v2.__internal__", "EagerTensor"),
        ("tensorflow", "Tensor"),
        ("tensorflow.python.ops.resource_variable_ops", "ResourceVariable"),
        ("tensorflow", "Variable"),
    ],
    "torch": [("torch", "Tensor")],
    "ivy": [("ivy", "Array")],
}


def PRIVATE_CLASSES_TO_WRAP(fw):
    return {
        "numpy": [],
        "paddle": [],
        "jax": version_resolver(
            get_version(fw),
            {
                "0.4.7 and above": [
                    ("jax._src.numpy.array_methods", "_IndexUpdateRef"),
                    ("jax._src.numpy.array_methods", "_IndexUpdateHelper"),
                ],
                "0.4.6 and below": [
                    ("jax._src.numpy.lax_numpy", "_IndexUpdateRef"),
                    ("jax._src.numpy.lax_numpy", "_IndexUpdateHelper"),
                ],
            },
        ),
        "tensorflow": [],
        "torch": [],
        "ivy": [],
    }[fw]


FUNCTIONS_ATTRS_NOT_TO_WRAP = {
    "numpy": ["format_float_positional", "dragon4_positional"],
    "paddle": [],
    "jax": [
        "pjit",
        "_single_device_array_to_np_array",
        "__array__",
        "get_backend",
        "tree_flatten",
        "tree_unflatten",
        "canonicalize_platform",
        "backends",
        "devices",
        "device",
        "device_buffer",
        "platform",
        "clone",
        "block_host_until_ready",
        "block_until_ready",
        "copy_to_device",
        "copy_to_host_async",
        "_copy_single_device_array_to_host_async",
        "copy_to_remote_device",
        "delete",
        "is_deleted",
        "is_known_ready",
        "is_ready",
        "on_device_size_in_bytes",
        "to_py",
        "unsafe_buffer_pointer",
        "xla_dynamic_shape",
        "xla_shape",
        "default_prng_impl",
        "flattened_fun_in_tree",
        "flatten_fun",
        "flatten_fun_nokwargs",
        "get_aval",
        "concrete_aval",
        "function transformation_with_aux",
        "flatten_fun_for_vmap",
        "replace_thread_exc_traceback",
        "path_starts_with",
        "include_frame",
        "ignore_known_hidden_frame",
        "add_call_stack_frames",
        "format_exception_only",
        "xla_callable",
        "tree_leaves",
        "tree_map",
    ],
    "tensorflow": [
        "as_dtype",
        "flatten",
        "pack_sequence_as",
        "map_structure",
        "deprecated_argument_lookup",
    ],
    "torch": [
        "__torch_function__",
        "unpack_dual",
        "classes",
        "torch",
        "is_grad_enabled",
        "get_default_dtype",
        "numel",
        "cpu",
        "set_",
        "requires_grad_",
        "load",
    ],
    "ivy": [
        "args_to_ivy",
        "variable",
        "nested_map",
        "map_nest_at_index",
        "set_nest_at_index",
        "set_nest_at_indices",
        "multi_index_nest",
        "index_nest",
        "to_ivy",
        "exists",
        "default",
        "container_types",
        "to_native",
        "nested_argwhere",
        "map_nest_at_indices",
        "is_native_array",
        "current_backend_str",
        "is_array",
        "is_variable",
        "current_backend",
        "is_ivy_array",
        "get_backend",
        "with_grads",
        "check_elem_in_list",
        "check_isinstance",
        "check_all",
        "args_to_native",
        "nested_any",
        "is_ivy_container",
        "check_true",
        "handle_exceptions",
        "to_list",
        "as_ivy_dev",
        "dev",
        "dtype",
        "promote_types_of_inputs",
        "default_device",
        "handle_nestable",
        "outputs_to_ivy_arrays",
        "handle_array_like",
        "inputs_to_native_arrays",
        "inputs_to_ivy_arrays",
        "handle_out_argument",
        "as_int_dtype",
        "as_ivy_dtype",
        "gpu_is_available",
        "default_float_dtype",
        "is_float_dtype",
        "set_backend",
        "previous_backend",
        "del_global_attr",
        "check_false",
        "infer_device",
        "integer_arrays_to_float",
        "infer_dtype",
        "to_numpy",
        "as_native_dev",
        "is_int_dtype",
        "as_native_dtype",
        "default_dtype",
        "set_global_attr",
        "set_backend_to_specific_version",
    ],
}

ARRAY_BUILTINS = [
    "_pad",  # temp patch for ODSC kornia demo
    "_rewriting_take",
    "_slice_helper",
    "__neg__",
    "__pow__",
    "__rpow__",
    "__add__",
    "__radd__",
    "__iadd__",
    "__sub__",
    "__rsub__",
    "__isub__",
    "__mul__",
    "__mod__",
    "__rmod__",
    "__rmul__",
    "__imul__",
    "__matmul__",
    "__rmatmul__",
    "__truediv__",
    "__rtruediv__",
    "__itruediv__",
    "__floordiv__",
    "__rfloordiv__",
    "__ifloordiv__",
    "__idiv__",
    "__abs__",
    "__lt__",
    "__le__",
    "__eq__",
    "__ne__",
    "__gt__",
    "__ge__",
    "__and__",
    "__rand__",
    "__or__",
    "__ror__",
    "__invert__",
    "__xor__",
    "__rxor__",
    "__getitem__",
    "__setitem__",
    "__getattr__",
    "__setattr__",
    "__getattribute__",
    "__init__",
    "__repr__",
]

# Special Cases #
# ------------- #

GRAPH_ATTRIBUTES = {
    "numpy": ["shape", "ndim", "size", "itemsize", "T"],
    "paddle": ["shape"],
    "jax": ["at", "shape"],
    "tensorflow": ["shape"],
    "torch": ["data", "requires_grad", "shape", "T", "H", "mT", "mH"],
    "ivy": ["shape"],
}

INPLACE_FUNCTIONS_WITHOUT_RET = {
    "numpy": ["copyto"],
    "paddle": [],
    "jax": [],
    "tensorflow": [],
    "torch": [],
    "ivy": [],
}

INPLACE_METHODS_WITHOUT_RET = {
    "numpy": [
        "__setitem__",
        "resize",
        "sort",
        "partition",
        "fill",
        "setflags",
        "itemset",
    ],
    "paddle": ["__setitem__"],
    "jax": [],
    "tensorflow": ["assign", "assign_sub", "assign_add"],
    "torch": ["__setitem__"],
    "ivy": ["__setitem__"],
}

GENERATOR_FUNCTIONS = {
    "numpy": [
        "uniform",
        "normal",
        "rand",
        "randn",
        "random",
        "randint",
        "random_integers",
        "random_sample",
        "beta",
        "binomial",
        "chisquare",
        "dirichlet",
        "exponential",
        "f",
        "gamma",
        "geometric",
        "gumbel",
        "hypergeometric",
        "laplace",
        "logistic",
        "lognormal",
        "logseries",
        "multinomial",
        "multivariate_normal",
        "negative_binomial",
        "noncentral_chisquare",
        "noncentral_f",
        "pareto",
        "poisson",
        "rayleigh",
        "standard_cauchy",
        "standard_exponential",
        "standard_gamma",
        "standard_normal",
        "standard_t",
        "trinagular",
        "vonmises",
        "wald",
        "weibull",
        "zipf",
    ],
    "paddle": [
        "bernoulli",
        "multinomial",
        "normal",
        "poisson",
        "rand",
        "randint",
        "randint_like",
        "randn",
        "randperm",
        "standard_normal",
        "uniform",
    ],
    "jax": [
        "ball",
        "bernoulli",
        "beta",
        "categorical",
        "cauchy",
        "dirichlet",
        "double_sided_maxwell",
        "exponential",
        "gamma",
        "generalized_normal",
        "gumbel",
        "laplace",
        "loggamma",
        "logistic",
        "maxwell",
        "multivariate_normal",
        "normal",
        "orthogonal",
        "pareto",
        "poisson",
        "rademacher",
        "randint",
        "t",
        "truncated_normal",
        "uniform",
        "weibull_min",
    ],
    "tensorflow": [
        "random_uniform",
        "random_normal",
        "categorical",
        "random_gamma",
        "truncated_normal",
        "random_poisson_v2",
    ],
    "torch": [
        "rand",
        "normal",
        "multinomial",
        "randint",
        "bernoulli",
        "poisson",
        "randn",
        "randperm",
    ],
    "ivy": ["random_uniform", "random_normal", "multinomial", "randint"],
}

CLASSES_TO_TRACK = {
    "numpy": [],
    "paddle": [],
    "jax": [],
    "tensorflow": ["TensorShape"],
    "torch": ["Size"],
    "ivy": ["Shape"],
}

# Special cases- to avoid needing to create weird things in the frontends
NATIVE_TO_FRONTEND_PATH = {
    "tensorflow._api.v2.__internal__.EagerTensor.numpy": "tensorflow.Tensor.numpy",
    "jaxlib.xla_extension.ArrayImpl": "jax.DeviceArray",
}

FNS_TO_TRACK = {
    "numpy": ["shape"],
    "paddle": [],
    "jax": ["shape"],
    "tensorflow": [],
    "torch": ["size", "item"],
    "ivy": [],
}

ATTRS_TO_TRACK = {
    "numpy": ["shape", "ndim", "size", "itemsize"],
    "paddle": [],
    "jax": ["shape"],
    "tensorflow": [],
    "torch": [],
    "ivy": [],
}

CLASS_ATTRS_NOT_TO_TRACK = {
    "numpy": {},
    "paddle": {},
    "jax": {
        "DeviceArray": [
            "device_buffer",
            "is_fully_replicated",
        ],
        "ArrayImpl": [
            "device_buffer",
            "is_fully_replicated",
        ],
    },
    "tensorflow": {},
    "torch": {},
    "ivy": {},
}
