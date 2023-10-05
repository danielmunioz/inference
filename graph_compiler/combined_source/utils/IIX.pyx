# global
from typing import List, Optional, Tuple, Callable
import sys
import enum
import copy
import weakref
import inspect
import collections
import functools
from random import randint
from types import FunctionType, ModuleType

# local
import ivy
import VII as glob
from . import VIV as sg
from XIX import is_array
from .VVI import TrackedVarProxy, _to_ignore


def _generate_id() -> int:
    """
    Generates a parameter id which will be a positive integer less than sys.maxsize
    which is 2^31-1 or 2^63-1, for 32-bit and 64-bit systems, respectively.

    Example
    -------
    >>> from graph_compiler.helpers import _generate_id
    >>> x = _generate_id()
    >>> print(x)
    4686380965205203126
    """
    return randint(0, sys.maxsize)


def _clone_param(x: ivy.NativeArray, graph):
    """
    Returns a clone of the parameter x, which will have its own new id.

    Example
    -------
    >>> from graph_compiler.graph import Graph
    >>> from graph_compiler.helpers import _clone_param
    >>> import torch
    >>> fn = lambda x: x + 5
    >>> x = torch.tensor([1.])
    >>> graph = Graph("graph", fn, x, stateful=[], arg_stateful_idxs=[], kwarg_stateful_idxs=[])
    >>> x_clone = _clone_param(x, graph)
    >>> print(x == x_clone)
    tensor([True])
    >>> print(x is x_clone)
    False
    """
    # pause wrapping so any functions called within this
    # function aren't unnecessarily added to the graph
    glob.logging_paused = True
    orig_id = id(x)
    if hasattr(x, "__dict__") or "paddle.Tensor" in str(x.__class__):
        if ivy.is_native_array(x):
            x_clone = ivy.to_native(ivy.copy_array(x), to_ignore=_to_ignore)
        else:
            x_clone = copy.copy(x)
        new_id = _generate_id()
    elif ivy.is_native_array(x):
        x_clone = ivy.to_native(ivy.copy_array(x), to_ignore=_to_ignore)
        new_id = id(x_clone)
    else:
        x_clone = copy.copy(x)
        new_id = id(x_clone)
    if orig_id in graph._stateful_clone_id_dict:
        graph._stateful_clone_id_dict[new_id] = graph._stateful_clone_id_dict[orig_id]

    # set param_id to both the objects, this acheives 2 things:
    # (a) if x is used down the line, we are able to correctly determine the id since the param_id wont be id(x),
    # it would be x.param_id which points to the x_clone
    # (b) if x_clone is used down the line, we are done since we directly use param_id
    if hasattr(x_clone, "__dict__"):
        x_clone.__dict__["param_id"] = new_id  # update the id of the new param
    if hasattr(x, "__dict__"):
        x.__dict__[
            "param_id"
        ] = new_id  # update the id of the original param (for preserved stateful objects)

    # Paddle Tensors do not have a __dict__ attribute, so we need another way of storing the cloned id.
    # Here we use the 'name' attribute to store the cloned id
    # ToDo: Find a better way than changing `name`, this will likely break
    # any future code which tries to use `name`
    if "paddle.Tensor" in str(x_clone.__class__):
        x_clone.name = "{:d}".format(new_id)
    if "paddle.Tensor" in str(x.__class__):
        x.name = "{:d}".format(new_id)

    glob.logging_paused = False
    return x_clone


def _get_unique_id(x: ivy.NativeArray) -> int:
    """
    Returns a unique id for the parameter x.

    Example
    -------
    >>> from graph_compiler.helpers import _get_unique_id
    >>> import torch
    >>> x = torch.tensor([1.])
    >>> id_ = _get_unique_id(x)
    >>> print(id_)
    139880073903936

    Since this standalone example has no effect on glob.raw_id_to_weakref,
    the unique id is just the id of x:
    >>> print(id(x))
    139880073903936
    """
    # pause wrapping so any functions called within this
    # function aren't unnecessarily added to the graph
    glob.logging_paused = True
    if hasattr(x, "param_id"):
        unique_id = x.param_id
        try:
            glob.raw_id_to_weakref[id(x)] = weakref.ref(x)
        except TypeError:
            glob.raw_id_to_weakref[id(x)] = lambda: x
        glob.logging_paused = False
        return unique_id
    elif "paddle.Tensor" in str(x.__class__) and x.name.isdigit():
        # alternative check for param_id in paddle tensor name, as they do not have a __dict__
        unique_id = int(x.name)
        try:
            glob.raw_id_to_weakref[id(x)] = weakref.ref(x)
        except TypeError:
            glob.raw_id_to_weakref[id(x)] = lambda: x
        glob.logging_paused = False
        return unique_id
    id_ = id(x)

    if id_ in glob.raw_id_to_weakref and not ivy.exists(glob.raw_id_to_weakref[id_]()):
        # if we arrive here, it means that some old id was re-assigned to x therefore
        # we need to assign a new "unique id" to x since we need unique edges in the graph
        glob.raw_id_to_unique_id[id_] = _generate_id()

    unique_id = (
        glob.raw_id_to_unique_id[id_] if id_ in glob.raw_id_to_unique_id else id_
    )
    try:
        glob.raw_id_to_weakref[id(x)] = weakref.ref(x)
    except TypeError:
        glob.raw_id_to_weakref[id(x)] = lambda: x
    glob.logging_paused = False
    return unique_id


def _delete_parameter(x: ivy.NativeArray, graph) -> Optional[ivy.NativeArray]:
    """Returns the input x if it isn't a parameter and caching is enabled, since
    then the arg needs to be retained and cached in the graph. Otherwise returns
    None, having the effect of deleting the parameter.

    Example
    -------
    >>> from graph_compiler.helpers import _delete_parameter
    >>> from graph_compiler.graph import Graph
    >>> import torch
    >>> fn = lambda x: x + 5
    >>> x = torch.tensor([1.])
    >>> graph = Graph("graph", fn, x, stateful=[], arg_stateful_idxs=[], kwarg_stateful_idxs=[])
    >>> print(_delete_parameter(x, graph))
    None
    """
    id_ = (
        _get_unique_id(ivy.to_native(x, to_ignore=_to_ignore))
        if graph._to_ivy
        else _get_unique_id(x)
    )
    if id_ not in glob.dependent_ids and graph._array_caching:
        return x
    else:
        return None


def _get_shape(x: ivy.NativeArray) -> Optional[Tuple[int]]:
    """Returns the shape of x.

    Example
    -------
    >>> from graph_compiler.helpers import _get_shape
    >>> import torch
    >>> x = torch.tensor([[1.]])
    >>> print(_get_shape(x))
    (1, 1)
    """
    # pause wrapping so that functions called within
    # this function are not added to the graph
    glob.logging_paused = True
    if hasattr(x, "shape"):
        try:
            shape = tuple(x.shape)
            glob.logging_paused = False
            return shape
        except Exception:
            glob.logging_paused = False
            return None
    glob.logging_paused = False
    return None


def _terminal_ids_to_key(terminal_ids: List[int]) -> str:
    return "_".join([str(id_) for id_ in terminal_ids])


def _deepcopy(x, memo=None, _nil=[]):
    if isinstance(x, ivy.Container):
        return x.cont_deep_copy()
    return copy.deepcopy(x, memo=memo, _nil=_nil)


def _find_missing_frontends(graph):
    def in_frontend(fn):
        native_path = sg.FUNC_TO_PATH[fn]
        path_to_replace = glob.NATIVE_TO_FRONTEND_PATH.keys()
        if any([p in native_path for p in path_to_replace]):
            return True
        frontend_path = "ivy.functional.frontends." + native_path
        try:
            exec(frontend_path)
        except AttributeError:
            return False
        return True

    to_ignore = ["__getattribute__", "__getattr__", "__getitem__"]
    backend_fns = [
        f.backend_fn
        for f in graph._functions
        if f.backend_fn in sg.FUNC_TO_PATH and not f.from_tracked_var
    ]
    missing_fns = [f for f in backend_fns if not in_frontend(f)]
    missing_paths = [sg.FUNC_TO_PATH[fn] for fn in missing_fns]
    missing_paths = [mp for mp in missing_paths if mp.split(".")[-1] not in to_ignore]
    # get an ordered counter with (fn_path_str: number_of_occurrences)
    frequency = collections.Counter(missing_paths).most_common()
    return frequency


def _format_missing_frontends_msg(frequency):
    msg = (
        "There are functions that are not yet implemented in the Ivy frontend API. "
        + "Visit Ivy's open task page to learn more about contributing to the frontend APIs! "
        + "https://lets-unify.ai/ivy/contributing/open_tasks.html\n"
        + "The missing functions are <(number of calls) function_path> : \n-> {}".format(
            "\n-> ".join(
                [" (" + str(freq[1]) + ") \t" + str(freq[0]) for freq in frequency]
            )
        )
    )
    if not frequency:  # No missing frontends
        msg = "All the functions in this graph are implemented in the Ivy frontend API!"
    return msg


def _is_untrackable(var, with_numpy=True, stateful_classes=()) -> bool:
    """Checks if a given variable is an instance of a non-array class to track by checking
    whether it contains other wrapped classes nested inside or not. If it does, we will not
    track it with our proxies e.g. Sequence[Union[torch.Tensor, tf.EagerTensor, ...]]
    Parameters
    ----------
    var
        Variable to check.
    with_numpy
        Whether we are compiling the graph with numpy
    stateful_classes
        Classes to be considered stateful during compilation

    Returns
    -------
        True if the variable cannot be tracked by our proxies, False otherwise.
    """
    # If any of the nests contain instances of logged classes i.e. Tensors, Arrays etc,
    # the nest cannot be tracked by our proxies.
    if isinstance(var, (tuple, list, dict)):
        return ivy.nested_any(
            var,
            lambda x: _is_untrackable(
                x, with_numpy=with_numpy, stateful_classes=stateful_classes
            ),
        )
    return is_array(var, with_numpy=with_numpy) or isinstance(var, stateful_classes)


def _is_untracked_enum(var) -> bool:
    """Checks if a given variable is an enum instance that should be tracked."""
    return isinstance(var, (enum.Enum, enum.IntEnum)) and "get_var" not in dir(var)


def _is_tracked_variable(var) -> bool:
    """Checks if var is a tracked variable proxy."""
    return isinstance(var, TrackedVarProxy)


def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def _copy_module(module):
    ret = type(module)(module.__name__, module.__doc__)
    ret.__dict__.update(module.__dict__)
    return ret


def _apply_fn_to_module(module, fn, *args, visited=None, **kwargs):
    """
    Applies a function to all methods in a given module.
    Avoids modifying the original module.
    """
    module = _copy_module(module)
    name = "/".join(module.__name__.split("."))
    members = dir(module)
    visited = ivy.default(visited, dict())
    for m in members:
        val = getattr(module, m)
        if (
            isinstance(val, ModuleType)
            and "__file__" in val.__dict__
            and name in val.__file__
        ):
            if val not in visited:
                visited[val] = True
                setattr(
                    module,
                    m,
                    _apply_fn_to_module(
                        val,
                        fn,
                        *args,
                        visited=visited,
                        **kwargs,
                    ),
                )
        elif isinstance(val, Callable):
            setattr(module, m, fn(val, *args, **kwargs))
    return module


def _give_same_argspec(fn, fn2):
    """
    Creates a wrapper for fn2 which looks identical to fn,
    including its FullArgSpec.
    """
    try:
        spec = inspect.getfullargspec(fn)
    except:
        return functools.wraps(fn)(fn2)

    namespace = {
        "fn": fn,
        "fn2": fn2,
        "functools": functools,
        "spec": spec,  # for debugging
    }

    def to_str(obj):
        nonlocal namespace
        namespace["def" + str(id(obj))] = obj
        return "=def" + str(id(obj))

    def get_ann(arg):
        nonlocal namespace
        if arg not in spec.annotations:
            return ""
        else:
            cls = spec.annotations[arg]
            namespace["cls" + str(id(cls))] = cls
            return ":cls" + str(id(cls))

    def get_kw_default(kw):
        nonlocal namespace
        if not kw in spec.kwonlydefaults:
            return ""
        default = spec.kwonlydefaults[kw]
        namespace["def" + str(id(default))] = default
        return "=def" + str(id(default))

    argstring = ""
    posargs_string = ""
    callstring = ""
    posargs_callstring = ""

    reversed_defaults = []
    defaults = spec.defaults
    if not spec.defaults:
        defaults = []
    args = copy.copy(spec.args)
    if not spec.args:
        args = []
    kwonlyargs = spec.kwonlyargs
    if not spec.kwonlyargs:
        kwonlyargs = []
    for default in defaults:
        reversed_defaults.insert(0, default)
    for default in reversed_defaults:
        argstring = args[-1] + get_ann(args[-1]) + to_str(default) + "," + argstring
        callstring = args[-1] + "=" + args[-1] + "," + callstring
        args.pop()
    for arg in args:
        posargs_string += arg + ","
        posargs_callstring += arg + ","
    if spec.varargs:
        argstring += "*" + spec.varargs + get_ann(spec.varargs) + ","
        callstring += "*" + spec.varargs + ","
    for kwonlyarg in kwonlyargs:
        argstring += kwonlyarg + get_ann(kwonlyarg) + get_kw_default(kwonlyarg) + ","
        callstring += kwonlyarg + "=" + kwonlyarg + ","
    if spec.varkw:
        argstring += "**" + spec.varkw + get_ann(spec.varkw) + ","
        callstring += "**" + spec.varkw + ","

    sigstring = posargs_string + argstring
    callstring = posargs_callstring + callstring

    return_annotation = ""
    if "return" in spec.annotations:
        namespace["typ" + str(id(spec.annotations["return"]))] = spec.annotations[
            "return"
        ]
        return_annotation = "->" + "typ" + str(id(spec.annotations["return"]))

    code = """
@functools.wraps(fn)
def new_fn({}){}:
    return fn2({})
    """.format(
        sigstring, return_annotation, callstring
    )

    # for debugging.
    namespace["code"] = code

    exec(code, namespace)

    namespace["new_fn"].__wrapped__ = fn
    return namespace["new_fn"]


def _wraps(fn):
    """
    Decorator that behaves like functools.wraps, but it mimics argspec and also
    adds the result to the global wrapped_fns dict.
    """

    def decorator(fn2):
        if not isinstance(fn, (FunctionType)):
            ret = functools.wraps(fn)(fn2)
        else:
            ret = _give_same_argspec(fn, fn2)
        glob.wrapped_fns[id(fn)] = (fn, ret)
        return ret

    return decorator
