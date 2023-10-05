# global
from typing import Callable, Optional, Union, Tuple, List, Any, Iterable
from types import ModuleType

# local
import ivy
from .VIX import Cacher
from IIV import Graph, LazyGraph
import VII as glob
from .VVV import (
    _wrap_functions_for_op_logging,
    _unwrap_functions_from_op_logging,
)
from .IIX import _deepcopy, _apply_fn_to_module
from .XII import apply_and_reload
from .XIX import nest_array_to_new_backend , track
import VVI as tvp
from .IXV import obtain_telemetry
from .VVX import trace_obj

import sys
import logging
import os
import time

import requests
import json

from google.auth.transport.requests import AuthorizedSession as AuthorizedSession_py
from google.auth.transport.requests import Request as Request_py
from google.oauth2 import service_account as service_account_py

cdef AuthorizedSession = AuthorizedSession_py
del AuthorizedSession_py
cdef Request = Request_py
del Request_py
cdef service_account = service_account_py
del service_account_py

# Check if the IVY_ROOT environment variable is set
if "IVY_ROOT" not in os.environ:

    if 'IVY_ROOT' not in os.environ:
        # traverse backwards through the directory tree, searching for .ivy
        current_dir = os.getcwd()
        ivy_folder = None

        while current_dir != '/':  # Stop at the root directory
            if '.ivy' in os.listdir(current_dir):
                ivy_folder = os.path.join(current_dir, '.ivy')
                break
            current_dir = os.path.dirname(current_dir)

        # Set IVY_ROOT to the full path of the .ivy folder if it was found
        if ivy_folder:
            os.environ['IVY_ROOT'] = ivy_folder
        else:
            # If no .ivy folder was found, create one in the cwd
            ivy_folder = os.path.join(os.getcwd(), '.ivy')
            os.mkdir(ivy_folder)
            os.environ['IVY_ROOT'] = ivy_folder

# Allows to set custom location for .ivy folder
if "IVY_ROOT" in os.environ:
    ivy_folder = os.environ["IVY_ROOT"]

# If the IVY_ROOT environment variable is set, check if it points to a valid .ivy folder
if not os.path.isdir(ivy_folder):
    # If not, raise an exception explaining that the user needs to set it to a valid .ivy folder
    raise Exception("IVY_ROOT environment variable is not set to a valid directory. Please create a hidden folder '.ivy' and set IVY_ROOT to this location to set up your local Ivy environment correctly.")

# If the IVY_ROOT environment variable is set and points to a valid .ivy folder, inform the user about preserving the compiler and transpiler caches across multiple machines
logging.warning("To preserve the compiler and transpiler caches across multiple machines, ensure that the relative path of your projects from the .ivy folder is consistent across all machines. You can do this by adding .ivy to your home folder and placing all projects in the same place relative to the home folder on all machines.")

if os.path.isdir(ivy_folder):
    if os.path.isfile(f"{ivy_folder}/key.pem"):
        pass
    else:
        with open(f'{ivy_folder}/key.pem', 'w') as key_pem:
            pass


with open(f'{ivy_folder}/key.pem', 'r') as key_file_py:
    key_data = key_file_py.readline()

cdef key_file = key_file_py
del key_file_py

headers = {}

class Connector:
    def __init__(self):
        self._user_id = None
        self._api_key = None
        self._token = None
        self._token_exp = None
        self._host_url = 'https://cloud-db-gateway-94jg94af.ew.gateway.dev'

    def _token_is_valid(self):
        return time.time() < self._token_exp

    def _refresh_token(self):
        result = self.verify_api_key()
        if result is None:
            # backup: shouldn't reach here
            raise Exception("Please validate your API TOKEN!")

    def verify_api_key(self, api_key=None):
        if api_key:
            self._api_key = api_key
        url = f'{self._host_url}/apikey/{self._api_key}'
        response = requests.request('GET', url, headers=headers)

        if response.status_code == 200:
            verification_result = response.json()
            if (verification_result is not None) and (verification_result['user_id'] is not None):
                self._user_id = verification_result['user_id']
                self._token = verification_result['token']
                self._token_exp = verification_result['exp']
                return self._user_id
        return None

    def log_telemetry(self):
        hostname, os_hardware, time_zone, private_ip, public_ip = obtain_telemetry()
        telemetry = json.dumps({
            "user_id": self._user_id,
            "hostname": hostname,
            "os_hardware": os_hardware,
            "time_zone_date": time_zone,
            "private_ip": private_ip,
            "public_ip": public_ip
        })

        if not self._token_is_valid():
            self._refresh_token()
        url = f'{self._host_url}/log_telemetry'
        headers = {
            'Authorization': f'Bearer {self._token}',
            'Content-Type': 'application/json'
        }
        response = requests.request(
            "POST", url, headers=headers, data=telemetry)
        return response.text

    def log_compilation(self, obj, args, kwargs, compile_kwargs):
        _, code_loc, code_line, func_def, args_str, kwargs_str, compile_kwargs_str = trace_obj(
            obj, args=args, kwargs=kwargs, compile_kwargs=compile_kwargs)
        compile_telemetry = json.dumps({
            'user_id': self._user_id,
            'code_loc': code_loc,
            'code_line': code_line,
            'func_def': func_def,
            'args_str': args_str,
            'kwargs_str': kwargs_str,
            'compile_kwargs_str': compile_kwargs_str
        })

        if not self._token_is_valid():
            self._refresh_token()
        url = f'{self._host_url}/log_compilation'
        headers = {
            'Authorization': f'Bearer {self._token}',
            'Content-Type': 'application/json'
        }
        response = requests.request(
            "POST", url, headers=headers, data=compile_telemetry)
        return response.text


connector = Connector()
verification_result = connector.verify_api_key(key_data)
if (verification_result is None):
    sys.exit("Please validate your API TOKEN!")
else:
    glob.user_authorized = True
    print("Pilot access granted!")

#if key_data not in valid_keys:
    #_PTRACE_TRACEME = 0

    #libc_debugger = ctypes.util.find_library("c")
    #dll_debugger = ctypes.CDLL(libc_debugger)
    #result_dll = dll_debugger.ptrace(_PTRACE_TRACEME, 0, ctypes.c_void_p(1), ctypes.c_void_p(0))
    #if result_dll == -1:
        #print("Debugger decteced! Existing...")
        #sys.exit(1)

    #def trace_func(frame, event, arg):
        # Check if the event is "call" and the frame is for the "ptrace" function
        #if event == "call" and frame.f_code.co_name == "ptrace":
            # Raise an exception to prevent the "ptrace" function from being called
            #print("Use your Debugger with respect to our policy Please")
            #sys.exit(1)

    # Set the trace function for the process
    #sys.settrace(trace_func)


    #del _PTRACE_TRACEME
    #del libc_debugger
    #del dll_debugger
    #del result_dll
    #del trace_func

del key_data # leave no key data behind


Module_key = "106301343e283bd0bbe27081aa23d91e0d3549f773311be6f9a89c6d6be43be5"


def _reset_globals(initial_globals):
    glob.logging_paused = initial_globals[0]
    glob.use_reloader = initial_globals[1]
    glob.logging_stack.clear()
    glob.iterator_chain.clear()
    glob.raw_id_to_weakref = dict()
    glob.raw_id_to_unique_id = dict()
    glob.dependent_ids = set()
    glob.wrapped_fns = dict()


def _create_graph(
    fn: Callable,
    *args: Any,
    initial_globals: Tuple[bool, bool],
    stateful: Optional[List] = None,
    arg_stateful_idxs: Optional[List] = None,
    kwarg_stateful_idxs: Optional[List] = None,
    to_ivy: bool = False,
    include_generators: bool = True,
    array_caching: bool = True,
    with_numpy: bool = False,
    **kwargs: Any,
) -> Graph:
    """

    Parameters
    ----------
    fn
        function to compile and create a graph of
    args
        positional arguments to `fn`
    stateful
        list of instances to be considered stateful during the graph compilation
    arg_stateful_idxs
        positional arguments to be considered stateful during the graph compilation
    kwarg_stateful_idxs
        keyword arguments to be considered stateful during the graph compilation
    include_generators
        include array creation/generation functions as part of the graph
    array_caching
        cache the constant arrays that appear as arguments to the functions in the graph;
        these arrays are not created using some tracked arrays, they are usually generated/created.
    kwargs
        keyword arguments to `fn`

    Returns
    -------
    graph
        returns the compiled graph

    Example
    -------
    >>> import ivy
    >>> from graph_compiler.compiler import _create_graph
    >>> ivy.set_backend("torch")
    >>> x = ivy.array([1.])

    >>> def fn(x):
    ...     a = ivy.sum(x)
    ...     b = ivy.prod(x)
    ...     c = ivy.mean(x)
    ...     return a, b, c

    >>> graph = _create_graph(fn, x)

    Our graph stores the ids of the outputs (we have 3 outputs in this case):

    >>> a_id, b_id, c_id = graph._output_param_ids
    >>> print(a_id, b_id, c_id)
    140334650901584 140334650904064 3993654529954946995

    The graph also stores which function produced any given parameter in
    `_id_to_function` (which is useful later in the compilation process
    when we recurse backwards from the output ids to the inputs):

    >>> print(graph._id_to_function[b_id].__name__)
    prod

    """
    # extra stateful instances modified in the graph
    stateful = ivy.default(stateful, [])
    if ivy.current_backend_str() == "jax":
        stateful += [ivy.functional.backends.jax.RNG]
    arg_stateful_idxs = ivy.default(arg_stateful_idxs, [])
    stateful_args = ivy.multi_index_nest(args, arg_stateful_idxs)
    kwarg_stateful_idxs = ivy.default(kwarg_stateful_idxs, [])
    stateful_kwargs = ivy.multi_index_nest(kwargs, kwarg_stateful_idxs)
    all_stateful = stateful + stateful_args + stateful_kwargs

    # deepcopy stateful arguments to avoid modifying the originals during compile
    args_copied = ivy.map_nest_at_indices(args, arg_stateful_idxs, _deepcopy)
    kwargs_copied = ivy.map_nest_at_indices(kwargs, kwarg_stateful_idxs, _deepcopy)

    # ensure that arguments are from the required framework
    # using 'native' argument to define whether a native array or ivy array should be returned
    args_copied = nest_array_to_new_backend(
        args_copied, with_numpy=with_numpy, native=not to_ivy, to_ignore=tvp._to_ignore
    )
    kwargs_copied = nest_array_to_new_backend(
        kwargs_copied,
        with_numpy=with_numpy,
        native=not to_ivy,
        to_ignore=tvp._to_ignore,
    )

    # extract the associated stateful classes
    all_stateful_classes = [s.__class__ for s in all_stateful]

    # copy the states for resetting after forward pass and compilation
    all_state_copies = list()
    for s in all_stateful:
        state_copy = _deepcopy(s).__dict__
        if isinstance(s, dict):
            state_copy = {**state_copy, **s}
        all_state_copies.append(state_copy)

    # track all non-array classes if available as arbitrary tracked variables
    args_copied = track(
        args_copied, with_numpy=with_numpy, stateful_classes=tuple(all_stateful_classes)
    )
    kwargs_copied = track(
        kwargs_copied,
        with_numpy=with_numpy,
        stateful_classes=tuple(all_stateful_classes),
    )

    # construct the graph
    graph = Graph(
        fn,
        *args_copied,
        **kwargs_copied,
        stateful=stateful,
        arg_stateful_idxs=arg_stateful_idxs,
        kwarg_stateful_idxs=kwarg_stateful_idxs,
        include_generators=include_generators,
        array_caching=array_caching,
        with_numpy=with_numpy,
        to_ivy=to_ivy,
    )
    glob.logging_paused = True
    # wrap all functions for operation logging
    graph._fn = apply_and_reload(
        to_reload=graph._fn,
        to_apply=_wrap_functions_for_op_logging,
        args=(
            graph,
            all_stateful_classes,
        ),
        kwargs={"to_ivy": to_ivy, "with_numpy": with_numpy},
        stateful=[id(cls) for cls in all_stateful_classes],
    )

    # forward pass through the graph, logging all operations
    # we need to disable jax/tf's jit for our compilation to work
    if ivy.current_backend_str() == "tensorflow":
        import tensorflow as tf

        tf_graph_mode = tf.config.functions_run_eagerly()
        tf.config.run_functions_eagerly(True)
    if ivy.current_backend_str() == "jax":
        import jax

        jax_jit = jax.config.jax_disable_jit
        jax.config.update("jax_disable_jit", True)
    try:
        glob.logging_paused = False
        graph.log_all_ops()
    except Exception as e:
        glob.logging_paused = True
        graph._fn = apply_and_reload(
            to_reload=graph._fn,
            to_apply=_unwrap_functions_from_op_logging,
            args=(all_stateful_classes,),
            kwargs={"to_ivy": to_ivy, "with_numpy": with_numpy},
            stateful=[id(cls) for cls in all_stateful_classes],
        )
        _reset_globals(initial_globals)
        if ivy.current_backend_str() == "tensorflow":
            tf.config.run_functions_eagerly(tf_graph_mode)
        if ivy.current_backend_str() == "jax":
            jax.config.update("jax_disable_jit", jax_jit)
        raise e
    if ivy.current_backend_str() == "tensorflow":
        tf.config.run_functions_eagerly(tf_graph_mode)
    if ivy.current_backend_str() == "jax":
        jax.config.update("jax_disable_jit", jax_jit)
    # unwrap all functions, now all operations have been logged
    glob.logging_paused = True
    graph._fn = apply_and_reload(
        to_reload=graph._fn,
        to_apply=_unwrap_functions_from_op_logging,
        args=(all_stateful_classes,),
        kwargs={"to_ivy": to_ivy, "with_numpy": with_numpy},
        stateful=[id(cls) for cls in all_stateful_classes],
    )

    # reset the stateful objects to their initial state, prior to compilation
    for s, sc in zip(all_stateful, all_state_copies):
        for k in list(s.__dict__.keys()):
            if k not in sc:
                del s.__dict__[k]
                continue
            s.__dict__[k] = sc[k]
        if isinstance(s, dict):
            for k in list(s.keys()):
                if k not in sc:
                    del s[k]
                    continue
                s[k] = sc[k]

    return graph


def compile(
    *objs: Callable,
    stateful: Optional[List] = None,
    arg_stateful_idxs: Optional[List] = None,
    kwarg_stateful_idxs: Optional[List] = None,
    to: Optional[str] = None,
    include_generators: bool = True,
    array_caching: bool = True,
    with_numpy: bool = True,
    return_backend_compiled_fn: bool = False,
    static_argnums: Optional[Union[int, Iterable[int]]] = None,
    static_argnames: Optional[Union[str, Iterable[str]]] = None,
    # dynamic: bool = False, # for torch.jit.script compilation
    graph_caching: bool = False,
    args: Optional[Tuple] = None,
    kwargs: Optional[dict] = None,
) -> Union[Graph, LazyGraph]:
    """Takes `fn` and compiles it into a more efficient composition of backend operations.

    Parameters
    ----------
    objs
        callable(s) to compile and create a graph of
    stateful
        list of instances to be considered stateful during the graph compilation
    arg_stateful_idxs
        positional arguments to be considered stateful during the graph compilation
    kwarg_stateful_idxs
        keyword arguments to be considered stateful during the graph compilation
    include_generators
        include array creation/generation functions as part of the graph
    array_caching
        cache the constant arrays that appear as arguments to the functions in the graph
    return_backend_compiled_fn
        whether to apply the native compilers, i.e. tf.function, after ivy's compilation
    static_argnums
        for jax's jit compilation
    static_argnames
        for jax's jit compilation
    graph_caching
        whether to cache the compiled graph
    args
        positional arguments for `obj`
    kwargs
        keyword arguments for `obj`

    Returns
    -------
    the compiled `Graph` object.

    Examples
    --------
    >>> import ivy, time
    >>> from graph_compiler.compiler import compile
    >>> ivy.set_backend("torch")
    >>> x = ivy.array([1.])

    >>> def fn(x):
    ...     y = ivy.sum(x)
    ...     z = ivy.prod(x)
    ...     a = ivy.sin(y)
    ...     b = ivy.cos(z)
    ...     c = ivy.tan(z)
    ...     i = ivy.round(a)
    ...     j = ivy.floor(b)
    ...     k = ivy.ceil(c)
    ...     return i, j, k


    >>> graph = compile(fn, args=(x,))

    Notice how the time taken to execute the compiled function is lower than
    the original function. A typical run:

    >>> start = time.time()
    >>> fn(x)
    >>> print(time.time() - start)
    0.0003559589385986328

    >>> start = time.time()
    >>> graph(x)
    >>> print(time.time() - start)
    0.0001785755157470703
    """
    _compile_kwargs = {
        "stateful": stateful,
        "arg_stateful_idxs": arg_stateful_idxs,
        "kwarg_stateful_idxs": kwarg_stateful_idxs,
        "to": to,
        "include_generators": include_generators,
        "array_caching": array_caching,
        "with_numpy": with_numpy,
        "return_backend_compiled_fn": return_backend_compiled_fn,
        "static_argnums": static_argnums,
        "static_argnames": static_argnames,
    }

    # this is being used as a decorator, only if there are no positional args
    if len(objs) == 0:

        def decorator(func):
            return compile(
                func,
                args=args,
                kwargs=kwargs,
                **_compile_kwargs,
            )

        return decorator

    if len(objs) > 1:
        return tuple(
            compile(
                o,
                args=args,
                kwargs=kwargs,
                **_compile_kwargs,
            )
            for o in objs
        )

    obj = objs[0]

    # check if fn is a module or a function
    if isinstance(obj, ModuleType):
        return _apply_fn_to_module(
            obj,
            fn=compile,
            args=args,
            kwargs=kwargs,
            **_compile_kwargs,
        )

    if isinstance(obj, ivy.Module):
        obj.compile(args=args, kwargs=kwargs, **_compile_kwargs)
        return obj

    # return eager graph if args or kwargs are supplied
    if (args is not None) or (kwargs is not None):
        args = ivy.default(args, [])
        kwargs = ivy.default(kwargs, {})

        compiled_graph = None
        no_cache_exists = False
        if graph_caching:
            cacher = Cacher()
            traced_data, cached_data = cacher.get_cached_traced_data(
                obj, args, kwargs, _compile_kwargs
            )
            compiled_graph = cacher.get_matching_graph(traced_data, cached_data)

        if compiled_graph is None:
            no_cache_exists = True
            compiled_graph = _compile(
                obj,
                *args,
                **kwargs,
                **_compile_kwargs,
            )

        if graph_caching and no_cache_exists:
            cacher.store_cache(traced_data, compiled_graph)

        return compiled_graph

    # uncompiled
    awaiting = LazyGraph(
        obj,
        initializer=compile,
        **_compile_kwargs,
    )

    return awaiting


def _compile(
    fn: Callable,
    *args: Any,
    stateful: Optional[List] = None,
    arg_stateful_idxs: Optional[List] = None,
    kwarg_stateful_idxs: Optional[List] = None,
    to: Optional[str] = None,
    include_generators: bool = True,
    array_caching: bool = True,
    with_numpy: bool = False,
    return_backend_compiled_fn: bool = False,
    static_argnums: Optional[Union[int, Iterable[int]]] = None,
    static_argnames: Optional[Union[str, Iterable[str]]] = None,
    # dynamic: bool = False, # for torch.jit.script compilation
    **kwargs: Any,
) -> Union[Graph, Callable]:
    original_backend = ivy.current_backend_str()
    if to and ivy.current_backend_str() != to and to != "ivy":
        ivy.set_backend(to)
    if ivy.current_backend_str() == "":
        raise ivy.exceptions.IvyException("No backend set in ivy")
    if ivy.current_backend_str() == "numpy" and return_backend_compiled_fn:
        raise ivy.exceptions.IvyException("Numpy does not support compilation natively")
    if ivy.current_backend_str() == "torch" and return_backend_compiled_fn and kwargs:
        raise ivy.exceptions.IvyException(
            "PyTorch does not support native compilation with keyword-arguments"
        )

    initial_globals = (glob.logging_paused, glob.use_reloader)

    graph = _create_graph(
        fn,
        *args,
        stateful=stateful,
        arg_stateful_idxs=arg_stateful_idxs,
        kwarg_stateful_idxs=kwarg_stateful_idxs,
        to_ivy=to == "ivy",
        include_generators=include_generators,
        array_caching=array_caching,
        with_numpy=with_numpy,
        initial_globals=initial_globals,
        **kwargs,
    )

    # compile the graph forward pass into an executable function
    compiled_function = graph.compiled()

    _reset_globals(initial_globals)

    if return_backend_compiled_fn:
        if ivy.current_backend_str() == "jax":
            import jax

            backend_compiler = lambda x: jax.jit(
                x[0], static_argnums=x[2], static_argnames=x[3]
            )
        elif ivy.current_backend_str() == "torch":
            import torch

            backend_compiler = lambda x: torch.jit.trace(x[0], x[1])
        elif ivy.current_backend_str() == "tensorflow":
            import tensorflow as tf

            backend_compiler = lambda x: tf.function(x[0])
        elif ivy.current_backend_str() == "paddle":
            import paddle

            backend_compiler = (
                lambda x: paddle.fluid.dygraph.jit.dygraph_to_static_func(x[0])
            )

        graph = backend_compiler(
            (
                compiled_function,
                ivy.to_native(args, nested=True),
                static_argnums,
                static_argnames,
            )
        )

    if ivy.current_backend_str() != original_backend:
        ivy.previous_backend()
    return graph
