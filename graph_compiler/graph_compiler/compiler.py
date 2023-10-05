# global
from typing import Callable, Optional, Union, Tuple, List, Any, Iterable
from types import ModuleType

# local
import ivy
from tracing_caching.cacher import Cacher
from graph_compiler.graph import Graph, LazyGraph
from graph_compiler import globals as glob
from graph_compiler.wrapping import (
    _wrap_functions_for_op_logging,
    _unwrap_functions_from_op_logging,
    _wrap_function_for_op_logging,
    FUNC_TO_PATH,
)
from graph_compiler.helpers import _deepcopy, _apply_fn_to_module
from graph_compiler.reloader import apply_and_reload
from graph_compiler.conversion import nest_array_to_new_backend, track
import graph_compiler.tracked_var_proxy as tvp

# import control_flow_experimental.autograph_ivy.core.api as cfe


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

    # control flow AST conversion
    if ivy.current_backend_str() == "torch" and not to_ivy:
        # fn = cfe.to_functional_form(fn)
        pass

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
    if graph._fn in FUNC_TO_PATH:
        graph._fn = _wrap_function_for_op_logging(graph._fn, graph, to_ivy=to_ivy)
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
    if to not in ["torch", "tensorflow", "jax", "numpy", "paddle", "ivy", None]:
        raise ivy.exceptions.IvyException(
            "'to' must be one of 'torch', 'tensorflow', 'jax', 'numpy', 'paddle' or 'ivy'. "
        )

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
    if to is None and ivy.current_backend_str() == "":
        raise ivy.exceptions.IvyException(
            "The source framework must be specified either with the 'to' argument, "
            "which can be one from ['torch', 'tensorflow', 'jax', 'numpy', 'paddle', 'ivy'], "
            "or by setting ivy's backend with ivy.set_backend('torch'), for example."
        )
    if to and ivy.current_backend_str() != to and to != "ivy":
        ivy.set_backend(to)
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

            if paddle.__version__ > "2.4.2":
                backend_compiler = lambda x: paddle.jit.api.dygraph_to_static_func(x[0])
            else:
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
