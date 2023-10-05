from typing import Callable, Iterable, Optional, List
import time
import inspect
import importlib
import numpy as np
from types import FunctionType, BuiltinFunctionType
import functools

# local
import ivy
from ivy.functional.ivy.gradients import _is_variable
from .XIX import (
    to_custom_numpy_type,
    is_array,
    native_array_to_frontend,
    _batched_tracer_to_array,
    _convert_to_ivy_dtype,
    _custom_to_numpy,
    track,
    untrack,
)
import VII as glob
from .IIV import Graph
import VVI as tvp
from .III import NewNDArray, custom_np_classes
import IIX
from .IXX import _get_argument_reprs, _get_output_reprs
from . IVV import (
    process_vmap_fn,
    process_scalar_fn,
    process_vectorized_fn,
    add_incoming_subgraph_fns,
    add_subgraph_fns_to_dict,
)


class Node:
    """Represents a node in the graph i.e. some native/ivy function."""

    pass


def _find_parameter_indexes(nest, with_numpy, stateful_classes, tracked_var_idxs=[]):
    """Find the indexes of the parameters in the args and kwargs."""
    tracked_idxs = (
        ivy.nested_argwhere(
            nest,
            lambda x: is_array(x, with_numpy=with_numpy)
            or isinstance(x, stateful_classes),
            check_nests=True,
            to_ignore=tvp._to_ignore,
        )
        + tracked_var_idxs
    )
    return tracked_idxs


def _record_parameters_info(
    args, to_ivy, with_numpy, stateful_classes, tracked_var_idxs=[]
):
    indexes = _find_parameter_indexes(
        args, with_numpy, stateful_classes, tracked_var_idxs
    )
    parameters = ivy.multi_index_nest(args, indexes)
    ids = [
        IIX._get_unique_id(ivy.to_native(p, to_ignore=tvp._to_ignore))
        if to_ivy
        else IIX._get_unique_id(p)
        for p in parameters
    ]
    types = [p.__class__ for p in parameters]
    var_flags = [
        _is_variable(p, exclusive=True, to_ignore=tvp._to_ignore) for p in parameters
    ]
    shapes = [IIX._get_shape(p) for p in parameters]
    return indexes, parameters, ids, types, var_flags, shapes


def _cache_constant_args(args, kwargs, node, to_ivy):
    args, kwargs = _convert_to_ivy_dtype(args, kwargs, to_ivy)
    args, kwargs = _custom_to_numpy(args, kwargs)
    node.args = args
    node.kwargs = kwargs
    return args, kwargs


def _wrap_numpy_ufuncs(wrapped, original, graph):
    """NumPy ufuncs (eg np.add) aren't functions, but instances of a class.
    Hence functools.wraps won't properly handle copying over the attributes to
    the wrapped function. This function does that manually.
    Also some attributes (eg np.add.reduce) could also be in the graph, so we
    wrap these methods before copying them over.
    """
    if isinstance(original, np.ufunc):
        wrapped.nin = original.nin
        wrapped.nout = original.nout
        wrapped.nargs = original.nargs
        wrapped.ntypes = original.ntypes
        wrapped.types = original.types
        wrapped.ntypes = original.ntypes
        wrapped.signature = original.signature
        wrapped.identity = original.identity
        wrapped.reduce = _wrap_function_for_op_logging(original.reduce, graph)
        wrapped.accumulate = _wrap_function_for_op_logging(original.accumulate, graph)
        wrapped.reduceat = _wrap_function_for_op_logging(original.reduceat, graph)
        wrapped.outer = _wrap_function_for_op_logging(original.outer, graph)
        wrapped.at = _wrap_function_for_op_logging(original.at, graph)


def _unwrap_numpy_ufuncs(wrapped, original):
    """Since called attributes of NumPy ufuncs aren't exposed through the normal paths,
    we need to look inside the attributes of wrapped functions
    during unwrapping to find and unwrap these.
    """
    if isinstance(original, np.ufunc):
        wrapped.reduce = _unwrap_function_from_op_logging(wrapped.reduce)
        wrapped.accumulate = _unwrap_function_from_op_logging(wrapped.accumulate)
        wrapped.reduceat = _unwrap_function_from_op_logging(wrapped.reduceat)
        wrapped.outer = _unwrap_function_from_op_logging(wrapped.outer)
        wrapped.at = _unwrap_function_from_op_logging(wrapped.at)


def _wrap_function_for_op_logging(
    fn: Callable,
    graph: Graph,
    limit_attributes: bool = True,
    from_tracked_var: bool = False,
    stateful_classes: Optional[Iterable] = None,
    to_ivy: bool = False,
) -> Callable:
    """Wraps all the functions of a class/module so that these functions can be
    logged/stored while doing the preliminary forward pass.

    Parameters
    ----------
    fn
        function/method to potentially wrap.
    graph
        graph instance for the function we are compiling.
    limit_attributes
        limit attributes being added to the graph.
    from_tracked_var
        flag that indicates if the method being wrapped is from the TrackedVarProxy class.
    stateful_classes
        stateful classes that we also want to wrap and be included in the graph.

    Returns
    -------
        the wrapped function which would be called instead of the native function during
        the initial forward pass through the function we are compiling.
    """
    stateful_classes = tuple(ivy.default(stateful_classes, tuple()))

    def is_already_wrapped():
        if hasattr(fn, "wrapped_for_compiling"):
            if fn.wrapped_for_compiling == id(graph):
                return True
            return False
        return False

    # do not wrap default __init__
    if fn is object.__init__:
        return fn

    # Do not wrap the function:
    # (a) if it's a special method but not in ARRAY_BUILTINS
    # (b) if it's already wrapped
    # (c) if we are wrapping TrackedVarProxy and fn is in TRACKED_VAR_NON_WRAPPED_METHODS or tvp.RAW_RET_METHODS
    if (
        (
            hasattr(fn, "__name__")
            and not from_tracked_var
            and (fn.__name__[0] == "_" and fn.__name__ not in glob.ARRAY_BUILTINS)
        )
        or is_already_wrapped()
        or (
            from_tracked_var
            and fn.__name__ in tvp.NON_WRAPPED_METHODS + tvp.RAW_RET_METHODS
        )
    ):
        return fn

    @IIX._wraps(fn)
    def _tracing_function(*args, **kwargs):
        """
        This is the function that will be called instead of the native function e.g.
        `torch.add` when doing the operation logging forward pass. This wrapped
        function of course executes the original native function but also records a
        variety of information such as the ids of the outputs and the native function
        which produced them. This is to enable the compiler to connect functions
        together in the correct order, during the later construction of the graph.
        """
        # if logging is paused (as it is before the op logging forward pass begins or during
        # execution of some helpers etc), just execute original native function
        if glob.logging_paused:
            return fn(*args, **kwargs)

        if isinstance(fn, functools.partial):
            return fn(*args, **kwargs)

        target_framework = "ivy" if to_ivy else ivy.current_backend_str()

        # if the logging_stack is non-empty, we are already
        # in the middle of adding a function to the graph
        if glob.logging_stack:
            # return if a function is already being logged on a higher level, and it's not a built-in which legitimately
            # might need to be nested, unless it's a built-in recursion loop (ie for __getattribute__) in which case return
            if glob.logging_stack[-1].__name__[0:2] != "__" or (
                glob.logging_stack[-1].__name__ == fn.__name__
                and args == args
                and kwargs == kwargs
            ):
                # if the function is part of vmap, or we are inside the scalar function,
                # continue logging.
                if (
                    hasattr(fn, "__name__")
                    and fn.__name__ in ("scalar_fn", "vectorized_fn", "vmap")
                    or glob.logging_stack[-1].__name__ == "scalar_fn"
                ):
                    pass
                else:
                    return fn(*args, **kwargs)

            # return if the current function is a (possibly reversed) built-in operator,
            # and the last entry of the logging stack is a version of that same operator
            elif fn.__name__.replace("r", "").replace("_", "") in glob.logging_stack[
                -1
            ].__name__.replace("r", "").replace("_", ""):
                return fn(*args, **kwargs)

        # attributes to ignore
        att_name = None
        if fn.__name__ in ["__getattr__", "__setattr__", "__getattribute__"]:
            att_name = args[1]
            # return if the attribute being retrieved is another built-in method
            if att_name[0:2] == "__":
                return fn(*args, **kwargs)
            # if the attribute is not recognized as one which can form part of the graph, then return
            if (
                limit_attributes
                and att_name
                not in glob.GRAPH_ATTRIBUTES[target_framework] + tvp.ATTRIBUTES
            ):
                # if this attribute tries to call another attribute that we are tracking,
                # proceed to pause logging here. This is done to prevent tracking of some attributes like
                # jax.shape when called internally via some private attributes that we do not wish to log like device_buffer etc.
                initial_logging_status = glob.logging_paused
                if tvp.should_not_be_logged(
                    fn.__name__, args, att_name, target_framework
                ):
                    glob.logging_paused = True
                ret = fn(*args, **kwargs)
                glob.logging_paused = initial_logging_status
                return ret

        # if none of the above exceptions apply, then we log the
        # function and add it to the stack to indicate this
        glob.logging_stack.append(fn)

        _tracked_var_backend_fn = fn

        # Store the wrapped method in case it's a valid method from TrackedVarProxy
        # The only exceptions are methods from TrackedVarProxyMeta classes since they
        # will not contain any backend var in them, only the TrackedVarProxy class instance itself
        if (
            from_tracked_var
            and fn.__name__ != "input_to_output"
            and fn.__name__[3:] not in tvp.AVAILABLE_RAW_RET
            and not (
                hasattr(fn, "__qualname__")
                and any(
                    [
                        metacls in fn.__qualname__
                        for metacls in tvp.TRACKED_VAR_PROXY_META_CLASSES
                    ]
                )
            )
        ):
            # Need to support tracking enums in the args since they are subclasses of int
            # and use the same tracked methods
            _arg = args[0]

            initial_logging_status = glob.logging_paused
            glob.logging_paused = True

            _arg = (
                track(
                    _arg,
                    with_numpy=graph._with_numpy,
                    stateful_classes=stateful_classes,
                )
                if IIX._is_untracked_enum(_arg)
                else _arg
            )

            glob.logging_paused = initial_logging_status

            # There is no _tracked_var_backend_fn for iterator proxies
            if fn.__name__ not in tvp.ITERATOR_METHODS:
                try:
                    _arg_var = _arg.get_var()
                except AttributeError:
                    raise AttributeError(
                        f"{type(_arg).__name__} is not a valid tracked proxy"
                    )
                else:
                    try:
                        _tracked_var_backend_fn = getattr(
                            _arg_var.__class__, fn.__name__
                        )
                    except AttributeError:
                        raise AttributeError(
                            f"{fn.__name__} method not implemented"
                            f" in tracked {_arg_var.__class__}"
                        )

        # If the function is __init__, strip the first argument.
        if fn.__name__ == "__init__":
            new_args = args[1:]
            f_arg = args[0]

            def wrap_init(init_fn):
                def __init__(*a, **k):
                    return init_fn(f_arg, *a, **k)

                return __init__

            fn_ = wrap_init(fn)
            args = new_args
        else:
            fn_ = fn

        # store information about this vmap function which will later be
        # used in reconstructing vmap
        if fn.__name__ == "vmap":
            args, kwargs = process_vmap_fn(graph, fn, args, kwargs)

        # Flag to determine if the function is from the iterator classes for TVPs
        from_tracked_var_iterators = (
            any(
                [
                    itercls in fn.__qualname__
                    for itercls in tvp.tracked_var_proxy_iter_classes
                ]
            )
            if hasattr(fn, "__qualname__")
            else False
        )

        # Flag to determine if the function forms a part of an ongoing iterator
        # chain i.e. __iter__ ---> __next__ ---> fn
        from_iterator_chain = from_tracked_var_iterators or (
            not from_tracked_var_iterators
            and from_tracked_var
            and fn.__name__ in tvp.BUILTIN_ITERATOR_METHODS
        )

        # Destroy the iterator chain if we are done chaining iterator methods and next fn
        # is not a valid iterator method
        if not from_iterator_chain and len(glob.iterator_chain) > 0:
            glob.iterator_chain.pop()

        # args and kwargs to native arrays
        if not to_ivy:
            args = ivy.to_native(
                args, True, cont_inplace=True, to_ignore=tvp._to_ignore
            )
            kwargs = ivy.to_native(
                kwargs, True, cont_inplace=True, to_ignore=tvp._to_ignore
            )

        args = ivy.nested_map(args, _batched_tracer_to_array, to_ignore=tvp._to_ignore)
        kwargs = ivy.nested_map(
            kwargs, _batched_tracer_to_array, to_ignore=tvp._to_ignore
        )

        if fn.__name__ == "scalar_fn" and graph._to_ivy:
            args = ivy.nested_map(
                args, native_array_to_frontend, to_ignore=tvp._to_ignore
            )
            kwargs = ivy.nested_map(
                kwargs, native_array_to_frontend, to_ignore=tvp._to_ignore
            )

        # check if there are slices with TrackedVars inside
        arg_tracked_slices_idxs = ivy.nested_argwhere(
            args, tvp.is_tracked_slice, to_ignore=tvp._to_ignore
        )
        kwarg_tracked_slices_idxs = ivy.nested_argwhere(
            kwargs, tvp.is_tracked_slice, to_ignore=tvp._to_ignore
        )
        # convert slices to slice-lists
        args = ivy.map_nest_at_indices(args, arg_tracked_slices_idxs, tvp.slice_to_list)
        kwargs = ivy.map_nest_at_indices(
            kwargs, kwarg_tracked_slices_idxs, tvp.slice_to_list
        )

        # get idxs from TrackedVarProxies
        args_tracked_var_idxs = ivy.nested_argwhere(
            args, IIX._is_tracked_variable, to_ignore=tvp._to_ignore
        )
        kwargs_tracked_var_idxs = ivy.nested_argwhere(
            kwargs, IIX._is_tracked_variable, to_ignore=tvp._to_ignore
        )

        # Pause logging to avoid logging iterating over TVPs since
        # we support this now
        glob.logging_paused = True

        node = Node()
        (
            node.arg_tracked_idxs,
            arg_parameters,
            node.arg_param_ids,
            node.arg_param_types,
            node.arg_param_var_flags,
            _,
        ) = _record_parameters_info(
            args, to_ivy, graph._with_numpy, stateful_classes, args_tracked_var_idxs
        )

        (
            node.kwarg_tracked_idxs,
            _,
            node.kwarg_param_ids,
            node.kwarg_param_types,
            node.kwarg_param_var_flags,
            _,
        ) = _record_parameters_info(
            kwargs, to_ivy, graph._with_numpy, stateful_classes, kwargs_tracked_var_idxs
        )

        glob.logging_paused = False

        # set the backend function
        backend_fn = fn_

        if from_tracked_var:
            # Store the TrackedVarProxy to update its var instead of creating a new one
            tracked_var_instance = args[0]
            # If the function comes a TrackedVarProxy instance, the function we should store is the
            # corresponding function from the wrapped var. This way, once the function is compiled,
            # the methods from the original variables are called
            backend_fn = _tracked_var_backend_fn

        # Need to untrack the args/kwargs here except when the function is one of the
        # iterator protocols for tracked proxies since these iterator protocols
        # take tracked proxies as inputs not the original backend vars
        untrack_post_ret = False
        if not (
            from_tracked_var_iterators
            or (
                not from_tracked_var_iterators
                and from_tracked_var
                and fn.__name__ in tvp.DICT_ITERATOR_METHODS
            )
        ):
            args = untrack(args)
            kwargs = untrack(kwargs)
        else:
            untrack_post_ret = True

        # convert slice-lists to slices
        args = ivy.map_nest_at_indices(args, arg_tracked_slices_idxs, tvp.list_to_slice)
        kwargs = ivy.map_nest_at_indices(
            kwargs, kwarg_tracked_slices_idxs, tvp.list_to_slice
        )

        # call the original native function. We pause logging here since we don't
        # want to add any functions called inside `backend_fn` to the graph as well
        if backend_fn.__name__ == "scalar_fn":
            # store the current fns we have traced
            graph._tmp_subgraph_id_to_function.insert(
                -1, graph._subgraph_id_to_function
            )

            # reset/initialize the subgraph data structures
            graph._subgraph_id_to_function = dict()

        glob.logging_paused = (
            False if backend_fn.__name__ in ("scalar_fn", "vectorized_fn") else True
        )
        try:
            if fn.__name__ == "vectorized_fn":
                # get the scalar fn
                scalar_fn = graph.scalar_fn
                # strip off the batch dimension of the inputs
                new_args = ivy.nested_map(
                    args,
                    lambda x: x[0] if is_array(x) else x,
                    to_ignore=tvp._to_ignore,
                    shallow=False,
                )
                new_kwargs = ivy.nested_map(
                    kwargs,
                    lambda k, v: v[0] if is_array(v) else v,
                    to_ignore=tvp._to_ignore,
                    shallow=False,
                )
                # trace into the scalar fn with logging paused
                scalar_fn(*new_args, **new_kwargs)
            else:
                ret = backend_fn(*args, **kwargs)

            if backend_fn.__name__ == "vmap":
                # change the name to track in the logging stack
                setattr(ret, "__name__", "vectorized_fn")

                # delete the wrapped_for_compiling attribute if exists
                if hasattr(ret, "wrapped_for_compiling"):
                    delattr(ret, "wrapped_for_compiling")

                # wrap the function to enable logging
                ret = _wrap_function_for_op_logging(ret, graph, to_ivy=graph._to_ivy)

                # store the scalar fn
                graph.scalar_fn = args[0]

            if backend_fn.__name__ == "vectorized_fn":
                # re-compute the output as tracing gave incorrect results
                glob.logging_paused = True
                ret = backend_fn(*args, **kwargs)
        except Exception as e:
            glob.logging_paused = False
            glob.logging_stack.pop()
            glob.iterator_chain.clear()
            raise e
        glob.logging_paused = False

        # Need to untrack here for the dependent params to be correctly deleted
        # in case the function was one of the iterator protocols of the tracked proxies,
        # because we didn't untrack the args/kwargs above.
        args = untrack(args) if untrack_post_ret else args
        kwargs = untrack(kwargs) if untrack_post_ret else kwargs

        ret_id = id(ret)
        is_inplace_fn = (
            (
                ret_id in [id(arg) for arg in args]
                or ("out" in kwargs and ret_id == id(kwargs["out"]))
            )
            and not from_tracked_var
            and fn.__name__ not in ["to", "type", "as_tensor"]
        )
        # to, type, as_tensor sometimes just return their inputs- they dont edit inplace

        # convert slices to slices-lists
        args = ivy.map_nest_at_indices(args, arg_tracked_slices_idxs, tvp.slice_to_list)
        kwargs = ivy.map_nest_at_indices(
            kwargs, kwarg_tracked_slices_idxs, tvp.slice_to_list
        )

        # find ids for dependent paramaters in the args/kwargs so that we don't track ret
        # if the fn had no dependent parameters in the args/kwargs
        input_parameter_ids = node.arg_param_ids + node.kwarg_param_ids
        with_dependent_parameters = any(
            [x in glob.dependent_ids for x in input_parameter_ids]
        )

        # added so tensorflow inplace variable updates work properly (return is set
        # to first arg since this is the variable updated inplace)
        # provide return value for __setattr__ and similar functions
        inplace_fn = False
        if (
            not from_tracked_var
            and fn.__name__
            in ["__setattr__"]
            + glob.INPLACE_METHODS_WITHOUT_RET[target_framework]
            + glob.INPLACE_FUNCTIONS_WITHOUT_RET[target_framework]
            + (
                glob.INPLACE_METHODS_WITHOUT_RET["numpy"]
                + glob.INPLACE_FUNCTIONS_WITHOUT_RET["numpy"]
                if graph._with_numpy
                else []
            )
        ) or (from_tracked_var and fn.__name__ in tvp.INPLACE_METHODS_WITHOUT_RET):
            ret = args[0]
            inplace_fn = True

        # track output if the fn is a TrackedVarProxy method or if ret is an instance of a class
        # that should always be tracked, however don't track ret if the inputs had no dependent params
        glob.logging_paused = True
        if with_dependent_parameters and (
            from_tracked_var
            or tvp.should_be_tracked(fn.__name__, att_name, ret, target_framework)
        ):
            if fn.__name__ not in tvp.RAW_RET_METHODS:
                if fn.__name__ in tvp.INPLACE_METHODS_WITHOUT_RET:
                    ret = tracked_var_instance
                else:
                    ret = track(
                        ret,
                        with_numpy=graph._with_numpy,
                        stateful_classes=stateful_classes,
                        _deepcopy=False,
                    )

        # remove parameters from args and kwargs
        args = ivy.map_nest_at_indices(
            args,
            node.arg_tracked_idxs,
            lambda x: IIX._delete_parameter(x, graph),
        )
        kwargs = ivy.map_nest_at_indices(
            kwargs,
            node.kwarg_tracked_idxs,
            lambda x: IIX._delete_parameter(x, graph),
        )

        # convert return to list
        ret_listified = False
        if (
            isinstance(ret, tuple)
            and not hasattr(ret, "is_tracked_proxy")
            and not hasattr(ret, "_fields")
        ):
            tuple_type = type(ret)
            ret_list = list(ret)
        else:
            ret_list = [ret]
            ret_listified = True

        glob.logging_paused = True
        new_ret = to_custom_numpy_type(ret_list, with_numpy=graph._with_numpy)
        glob.logging_paused = False

        if to_ivy:
            ret_list = ivy.to_native(ret_list, nested=True, to_ignore=tvp._to_ignore)

        output_tracked_var_idxs = ivy.nested_argwhere(
            ret_list, IIX._is_tracked_variable, to_ignore=tvp._to_ignore
        )

        # Pause logging to avoid logging iterating over TVPs since
        # we support this now
        glob.logging_paused = True

        (
            node.output_tracked_idxs,
            _,
            node.output_param_ids,
            node.output_param_types,
            node.output_param_var_flags,
            node.output_param_shapes,
        ) = _record_parameters_info(
            ret_list,
            to_ivy,
            graph._with_numpy,
            stateful_classes,
            output_tracked_var_idxs,
        )

        glob.logging_paused = False

        # return if there are no tracked outputs
        if not node.output_tracked_idxs:
            glob.logging_stack.pop()
            return new_ret[0] if ret_listified else tuple_type(new_ret)

        # clone the param when getting an attribute, to preserve uniqueness in the graph.
        # due to functional nature, any output of any function cannot have colliding param id with existing objects.
        # so we clone the existing object (if it exists) and assign it a new param id.
        if fn.__name__ in ["__getattr__", "__getattribute__"]:
            # update the param_id for each param in the retrieved attribute in the graph
            ret_list = ivy.map_nest_at_indices(
                ret_list, node.output_tracked_idxs, lambda x: IIX._clone_param(x, graph)
            )

        # find all those outputs which have the same id as one of the inputs
        # we will have to clone those outputs to preserve uniqueness in the graph
        duplicates = list()
        for i, ret_id in enumerate(node.output_param_ids):
            if ret_id in input_parameter_ids:
                duplicates.append(i)

        # clone all repeated return parameters to give unique parameter ids in the graph
        duplicate_tracked_idxs = [node.output_tracked_idxs[i] for i in duplicates]
        ret_list = ivy.map_nest_at_indices(
            ret_list, duplicate_tracked_idxs, lambda x: IIX._clone_param(x, graph)
        )

        # get return param ids after cloning
        output_vals = ivy.multi_index_nest(ret_list, node.output_tracked_idxs)
        node.output_param_ids = [IIX._get_unique_id(x) for x in output_vals]

        gen_fns = glob.GENERATOR_FUNCTIONS[target_framework]
        if graph._with_numpy and target_framework != "numpy":
            gen_fns += glob.GENERATOR_FUNCTIONS["numpy"]

        # Flag to decide on whether to add output ids to set of
        # dependent ids
        add_gen_fn = fn.__name__ in gen_fns and graph._include_generators

        # maybe add to set of dependent_ids
        if add_gen_fn or with_dependent_parameters:
            [glob.dependent_ids.add(id_) for id_ in node.output_param_ids]

        args, kwargs = _cache_constant_args(args, kwargs, node, to_ivy)

        glob.logging_paused = True

        # store info about this node
        node.backend_fn = backend_fn

        node.from_tracked_var = from_tracked_var
        node.from_tracked_var_iterators = from_tracked_var_iterators
        node.from_iterator_chain = from_iterator_chain
        node.is_inplace_fw_fn = (
            ivy.current_backend_str() in ["torch", "numpy"] and is_inplace_fn
        )

        try:
            sig = inspect.signature(fn)
            sig_keys = list(sig.parameters.keys())
        except ValueError:
            sig_keys = list()
        node.arg_n_kwarg_reprs = _get_argument_reprs(sig_keys, args, kwargs)
        ret_placeholder = ivy.set_nest_at_indices(
            ret_list,
            node.output_tracked_idxs,
            lambda x_: IIX._delete_parameter(x_, graph),
            shallow=False,
        )
        node.output = ret_placeholder
        node.remove_output_tuple = (
            isinstance(ret, tuple)
            and not isinstance(ret, tvp._to_ignore)
            and len(ret) == 1
        )
        node.output_reprs = _get_output_reprs(ret_list)

        node.timestamp = time.perf_counter()
        node.terminal = True
        node.is_constant = len(input_parameter_ids) == 0 and (
            not graph._include_generators or fn.__name__ not in gen_fns
        )
        node.inplace_fn = inplace_fn
        node.with_tracked_slices = arg_tracked_slices_idxs + kwarg_tracked_slices_idxs

        glob.logging_paused = False

        if backend_fn.__name__ == "scalar_fn":
            ret = new_ret[0] if ret_listified else tuple(new_ret)
            ret = ret if isinstance(ret, tuple) else (ret,)
            process_scalar_fn(graph, backend_fn, arg_parameters, ret)

        if backend_fn.__name__ == "vectorized_fn":
            node = process_vectorized_fn(graph, node)

        prev_fn = False
        if (
            len(glob.logging_stack) > 1
            and glob.logging_stack[-2].__name__ == "scalar_fn"
        ):
            fns_in = add_incoming_subgraph_fns(
                graph,
                fn,
                input_parameter_ids,
            )
        # Iterator methods form a chain but the outputs of previous nodes (__next__)
        # is not the input of the next node (__next__) so need to account for that
        elif from_iterator_chain and len(glob.iterator_chain) > 0:
            fns_in = [glob.iterator_chain.pop()]
            prev_fn = True
        else:
            fns_in = [
                graph._id_to_function[id_]
                for id_ in input_parameter_ids
                if id_ in graph._id_to_function
            ]

        # add this function as the outgoing function of the incoming functions
        if node.output_param_ids:
            for fn_in in fns_in:
                fn_in.terminal = False
                if node not in fn_in.fns_out:
                    fn_in.fns_out.append(node)

        node.fns_in = fns_in
        node.fns_out = list()

        # For iterator chains s.t.  __iter__ ---> __next__ ---> __next__
        node.prev_fn = fns_in[0] if prev_fn else None

        # assign the same name to `node` as it is in the backend
        node.__repr__ = lambda: node.__name__
        if fn.__name__ == "vectorized_fn":
            node.__name__ = "vmap"
        else:
            node.__name__ = fn.__name__

        # add this function to the graph for each output id
        if fn.__name__ not in (
            "scalar_fn",
            "safe_map",
        ):  # do not add these functions to the main graph
            if (
                len(glob.logging_stack) > 1
                and glob.logging_stack[-2].__name__ == "scalar_fn"
            ):
                add_subgraph_fns_to_dict(graph, fn, node, node.output_param_ids)
            else:
                for id_ in node.output_param_ids:
                    if id_ not in graph._id_to_function:
                        graph.add_fn_to_dict(id_, node)

        # Add the current fn to the iterator chain to correctly chain consecutive
        # __next__ nodes in the graph
        if from_iterator_chain:
            glob.iterator_chain.append(node)

        # remove function from stack, now logging has occurred
        glob.logging_stack.pop()

        # return the function output
        return new_ret[0] if ret_listified else tuple_type(new_ret)

    _tracing_function.wrapped_for_compiling = id(graph)
    _wrap_numpy_ufuncs(_tracing_function, fn, graph)
    return _tracing_function


def _unwrap_function_from_op_logging(function_wrapped):
    if hasattr(function_wrapped, "wrapped_for_compiling"):
        glob.wrapped_fns[id(function_wrapped)] = (
            function_wrapped,
            function_wrapped.__wrapped__,
        )
        _unwrap_numpy_ufuncs(function_wrapped, function_wrapped.__wrapped__)
        return function_wrapped.__wrapped__
    return function_wrapped


def _should_be_wrapped(obj):
    return callable(obj) and not inspect.isclass(obj)


FUNC_TO_PATH = {}


def _wrap_or_unwrap_module(
    wrap_or_unwrap_fn,
    module,
    framework=None,
    to_ivy=False,
):
    framework = ivy.current_backend_str() if framework is None else framework
    framework = "ivy" if to_ivy else framework
    for k in dir(module):
        v = getattr(module, k)
        if (
            k in glob.FUNCTIONS_ATTRS_NOT_TO_WRAP[framework]
            or k[0] == "_"
            or not _should_be_wrapped(v)
        ):
            continue
        try:
            setattr(module, k, wrap_or_unwrap_fn(v))
            if not hasattr(v, "wrapped_for_compiling"):
                FUNC_TO_PATH[v] = module.__name__ + "." + k
        except Exception:
            pass


def _wrap_or_unwrap_class(
    wrap_or_unwrap_fn, cls, cls_path=None, framework=None, to_ivy=False
):
    if cls is None:
        return
    framework = ivy.current_backend_str() if framework is None else framework
    framework = "ivy" if to_ivy else framework
    for k in dir(cls):
        attr = getattr(cls, k)
        if k in glob.FUNCTIONS_ATTRS_NOT_TO_WRAP[framework] or not _should_be_wrapped(
            attr
        ):
            continue
        if ivy.current_backend_str() == "jax":
            import jaxlib

            if hasattr(jaxlib.xla_extension, "ArrayImpl"):
                if attr == jaxlib.xla_extension.ArrayImpl.__init__:
                    continue
        try:
            setattr(cls, k, wrap_or_unwrap_fn(attr))
        except Exception:
            pass
        if cls_path is not None:
            FUNC_TO_PATH[attr] = ".".join(cls_path) + "." + k


def _wrap_or_unwrap_intenum(wrap_or_unwrap_fn, int_enum_proxy):
    val_source = tvp.ATTRS_TO_WRAP_AND_MAP[int_enum_proxy.__name__]["mapping"]
    to_map = tvp.ATTRS_TO_WRAP_AND_MAP[int_enum_proxy.__name__].get("to_map")
    to_ignore = tvp.ATTRS_TO_WRAP_AND_MAP[int_enum_proxy.__name__].get("to_ignore")
    dir_source = [_m for _m in dir(val_source) if _m not in to_ignore]
    dir_source = (
        [_m for _m in dir(val_source) if _m not in to_ignore]
        if "*" in to_map
        else dir_source
    )
    dir_source = (
        [_m for _m in dir(val_source) if _m in to_map]
        if "*" in to_ignore
        else dir_source
    )
    for k in dir_source:
        v = getattr(val_source, k)
        if not _should_be_wrapped(v):
            continue
        try:
            setattr(int_enum_proxy, k, wrap_or_unwrap_fn(v))
        except Exception:
            pass


def _load_classes_from(ctw: List):
    classes = []
    for _ctw in ctw:
        try:
            classes.append(getattr(importlib.import_module(_ctw[0]), _ctw[1]))
        except AttributeError:
            classes.append(None)
    return classes


def _load_modules_from(mtw: List):
    modules = []
    for _mtw in mtw:
        try:
            modules.append(importlib.import_module(_mtw))
        except:
            pass
    return modules


def _wrap_functions_for_op_logging(
    graph, stateful_classes=None, to_ivy=False, with_numpy=False
):
    glob.wrapped_fns = {}
    target = "ivy" if to_ivy else ivy.current_backend_str()
    private_class_paths = glob.PRIVATE_CLASSES_TO_WRAP(target)
    private_classes = _load_classes_from(private_class_paths)
    for cls, path in zip(private_classes, private_class_paths):
        _wrap_or_unwrap_class(
            lambda fn: _wrap_function_for_op_logging(
                fn, graph, stateful_classes=private_classes, to_ivy=to_ivy
            ),
            cls,
            path,
            to_ivy=to_ivy,
        )
    class_paths = glob.CLASSES_TO_WRAP[target]
    classes = _load_classes_from(class_paths)
    for cls, path in zip(classes, class_paths):
        _wrap_or_unwrap_class(
            lambda fn: _wrap_function_for_op_logging(
                fn, graph, stateful_classes=private_classes, to_ivy=to_ivy
            ),
            cls,
            path,
            to_ivy=to_ivy,
        )
    if target == "tensorflow":
        import tensorflow as tf

        # these tf modules can't be imported from a string, so adding them manually
        modules_to_wrap = [
            tf.compat.v2.compat.v1.nn,
            tf.compat.v2.compat.v1.linalg,
            tf.compat.v2.compat.v1.math,
        ]
    else:
        modules_to_wrap = []
    modules_to_wrap += _load_modules_from(glob.MODULES_TO_WRAP[target])
    for module in modules_to_wrap:
        _wrap_or_unwrap_module(
            lambda fn: _wrap_function_for_op_logging(
                fn, graph, stateful_classes=private_classes, to_ivy=to_ivy
            ),
            module,
            to_ivy=to_ivy,
        )

    # wrap numpy after wrapping modules of current backend. wrapping before causes
    # issues with modules like jax.scipy.optimise where they import like
    # `from numpy import asarray` which would then import the wrapped version of
    # numpy.asarray, and would not be unwrapped afterwards. this is only a problem
    # with modules in jax.scipy because they are not initialised upon `import jax`,
    # and so will be initialised when we import them to wrap.
    if with_numpy:
        for custom_class in custom_np_classes:
            _wrap_or_unwrap_class(
                lambda fn: _wrap_function_for_op_logging(fn, graph),
                custom_class,
                framework="numpy",
                to_ivy=to_ivy,
            )
        for module in _load_modules_from(glob.MODULES_TO_WRAP["numpy"]):
            _wrap_or_unwrap_module(
                lambda fn: _wrap_function_for_op_logging(fn, graph),
                module,
                framework="numpy",
                to_ivy=to_ivy,
            )

    # wrap TrackedVarProxy classes
    for proxy_class in tvp.proxy_classes():
        if proxy_class.__name__ in tvp.ATTRS_TO_WRAP_AND_MAP:
            _wrap_or_unwrap_intenum(
                lambda fn: _wrap_function_for_op_logging(
                    fn, graph, from_tracked_var=True
                ),
                proxy_class,
            )
        _wrap_or_unwrap_class(
            lambda fn: _wrap_function_for_op_logging(fn, graph, from_tracked_var=True),
            proxy_class,
        )

    # wrap functorch.vmap
    if target == "torch":
        try:
            import functorch

            functorch.vmap = _wrap_function_for_op_logging(
                functorch.vmap,
                graph,
            )
        except:
            # do not wrap functorch.vmap if it is not installed,
            # which can occur when using torch versions < 1.13.0
            pass

    # wrap any functions in the arguments
    graph._args = ivy.nested_map(
        graph._args,
        lambda x: _wrap_function_for_op_logging(x, graph)
        if isinstance(x, (FunctionType, BuiltinFunctionType))
        else x,
    )
    graph._kwargs = ivy.nested_map(
        graph._kwargs,
        lambda x: _wrap_function_for_op_logging(x, graph)
        if isinstance(x, (FunctionType, BuiltinFunctionType))
        else x,
    )

    # wrap stateful classes
    stateful_classes = ivy.default(stateful_classes, [])
    for cls in stateful_classes:
        assert hasattr(cls, "__setattr__") and (
            hasattr(cls, "__getattr__") or hasattr(cls, "__getattribute__")
        )
        assert hasattr(cls, "__init__")
        cls.__init__ = _wrap_function_for_op_logging(
            cls.__init__,
            graph,
            limit_attributes=False,
            stateful_classes=stateful_classes,
        )
        cls.__setattr__ = _wrap_function_for_op_logging(
            cls.__setattr__,
            graph,
            limit_attributes=False,
            stateful_classes=stateful_classes,
        )
        if hasattr(cls, "__getattr__"):
            cls.__getattr__ = _wrap_function_for_op_logging(
                cls.__getattr__,
                graph,
                limit_attributes=False,
                stateful_classes=stateful_classes,
            )
        if hasattr(cls, "__getattribute__"):
            cls.__getattribute__ = _wrap_function_for_op_logging(
                cls.__getattribute__,
                graph,
                limit_attributes=False,
                stateful_classes=stateful_classes,
            )


def _unwrap_functions_from_op_logging(
    stateful_classes=None, to_ivy=False, with_numpy=False
):
    glob.wrapped_fns = {}
    target = "ivy" if to_ivy else ivy.current_backend_str()
    if with_numpy:
        for custom_class in custom_np_classes:
            _wrap_or_unwrap_class(
                _unwrap_function_from_op_logging,
                custom_class,
                framework="numpy",
            )
        for module in _load_modules_from(glob.MODULES_TO_WRAP["numpy"]):
            _wrap_or_unwrap_module(
                _unwrap_function_from_op_logging,
                module,
                framework="numpy",
            )

    # unwrap proxy classes
    for proxy_class in tvp.proxy_classes():
        if proxy_class.__name__ in tvp.ATTRS_TO_WRAP_AND_MAP:
            _wrap_or_unwrap_intenum(
                _unwrap_function_from_op_logging,
                proxy_class,
            )
        _wrap_or_unwrap_class(
            _unwrap_function_from_op_logging,
            proxy_class,
        )
    modules_to_unwrap = _load_modules_from(glob.MODULES_TO_WRAP[target])
    if target == "tensorflow":
        import tensorflow as tf

        modules_to_unwrap += [
            tf.compat.v2.compat.v1.nn,
            tf.compat.v2.compat.v1.linalg,
            tf.compat.v2.compat.v1.math,
        ]
    for module in modules_to_unwrap:
        _wrap_or_unwrap_class(
            _unwrap_function_from_op_logging,
            module,
            to_ivy=to_ivy,
        )

    # unwrap backend classes
    ctu = glob.CLASSES_TO_WRAP[target]
    classes_to_unwrap = _load_classes_from(ctu) + stateful_classes
    for cls in classes_to_unwrap:
        _wrap_or_unwrap_class(
            _unwrap_function_from_op_logging,
            cls,
            to_ivy=to_ivy,
        )

    # unwrap private classes
    pctw = glob.PRIVATE_CLASSES_TO_WRAP(target)[::-1]
    priv_classes_to_wrap = _load_classes_from(pctw)
    for pctw in priv_classes_to_wrap:
        _wrap_or_unwrap_class(
            _unwrap_function_from_op_logging,
            pctw,
            to_ivy=to_ivy,
        )

    # unwrap functorch.vmap
    if target == "torch":
        try:
            import functorch

            functorch.vmap = _unwrap_function_from_op_logging(
                functorch.vmap,
            )
        except:
            pass

    # unwrap stateful classes
    stateful_classes = ivy.default(stateful_classes, [])
    for cls in stateful_classes:
        assert hasattr(cls, "__init__")
        cls.__init__ = _unwrap_function_from_op_logging(cls.__init__)
        assert hasattr(cls, "__setattr__") and (
            hasattr(cls, "__getattr__") or hasattr(cls, "__getattribute__")
        )
        cls.__setattr__ = _unwrap_function_from_op_logging(cls.__setattr__)
        if hasattr(cls, "__getattr__"):
            cls.__getattr__ = _unwrap_function_from_op_logging(cls.__getattr__)
        if hasattr(cls, "__getattribute__"):
            cls.__getattribute__ = _unwrap_function_from_op_logging(
                cls.__getattribute__
            )