# global
from typing import Callable, List, Optional, Tuple, Type, Any
from types import FunctionType
import os
import sys
import copy
import json
import inspect
import collections
import numpy as np

try:
    import networkx as nx
except ModuleNotFoundError:
    nx = None

try:
    from pyvis.network import Network
except ModuleNotFoundError:
    Network = None

# local
import ivy
from ivy.functional.ivy.gradients import _is_variable
from graph_compiler.param import Param
from graph_compiler import tracked_var_proxy as tvp
from graph_compiler import globals as glob
from graph_compiler import source_gen as sg  # infuse
from graph_compiler.numpy_proxy import NewNDArray
from graph_compiler.helpers import (
    _get_shape,
    _get_unique_id,
    _terminal_ids_to_key,
    _is_tracked_variable,
    _find_missing_frontends,
    _format_missing_frontends_msg,
)
from graph_compiler.conversion import (
    is_frontend_array,
    to_custom_numpy_type,
    to_native,
    is_array,
    native_array_to_frontend,
    array_to_new_backend,
    nest_array_to_new_backend,
)
from graph_compiler.visualisation import (
    _args_str_from_fn,
    _output_str_from_fn,
    _param_to_label,
    _copy_func,
)


class Graph:
    def __init__(
        self,
        fn: Callable,
        *args: Any,
        stateful: Optional[List] = None,
        arg_stateful_idxs: Optional[List] = None,
        kwarg_stateful_idxs: Optional[List] = None,
        include_generators: bool = True,
        array_caching: bool = True,
        with_numpy: bool = False,
        source: Optional[str] = None,
        target: Optional[str] = None,
        to_ivy: bool = False,
        empty: bool = False,
        **kwargs: Any,
    ):
        # config
        self._include_generators = include_generators
        self._array_caching = array_caching
        self._with_numpy = with_numpy
        self._transpiling = source is not None
        self._orig_recursion_limit = sys.getrecursionlimit()
        self.backend = source if self._transpiling else ivy.current_backend_str()
        self.to = target
        self._to_ivy = to_ivy

        # track args passed through gc.track()
        arg_tracked_var_idxs = ivy.nested_argwhere(
            args, _is_tracked_variable, to_ignore=tvp._to_ignore
        )
        self.arg_tracked_var_idxs = arg_tracked_var_idxs
        kwarg_tracked_var_idxs = ivy.nested_argwhere(
            kwargs, _is_tracked_variable, to_ignore=tvp._to_ignore
        )
        self.kwarg_tracked_var_idxs = kwarg_tracked_var_idxs

        # stateful
        self._stateful = ivy.default(stateful, [])
        self._stateful_classes = tuple([x.__class__ for x in self._stateful])
        self._stateful_param_ids = [id(x) for x in self._stateful]
        self._stateful_param_var_flags = [
            _is_variable(x, exclusive=True) for x in self._stateful
        ]
        self._stateful_param_shapes = [_get_shape(x) for x in self._stateful]

        # all stateful
        arg_stateful_idxs = ivy.default(arg_stateful_idxs, [])
        kwarg_stateful_idxs = ivy.default(kwarg_stateful_idxs, [])
        stateful_args = ivy.multi_index_nest(args, arg_stateful_idxs)
        stateful_kwargs = ivy.multi_index_nest(kwargs, kwarg_stateful_idxs)
        self._all_stateful = self._stateful + stateful_args + stateful_kwargs
        self._all_stateful_classes = tuple([x.__class__ for x in self._all_stateful])
        self._all_stateful_param_ids = [id(x) for x in self._all_stateful]
        self._all_stateful_param_var_flags = [
            _is_variable(x, exclusive=True) for x in self._all_stateful
        ]
        self._all_stateful_param_shapes = [_get_shape(x) for x in self._all_stateful]

        # stateful clone id dict
        self._stateful_clone_id_dict = dict(
            zip(self._all_stateful_param_ids, self._all_stateful_param_ids)
        )

        # function being compiled into a graph
        self._fn = fn
        if isinstance(fn, FunctionType):
            self.__name__ = fn.__name__
        elif isinstance(fn, object):
            self.__name__ = type(fn).__name__

        # function args and kwargs
        try:
            self._fn_signature = (
                dict(inspect.signature(self._fn).parameters) if fn else {}
            )
        except:
            self._fn_signature = {}

        if self._transpiling:
            args = ivy.nested_map(
                args, native_array_to_frontend, to_ignore=tvp._to_ignore
            )
            kwargs = ivy.nested_map(
                kwargs, native_array_to_frontend, to_ignore=tvp._to_ignore
            )

        # positional args
        args = to_custom_numpy_type(args, with_numpy=self._with_numpy)
        arg_kwarg_tracked_fn = lambda x: ivy.is_native_array(x) or (
            self._with_numpy and isinstance(x, NewNDArray)
        )
        self._args = list(args)
        args = to_native(args, cont_inplace=True, to_ignore=tvp._to_ignore)
        self._arg_tracked_idxs = (
            ivy.nested_argwhere(args, arg_kwarg_tracked_fn)
            + arg_stateful_idxs
            + arg_tracked_var_idxs
        )
        arg_parameters = ivy.multi_index_nest(args, self._arg_tracked_idxs)
        self._arg_param_ids = [_get_unique_id(a) for a in arg_parameters]
        [glob.dependent_ids.add(id_) for id_ in self._arg_param_ids]
        self._arg_param_types = [a.__class__ for a in arg_parameters]
        self._arg_param_var_flags = [
            _is_variable(a, exclusive=True, to_ignore=tvp._to_ignore)
            for a in arg_parameters
        ]
        self._arg_param_shapes = [_get_shape(a) for a in arg_parameters]

        # key-word args
        kwargs = to_custom_numpy_type(kwargs, with_numpy=self._with_numpy)
        self._kwargs = dict(**kwargs)
        kwargs = to_native(kwargs, cont_inplace=False, to_ignore=tvp._to_ignore)
        self._kwarg_tracked_idxs = (
            ivy.nested_argwhere(kwargs, arg_kwarg_tracked_fn)
            + kwarg_stateful_idxs
            + kwarg_tracked_var_idxs
        )
        kwarg_parameters = ivy.multi_index_nest(kwargs, self._kwarg_tracked_idxs)
        self._kwarg_param_ids = [_get_unique_id(v) for v in kwarg_parameters]
        [glob.dependent_ids.add(id_) for id_ in self._kwarg_param_ids]
        self._kwarg_param_types = [v.__class__ for v in kwarg_parameters]
        self._kwarg_param_var_flags = [
            _is_variable(v, exclusive=True, to_ignore=tvp._to_ignore)
            for v in kwarg_parameters
        ]
        self._kwarg_param_shapes = [_get_shape(v) for v in kwarg_parameters]

        assert (
            empty or len(self._arg_tracked_idxs) + len(self._kwarg_tracked_idxs) != 0
        ), "No parameters detected in the inputs."

        # output param ids
        self._output = None  # initialized during op logging
        self._output_tracked_idxs = None  # initialized during op logging
        self._output_param_ids = list()

        # op logging storage
        self._id_to_function = dict()
        self._id_to_parameter = dict()

        # connected flag
        self._outer_connected = False
        self._all_connected = False

        # functions in graph
        self._grouped_functions = dict()
        self._functions = list()

        # create additional graph attributes for handling vmap nodes/subgraphs in general
        self._tmp_subgraph_id_to_function = [{}]
        self._subgraph_id_to_function = dict()
        self._sub_graphs = dict()
        self.vmap_node_ids = list()

        # graph formatting
        self._inter_node_color = "#00CC00"  # same -> (0.0, 0.8, 0.0)
        self._stateful_node_color = "#E6B233"  # prev -> (0.9, 0.7, 0.2)
        self._io_node_color = "#8075FF"  # prev -> (0.4, 0.4, 1.0)
        self._var_node_color = "#FF6699"  # prev -> (1.0, 0.4, 0.6)
        self._node_size = 20
        self._input_functions = dict()
        self._output_functions = dict()

    @classmethod
    def empty(cls):
        "Initialize an empty Graph instance"
        return cls(fn=None, empty=True)

    # Properties #
    # ---------- #

    @property
    def _all_grouped_functions(self):
        # ToDo: make this order more optimal, in the same manner by
        # which each sub-graph order is optimal
        all_grouped_functions = list()
        for gfs in self._grouped_functions.values():
            for i, fs in enumerate(gfs):
                if len(all_grouped_functions) == i:
                    all_grouped_functions.append(list())
                all_grouped_functions[i] += fs
        return all_grouped_functions

    @property
    def _max_graph_height(self):
        return len(self._all_grouped_functions)

    # Getters and Setters #
    # ------------------- #

    def add_param(
        self,
        id_: int,
        ptype: Type[ivy.NativeArray],
        is_var: bool,
        shape: Tuple,
    ):
        self._id_to_parameter[id_] = Param(ptype, is_var, shape)

    def add_fn_to_dict(self, id_: int, fn: Callable):
        self._id_to_function[id_] = fn

    # Forward with Op Logging #
    # ----------------------- #

    def _compute_return(self) -> Tuple:
        """Runs the forward pass and returns the final output.

        Example
        -------
        >>> import ivy
        >>> from graph_compiler.compiler import _create_graph
        >>> ivy.set_backend("torch")
        >>> x = ivy.array([0., 32.])

        >>> def function(x):
        ...     y = ivy.mean(x)
        ...     z = ivy.sqrt(y)
        ...     return z

        >>> graph = _create_graph(function, x)
        >>> print(graph._compute_return())
        (ivy.array(4., dtype=float32),)
        """
        ret = self._fn(*self._args, **self._kwargs)
        glob.logging_paused = True
        return ret if isinstance(ret, tuple) and not hasattr(ret, "_fields") else (ret,)

    def _register_output(self, ret: Tuple):
        """Record information about the final output `ret` of the forward pass."""
        self._output = ret
        self._output_tracked_idxs = ivy.nested_argwhere(
            ret,
            lambda x: is_array(x, with_numpy=self._with_numpy)
            or id(x) in self._all_stateful_param_ids
            or _is_tracked_variable(x),
            to_ignore=tvp._to_ignore,
        )
        self._output_param_ids = [
            _get_unique_id(to_native(x, to_ignore=tvp._to_ignore))
            for x in ivy.multi_index_nest(list(ret), self._output_tracked_idxs)
        ]

        # find any inputs which were fed directly to the output, and update id_ and add identity function
        for i, id_ in enumerate(self._output_param_ids):
            if id_ in self._arg_param_ids + self._kwarg_param_ids:

                def input_to_output(a, _):
                    return a

                # this is here to avoid circular imports
                from graph_compiler.wrapping import _wrap_function_for_op_logging

                if id_ in self._arg_param_ids:
                    index = self._arg_param_ids.index(id_)
                    arg = ivy.index_nest(self._args, self._arg_tracked_idxs[index])
                else:
                    index = self._kwarg_param_ids.index(id_)
                    arg = ivy.index_nest(self._kwargs, self._kwarg_tracked_idxs[index])
                if is_frontend_array(arg):
                    arg = arg.ivy_array
                from_tracked_var = True if _is_tracked_variable(arg) else False
                input_to_output = _wrap_function_for_op_logging(
                    input_to_output, self, from_tracked_var=from_tracked_var
                )
                ret = input_to_output(arg, None)
                self._output_param_ids[i] = _get_unique_id(ret)

    def log_all_ops(self):
        """Run a forward pass with operation logging turned on,
        so that we can keep track of all the functions executed
        in the forward pass.
        """
        glob.logging_paused = False
        out = self._compute_return()
        glob.logging_paused = True
        self._register_output(out)

    # Function creation #
    # ----------------- #

    def stop_tracking_constant_fns_output(self, receiving_fn, id_):
        if not ivy.exists(receiving_fn):
            idx = self._output_param_ids.index(id_)
            del self._output_tracked_idxs[idx]
            del self._output_param_ids[idx]
            return
        if id_ in receiving_fn.arg_param_ids:
            idx = receiving_fn.arg_param_ids.index(id_)
            del receiving_fn.arg_tracked_idxs[idx]
            del receiving_fn.arg_param_ids[idx]
            del receiving_fn.arg_param_types[idx]
        if id_ in receiving_fn.kwarg_param_ids:
            idx = receiving_fn.kwarg_param_ids.index(id_)
            del receiving_fn.kwarg_tracked_idxs[idx]
            del receiving_fn.kwarg_param_ids[idx]
            del receiving_fn.kwarg_param_types[idx]
        # if there are no arguments other than the one we just deleted,
        # the function can be treated as a constant
        gen_fns = glob.GENERATOR_FUNCTIONS[ivy.current_backend_str()]
        if self._with_numpy and ivy.current_backend_str() != "numpy":
            gen_fns = gen_fns + glob.GENERATOR_FUNCTIONS["numpy"]

        receiving_fn.is_constant = (
            len(receiving_fn.arg_param_ids + receiving_fn.kwarg_param_ids) == 0
        ) and (not self._include_generators or receiving_fn.__name__ not in gen_fns)

    def get_param_recursive(
        self, id_: int, depth: int, receiving_fn: Optional[Callable] = None
    ):
        """This function is first called on the final output, and traverses backwards through
        the graph until we reach the inputs, keeping track of any parameters in _id_to_parameter
        and any functions called in the list _tmp_sub_functions.

        Parameters
        ----------
        id_
            parameter id
        depth
            the depth of the parameter in the graph, initialised as 0 the first time it's called
            since we start at the outputs which have depth 0.
        receiving_fn
            the function which the parameter is inputted to. On the first call it is None, since
            of course the final outputs of the graph won't be inputted into any function.
        """
        # return if the parameter is already in the dict or if we reach the graph inputs (as the inputs
        # are already in _id_to_parameter)
        if id_ in self._id_to_parameter:
            return

        gen_fns = glob.GENERATOR_FUNCTIONS[ivy.current_backend_str()]
        if self._with_numpy and ivy.current_backend_str() != "numpy":
            gen_fns = gen_fns + glob.GENERATOR_FUNCTIONS["numpy"]

        # obtain the function which generated the given output associated with `id_`
        if id_ in self._id_to_function:
            fn = self._id_to_function[id_]
            if not self._include_generators and fn.__name__ in gen_fns:
                # raise exception if we try to delete the output of a gen fn which has
                # tracked vars in inputs when include_generators is False
                if fn.arg_tracked_idxs or fn.kwarg_tracked_idxs:
                    raise Exception(
                        f"including generator functions is not permitted, but func: {fn.__name__}"
                        f" contains tracked vars in inputs."
                    )
                self.stop_tracking_constant_fns_output(receiving_fn, id_)
                return
        elif self._array_caching:
            self.stop_tracking_constant_fns_output(receiving_fn, id_)
            return
        else:
            raise Exception(
                "array caching is not permitted, but id {} was not found in _id_to_functions.".format(
                    id_
                )
            )
        # call function recursively on all inputs to the function unless the function
        # is from tracked_var_iterators since __next__ function calls have the
        # iterator object in the arguments which is the output of __iter__ (whereas
        # we want to recurse back to the previous __next__ function call to correctly
        # chain __iter__ and consecutive __next__ nodes)
        if fn.from_iterator_chain and fn.prev_fn:
            [
                self.get_param_recursive(input_id, depth + 1, fn)
                for input_id in copy.copy(fn.prev_fn.output_param_ids)
            ]
        else:
            [
                self.get_param_recursive(input_id, depth + 1, fn)
                for input_id in copy.copy(fn.arg_param_ids)
            ]
            [
                self.get_param_recursive(input_id, depth + 1, fn)
                for input_id in copy.copy(fn.kwarg_param_ids)
            ]
        # for any constant function when array caching is on, we delete its output as
        # an argument to any subsequent function calls (unless the next function in the
        # graph operates inplace- causing cached args to be changed on each call)
        next_fn_inplace = ivy.exists(receiving_fn) and receiving_fn.is_inplace_fw_fn
        if (
            self._array_caching
            and not next_fn_inplace
            and (
                fn.is_constant
                or (not self._include_generators and fn.__name__ in gen_fns)
            )
        ):
            for receiving_fn in fn.fns_out:
                for id_ in fn.output_param_ids:
                    if id_ in receiving_fn.arg_param_ids:
                        indices = [
                            i
                            for i, x in enumerate(receiving_fn.arg_param_ids)
                            if x == id_
                        ]
                        for idx in reversed(indices):
                            del receiving_fn.arg_tracked_idxs[idx]
                            del receiving_fn.arg_param_ids[idx]
                            del receiving_fn.arg_param_types[idx]
                    if id_ in receiving_fn.kwarg_param_ids:
                        indices = [
                            i
                            for i, x in enumerate(receiving_fn.kwarg_param_ids)
                            if x == id_
                        ]
                        for idx in reversed(indices):
                            del receiving_fn.kwarg_tracked_idxs[idx]
                            del receiving_fn.kwarg_param_ids[idx]
                            del receiving_fn.kwarg_param_types[idx]
                receiving_fn.is_constant = (
                    len(receiving_fn.arg_param_ids + receiving_fn.kwarg_param_ids) == 0
                ) and (
                    not self._include_generators or receiving_fn.__name__ not in gen_fns
                )
            fn.output_tracked_idxs.clear()
            fn.output_param_ids.clear()
            fn.output_param_types.clear()
            fn.output_param_shapes.clear()
        elif (
            fn.__name__ in gen_fns
            and not self._include_generators
            and not self._array_caching
        ):
            raise Exception(
                "Generator function {} detected, but include_generators and array_caching "
                "are both False".format(fn.__name__)
            )
        else:
            # keep track of the parameter and the function it came from
            self._functions.append(fn)
            [
                self.add_param(id_, ptype, is_var, shape)
                for id_, ptype, is_var, shape in zip(
                    fn.output_param_ids,
                    fn.output_param_types,
                    fn.output_param_var_flags,
                    fn.output_param_shapes,
                )
            ]
        return

    def _chain_functions(self, terminal_ids: List):
        """We recurse back from the outputs to the inputs, keeping track of all relevant
        parameters and functions in the graph. We then try to figure out the most efficient
        order of execution for operations such that in a multiprocessing context, if we are
        at height h-1 we can start executing functions at height h asap.

        Parameters
        ----------
        terminal_ids
            ids of the final outputs
        """
        dict_key = _terminal_ids_to_key(terminal_ids)
        # add input params to param dict
        [
            self.add_param(id_, ptype, is_var, shape)
            for id_, ptype, is_var, shape in zip(
                self._arg_param_ids,
                self._arg_param_types,
                self._arg_param_var_flags,
                self._arg_param_shapes,
            )
        ]
        [
            self.add_param(id_, ptype, is_var, shape)
            for id_, ptype, is_var, shape in zip(
                self._kwarg_param_ids,
                self._kwarg_param_types,
                self._kwarg_param_var_flags,
                self._kwarg_param_shapes,
            )
        ]
        # add stateful params to param dict
        [
            self.add_param(id_, ptype, is_var, shape)
            for id_, ptype, is_var, shape in zip(
                self._stateful_param_ids,
                self._stateful_classes,
                self._stateful_param_var_flags,
                self._stateful_param_shapes,
            )
        ]

        # recursively chain the graph via backward traversal from the outputs
        [self.get_param_recursive(id_, 0) for id_ in terminal_ids]
        # stop if there are no functions in the graph
        if not self._functions:
            return

        # for storing function heights
        def store_fn_heights(fn: Callable) -> int:
            if hasattr(fn, "tree_height"):
                return fn.tree_height
            heights_in = [
                store_fn_heights(fn_in)
                for fn_in in fn.fns_in
                if fn_in in self._functions
            ]
            if heights_in:
                _height = max(heights_in) + 1
            else:
                _height = 0
            fn.tree_height = _height
            return _height

        # store function heights
        [store_fn_heights(self._id_to_function[id_]) for id_ in terminal_ids]
        # find the height of the tree
        max_tree_height = max([fn.tree_height for fn in self._functions])
        # group the functions based on their height in the tree from the starting leaf nodes
        grouped_functions = list()
        for height in range(0, max_tree_height + 1):
            fns = [fn for fn in self._functions if fn.tree_height == height]
            # for functions at height 0, we want to execute the ones with more `fns_out` first
            # i.e. the function with the most functions at the next height which depend on it
            # should be executed first (this is only useful in a multiprocessing context)
            if height == 0:
                fns = sorted(
                    fns, key=lambda x: -len(x.fns_out) if hasattr(x, "fns_out") else 0
                )
            # at other heights, we want the order to be such that we call functions earlier if
            # they depend on a function at the height below which is called earlier. This is so
            # in a multiprocessing context we can make a start on functions at the next height asap
            else:
                fns_hm1 = grouped_functions[-1]
                leftmost_idxs = [
                    max(
                        enumerate(
                            [
                                fn in fn_hm1.fns_out
                                for fn_hm1 in fns_hm1
                                if hasattr(fn_hm1, "fns_out")
                            ]
                        ),
                        key=lambda x: x[1],
                    )[0]
                    for fn in fns
                ]
                fns = [
                    fn for fn, _ in sorted(zip(fns, leftmost_idxs), key=lambda x: x[1])
                ]
            grouped_functions.append(fns)
        self._grouped_functions[dict_key] = grouped_functions

    def connect(self, output_connected_only: bool = True):
        """Connects functions together into the final graph.

        Example
        -------
        >>> import ivy
        >>> from graph_compiler.compiler import _create_graph
        >>> ivy.set_backend("torch")
        >>> x = ivy.array([1.])

        >>> def toy_function(x):
        ...    y = ivy.sum(x)
        ...    z = ivy.prod(x)
        ...    a = ivy.sin(y)
        ...    b = ivy.cos(z)
        ...    c = ivy.tan(z)
        ...    i = ivy.round(a)
        ...    j = ivy.floor(b)
        ...    k = ivy.ceil(c)
        ...    return i, j, k

        Let us call connect() and view the resulting order in which the
        functions will be executed:

        >>> graph = _create_graph(toy_function, x)
        >>> graph.connect()
        >>> print([fn.__name__ for fn in graph._functions])
        ['prod', 'sum', 'cos', 'tan', 'sin', 'floor', 'ceil', 'round']

        We can see that the order of functions isn't the same as in `toy_function`,
        as they are in the order determined in `_chain_functions` (optimised for a
        multiprocessing context).
        """
        # python's max recursion depth of 1000 may not be sufficient when using deeper networks
        sys.setrecursionlimit(max(2 * len(self._id_to_function), 1000))

        terminal_ids = self._output_param_ids + [
            id_
            for id_, fn in self._id_to_function.items()
            if (
                (fn.terminal and id_ in self._stateful_clone_id_dict)
                or (
                    fn.inplace_fn
                    and fn.from_tracked_var
                    and not fn.from_tracked_var_iterators
                )
                or fn.is_inplace_fw_fn
            )
        ]
        assert len(terminal_ids) != 0, "No parameters detected in the outputs."
        self._chain_functions(terminal_ids)
        assert self._all_grouped_functions, "tried to connect an empty function"
        if not output_connected_only:
            for fn in self._id_to_function.values():
                if fn.terminal:
                    self._chain_functions(fn.output_param_ids)
            self._all_connected = True
        self._outer_connected = True
        sys.setrecursionlimit(self._orig_recursion_limit)
        self._functions = [i for sl in self._all_grouped_functions for i in sl]

    # Compiled Function #
    # ----------------- #

    def __call__(self, *args, **kwargs):
        """Runs when calls to the compiled graph are made.

        Examples
        --------
        >>> import ivy, time
        >>> from graph_compiler.compiler import compile
        >>> ivy.set_backend("torch")
        >>> x = ivy.array([1., 2., 3.])

        >>> def fn(x):
        ...     y = ivy.sum(x)
        ...     z = ivy.multiply(x, y)
        ...     return z

        >>> comp_fn, graph = compile(fn, x, return_graph=True)

        The compiled function `_call` runs quicker than the original `fn`:

        >>> start = time.time()
        >>> normal_ret = fn(x)
        >>> print(time.time() - start)
        0.0005664825439453125

        >>> start = time.time()
        >>> compiled_ret = graph._call(x)
        >>> print(time.time() - start)
        0.00022029876708984375

        Of course they also give the same output:

        >>> print(normal_ret, compiled_ret)
        ivy.array([ 6., 12., 18.], dtype=float32) ivy.array([ 6., 12., 18.], dtype=float32)
        """
        if self._to_ivy:
            self.constants = nest_array_to_new_backend(
                self.constants, to_ignore=tvp._to_ignore, native=False
            )
            args = nest_array_to_new_backend(
                args, to_ignore=tvp._to_ignore, native=False
            )
            kwargs = nest_array_to_new_backend(
                kwargs, to_ignore=tvp._to_ignore, native=False
            )
        else:
            tracked_mutable = tvp.get_tracked_args(args, kwargs, self)
            args, kwargs = ivy.args_to_native(
                *args, cont_inplace=True, to_ignore=tvp._to_ignore, **kwargs
            )
            args, kwargs = tvp.restore_tracked_args(args, kwargs, tracked_mutable, self)
        return self._scripted_call(*args, **kwargs, **self.constants)

    def compiled(
        self, time_chronological: bool = True, frontend: str = None
    ) -> Callable:
        """Returns the scripted function. If time_chronological is `True`, the order
        in which operations will be executed will be the same as the order of
        execution during the inital forward pass. If `False`, the order will be that
        which we constructed in `_chain_functions`.
        """
        # only connect graph if we haven't already
        if not self._outer_connected:
            self.connect()
        if time_chronological:
            self._functions = sorted(self._functions, key=lambda fn: fn.timestamp)
        self.reload_sourcecode(frontend=frontend)
        # return the handle of the call function
        return self.__call__

    def reload_sourcecode(self, frontend=None):
        # generate source code
        source_generator = sg.SourceGenerator(self)
        fn_str = source_generator.generate_source(graph=self, frontend=frontend)
        constants = source_generator.get_constants()
        # define function
        if os.getenv("IVY_DEBUG_SOURCE", "False").lower() == "true":
            compiled_fn = sg.load_fn_from_file(fn_str)
        else:
            compiled_fn = sg.load_fn_from_str(fn_str)
        self._scripted_call = compiled_fn
        self.__fn_str = fn_str
        self.constants = constants

    def obtain_sourcecode(self):
        return self.__fn_str, self.constants

    def initialize_from_cache(self, compiled_fn, constants):
        assert compiled_fn is not None, "compiled_fn must be specified."
        assert constants is not None, "constants must be specified."
        self._scripted_call = compiled_fn
        self.constants = constants

    # Helpers #
    # ------- #

    def list_function_frequencies(
        self,
        return_raw: bool = False,
        return_str: bool = False,
        save_json: str = None,
    ) -> None:
        """Logs a list of the functions used in the graph.

        Parameters
        ----------
        return_raw : bool, optional
            if set to True, the list of function objects will be returned,
            default is False
        return_str : bool, optional
            if set to True, the message will be returned as a string instead of printed,
            default is False
        save_json : str, optional
            if specified, path of the JSON file where the used functions
            will be logged, default is None
        """
        backend_fns = [f.backend_fn for f in self._functions]
        if return_raw:
            return backend_fns
        paths = [sg.FUNC_TO_PATH[fn] for fn in backend_fns if fn in sg.FUNC_TO_PATH]
        frequency = collections.Counter(paths).most_common()
        msg = "The functions being used are <(number of calls) function_path> : \n-> {}".format(
            "\n-> ".join(
                [" (" + str(freq[1]) + ") \t" + str(freq[0]) for freq in frequency]
            )
        )
        if save_json:
            with open(save_json, "w") as fp:
                data = {freq[0]: {"count": freq[1]} for freq in frequency}
                json.dump(data, fp, indent=4)
        if return_str:
            return msg
        else:
            print(msg)

    def list_missing_frontends(
        self,
        save_json: str = None,
    ) -> None:
        """Logs a list of the functions used in the graph that are currently missing
        a corresponding frontend function.

        Parameters
        ----------
        save_json : str, optional
            if specified, path of the JSON file where the missing functions
            will be logged, default is None
        fw : str, optional
            if specified, this framework will be used to look for missing
            frontends, default is None
        """
        frequency = _find_missing_frontends(self)
        msg = _format_missing_frontends_msg(frequency)
        if save_json:
            with open(save_json, "w") as fp:
                data = {freq[0]: {"count": freq[1]} for freq in frequency}
                json.dump(data, fp, indent=4)
        else:
            print(msg)

    # Graph Visualization #
    # --------------------#

    def _is_stateful(self, f):
        if hasattr(f, "args"):
            for a in f.arg_param_types:
                if a in self._stateful_classes:
                    return True
        if hasattr(f, "kwargs"):
            for kwa in f.kwarg_param_types:
                if kwa in self._stateful_classes:
                    return True
        return False

    def _add_edge(
        self,
        g,
        func,
        id_in,
        idx,
        inp,
        num_inputs,
        with_edge_labels,
        with_arg_labels,
        with_output_labels,
    ):
        start_color = self._io_node_color
        start_title = ""
        if id_in in self._id_to_function:
            fn_in = self._id_to_function[id_in]
            fn_id = fn_in.output_param_ids[0]
            start_color = self._inter_node_color
            start_title = f"{_args_str_from_fn(fn_in)}\n" if with_arg_labels else ""
            start_title = (
                start_title + _output_str_from_fn(fn_in)
                if with_output_labels
                else start_title
            )
        elif id_in in self._input_functions:
            fn_in = self._input_functions[id_in]
            fn_id = id_in
        else:
            fn_in = _copy_func(inp)
            idx0 = idx[0]
            sig = list(self._fn_signature.keys())
            if isinstance(idx0, str):
                arg_name = idx0
            elif "args" not in sig and isinstance(idx0, int) and idx0 < len(sig):
                arg_name = sig[idx0]
            else:
                arg_name = str(idx0)
            fnc_name = "input: " + arg_name
            idx1on = idx[1:]
            if idx1on:
                fnc_name += ", {}".format(idx1on)
            fn_in.__name__ = fnc_name
            fn_id = id_in
            self._input_functions[id_in] = fn_in
            num_inputs += 1
        # change color if is var
        start_color = (
            self._var_node_color
            if fn_id in self._id_to_parameter and self._id_to_parameter[fn_id].is_var
            else start_color
        )
        # add start node
        g.add_node(
            fn_id,
            label=fn_in.__name__,
            size=self._node_size,
            color=start_color,
        )
        if start_title != "":
            g.nodes[fn_id]["title"] = start_title
        # add end node
        end_title = f"{_args_str_from_fn(func)}\n" if with_arg_labels else ""
        end_title = (
            end_title + _output_str_from_fn(func) if with_output_labels else end_title
        )
        # change color if is var
        end_color = (
            self._var_node_color
            if func.output_param_ids[0] in self._id_to_parameter
            and self._id_to_parameter[func.output_param_ids[0]].is_var
            else self._inter_node_color
        )
        g.add_node(
            func.output_param_ids[0],
            label=func.__name__,
            size=self._node_size,
            color=end_color,
        )
        if end_title != "":
            g.nodes[func.output_param_ids[0]]["title"] = end_title
        edge_label = (
            _param_to_label(self._id_to_parameter[id_in]) if with_edge_labels else ""
        )
        g.add_edge(
            fn_id,
            func.output_param_ids[0],
            label=edge_label,
            arrowStrikethrough=False,
        )
        return num_inputs

    def _position_nodes(self, g, num_inputs, num_outputs, all_nodes, randomness_factor):
        pos_dict = dict()
        assert 0 <= randomness_factor <= 1

        # select position based on width and height of graph
        for height, nodes in enumerate(all_nodes):
            width = len(nodes)
            for w, n in enumerate(nodes):
                pos = np.array(
                    [
                        (height + 1) / (self._max_graph_height + 1),
                        0.5 if width == 1 else w / (width - 1),
                    ]
                )
                assert np.logical_and((0 <= pos), (pos <= 1)).all()
                h_delta = 0.5 / self._max_graph_height
                h_rand = np.random.uniform(-h_delta, h_delta)
                w_delta = 0.5 if width == 1 else 0.5 / (width - 1)
                w_delta_low = 0 if (w == 0 and width != 1) else -w_delta
                w_delta_high = 0 if (w == (width - 1) and width != 1) else w_delta
                w_rand = np.random.uniform(w_delta_low, w_delta_high)
                pos += np.array([h_rand, w_rand]) * randomness_factor
                assert np.logical_and((0 <= pos), (pos <= 1)).all()
                pos_dict[n[0]] = pos

        # add inputs
        if num_inputs > 0:
            input_idx = 0
            input_nodes = [
                n
                for n in g.nodes
                if n not in pos_dict and g.nodes[n]["label"][:5] == "input"
            ]
            min_output_y_coords = [
                min([pos_dict[e[1]][1] for e in g.edges if n in e]) for n in input_nodes
            ]
            input_nodes = [n for _, n in sorted(zip(min_output_y_coords, input_nodes))]
            for n in input_nodes:
                pos = np.array(
                    [0.0, 0.5 if num_inputs == 1 else input_idx / (num_inputs - 1)]
                )
                assert np.logical_and((0 <= pos), (pos <= 1)).all()
                h_delta = 0.5 / self._max_graph_height
                h_rand = np.random.uniform(0, h_delta)
                w_delta = 0.5 if num_inputs == 1 else 0.5 / (num_inputs - 1)
                w_delta_low = 0 if input_idx == 0 else -w_delta
                w_delta_high = 0 if input_idx == (num_inputs - 1) else w_delta
                w_rand = np.random.uniform(w_delta_low, w_delta_high)
                pos += np.array([h_rand, w_rand]) * randomness_factor
                assert np.logical_and((0 <= pos), (pos <= 1)).all()
                pos_dict[n] = pos
                input_idx += 1

        # add outputs
        if num_outputs > 0:
            output_idx = 0
            output_nodes = [
                n
                for n in g.nodes
                if n not in pos_dict and g.nodes[n]["label"][:6] == "output"
            ]
            min_input_y_coords = [
                min([pos_dict[e[0]][1] for e in g.edges if n in e])
                for n in output_nodes
            ]
            output_nodes = [n for _, n in sorted(zip(min_input_y_coords, output_nodes))]
            for n in output_nodes:
                pos = np.array(
                    [1.0, 0.5 if num_outputs == 1 else output_idx / (num_outputs - 1)]
                )
                assert np.logical_and((0 <= pos), (pos <= 1)).all()
                h_delta = 0.5 / self._max_graph_height
                h_rand = np.random.uniform(-h_delta, 0)
                w_delta = 0.5 if num_outputs == 1 else 0.5 / (num_outputs - 1)
                w_delta_low = 0 if output_idx == 0 else -w_delta
                w_delta_high = 0 if output_idx == (num_outputs - 1) else w_delta
                w_rand = np.random.uniform(w_delta_low, w_delta_high)
                pos += np.array([h_rand, w_rand]) * randomness_factor
                assert np.logical_and((0 <= pos), (pos <= 1)).all()
                pos_dict[n] = pos
                output_idx += 1

        return pos_dict

    def _populate_graph(
        self,
        g,
        functions,
        with_edge_labels,
        with_arg_labels,
        with_output_labels,
        output_connected_only,
        randomness_factor,
        pos=None,
    ):
        # config
        node_sep_x = (
            5 * self._node_size if not with_edge_labels else 10 * self._node_size
        )
        node_sep_y = 4 * self._node_size

        num_inputs = 0
        num_outputs = 0

        # add intermediate nodes
        def inp():
            pass

        for func in self._id_to_function.values():
            if func not in functions and output_connected_only:
                continue
            for id_in, idx in zip(func.arg_param_ids, func.arg_tracked_idxs):
                num_inputs = self._add_edge(
                    g,
                    func,
                    id_in,
                    idx,
                    inp,
                    num_inputs,
                    with_edge_labels,
                    with_arg_labels,
                    with_output_labels,
                )
            for id_in, idx in zip(func.kwarg_param_ids, func.kwarg_tracked_idxs):
                num_inputs = self._add_edge(
                    g,
                    func,
                    id_in,
                    idx,
                    inp,
                    num_inputs,
                    with_edge_labels,
                    with_arg_labels,
                    with_output_labels,
                )

        # add output nodes and edges
        for id_ in self._output_param_ids:
            # change color if is var
            color = (
                self._var_node_color
                if id_ in self._id_to_parameter and self._id_to_parameter[id_].is_var
                else self._io_node_color
            )
            g.add_node(
                f"{id_}_output",
                label=f"output\n" + _output_str_from_fn(self._id_to_function[id_]),
                size=self._node_size,
                color=color,
            )
            edge_label = (
                _param_to_label(self._id_to_parameter[id_]) if with_edge_labels else ""
            )
            g.add_edge(
                id_,
                f"{id_}_output",
                label=edge_label,
            )
            num_outputs += 1

        # calculate node positions
        all_nodes = list()
        max_graph_width = 0
        for fns in self._all_grouped_functions:
            nodes = [
                (f.output_param_ids[0], f, _args_str_from_fn(f), _output_str_from_fn(f))
                for f in fns
            ]
            seen = set()
            nodes = [n for n in nodes if not (n in seen or seen.add(n))]
            max_graph_width = max(max_graph_width, len(nodes))
            all_nodes.append(nodes)
        pos = ivy.default(
            pos,
            self._position_nodes(
                g, num_inputs, num_outputs, all_nodes, randomness_factor
            ),
        )
        # scale pos
        _x = lambda x: int(x * node_sep_x * (len(g.nodes)))
        _y = lambda y: int(y * node_sep_y * max_graph_width)
        pos = {n: [_x(p[0]), _y(p[1])] for n, p in pos.items() if n in g.nodes}

        # assert all positions are accounted for, if provided
        if ivy.exists(pos):
            assert min([n in pos for n in g.nodes])

        # Add position to nodes
        for id_ in g.nodes():
            g.nodes[id_]["x"] = pos[id_][0]
            g.nodes[id_]["y"] = pos[id_][1]

        # change color for stateful nodes
        stateful_nodes = [
            n
            for n in g.nodes
            if n in self._id_to_function and self._is_stateful(self._id_to_function[n])
        ]
        for sn in stateful_nodes:
            g.nodes[sn]["color"] = self._stateful_node_color

        # draw
        # ToDo: plt.draw_if_interactive() # check if it works in a notebook

        return

    def show(
        self,
        save_to_disk=False,
        notebook=False,
        with_edge_labels=True,
        with_arg_labels=True,
        with_output_labels=True,
        output_connected_only=True,
        randomness_factor=0.1,
        highlight_subgraph=None,
        fname=None,
    ):
        # ToDo: deal with highlight subgraph behaviour
        # ToDo: remove input_to_output_link from the graph
        # ToDo: add support to icons in small graphs maybe?
        # ToDo: add color based on the backend

        # assert that required visualization packages are installed
        if not ivy.exists(nx):
            raise Exception(
                "networkx python package must be installed in order to visualize computation graphs."
            )
        if not ivy.exists(Network):
            raise Exception(
                "pyvis python package must be installed in order to visualize computation graphs."
            )

        # restart helper dicts
        self._input_functions = dict()
        self._output_functions = dict()

        # ensure graph is connected
        if not self._outer_connected or (
            not output_connected_only and not self._all_connected
        ):
            self.connect(output_connected_only)

        # create directed networkX graph
        g = nx.DiGraph()
        subgraph_g = list()

        # create a pyvis Network
        notebook_kwargs = {}
        if notebook:
            notebook_kwargs = {
                "height": "300px",
                "notebook": True,
                "cdn_resources": "in_line",
            }
        nt = Network(directed=True, **notebook_kwargs)

        # build the graph
        all_functions = self._functions
        self._populate_graph(
            g,
            all_functions,
            with_edge_labels,
            with_arg_labels,
            with_output_labels,
            output_connected_only,
            randomness_factor,
        )

        # add nodes and edges from the main graph
        for node, node_attrs in g.nodes(data=True):
            nt.add_node(node, **node_attrs)

        for source, dest, edge_attrs in g.edges(data=True):
            nt.add_edge(source, dest, **edge_attrs)

        # add nodes and edges from the subgraph
        if self._sub_graphs:
            from graph_compiler.special_ops.vmap_helpers import (
                _handle_vmap_nodes,
                _remove_input_subgraph_nodes,
                _get_all_graph_functions,
            )

            # reorder the subgraphs
            all_functions = _get_all_graph_functions(self)
            ordered_subgraphs = [
                self._sub_graphs[id(fn)]
                for fn in all_functions
                if fn.__name__ == "vmap"
            ]

            for subgraph in ordered_subgraphs:
                subg = nx.DiGraph()
                subgraph_g.append(subg)

                subgraph._populate_graph(
                    subg,
                    subgraph._functions,
                    with_edge_labels,
                    with_arg_labels,
                    with_output_labels,
                    output_connected_only,
                    randomness_factor,
                )
                subgraph_nodes = _remove_input_subgraph_nodes(subg.nodes(data=True))
                for node, node_attrs in subgraph_nodes:
                    if not "output" in node_attrs["label"]:
                        node_attrs["color"] = "#b6f1f1"
                        node_attrs["borderWidth"] = 2
                        node_attrs["borderWidthSelected"] = 4

                    nt.add_node(node, **node_attrs)

                subgraph_edges = [
                    (source, dest, attrs)
                    for source, dest, attrs in subg.edges(data=True)
                    if not any(
                        "input" in subg.nodes[node]["label"] for node in (source, dest)
                    )
                ]
                for source, dest, edge_attrs in subgraph_edges:
                    nt.add_edge(source, dest, **edge_attrs)

            vmap_tuples = _handle_vmap_nodes(g, subgraph_g, all_functions)

            for vmap_node in vmap_tuples:
                nt.add_edge(
                    vmap_node[1],
                    vmap_node[0],
                    dashes=True,
                )
                nt.add_edge(
                    vmap_node[2],
                    vmap_node[1],
                    dashes=True,
                )

        # maybe highlight sub-graph (ToDo) self._functions?
        # if isinstance(highlight_subgraph, int):
        #     # set node color as red
        #     self._inter_node_color = (0.8, 0.0, 0.0)
        #     self._stateful_node_color = (0.8, 0.0, 0.0)
        #     self._io_node_color = (0.8, 0.0, 0.0)
        #     self._edge_color = (0.4, 0.0, 0.0)
        #
        #     # show highlighted sub-graph
        #     subgraph_id = list(self._functions.keys())[highlight_subgraph]
        #     self._show_for_functions(
        #         ax,
        #         self._functions[subgraph_id],
        #         with_edge_labels,
        #         with_arg_labels,
        #         with_output_labels,
        #         True,
        #         randomness_factor,
        #         False,
        #         cv2_labels,
        #         pos=pos,
        #     )

        # create a pyvis Network
        notebook_kwargs = {}
        if notebook:
            notebook_kwargs = {
                "height": "300px",
                "notebook": True,
                "cdn_resources": "in_line",
            }
        nt = Network(directed=True, **notebook_kwargs)
        # populates the nodes and edges data structures from the networkx graph
        nt.from_nx(g)
        nt.set_options(
            """
        const options = { "edges" : { "color": { "inherit": "both" }, 
                                      "smooth": false},
                          "physics": {"enabled": false}}
            """
        )

        # maybe save to disk -> nt.shows saves the file by default,
        # so to visualize the graph, save_to_disk must be set. This should
        # maybe be revisited
        if save_to_disk or fname:
            fname = ivy.default(
                fname,
                "graph_{}.html".format(
                    "".join(
                        [f.__name__.replace("_", "")[0] for f in self._functions][0:20]
                    )
                ),
            )
            if fname[-5:] != ".html":
                if "." in fname:
                    fname = ".".join(fname.split(".")[:-1])
                fname += ".html"
            if notebook:
                return nt.show(fname)
            nt.save_graph(fname)


class LazyGraph:
    def __init__(self, obj, initializer, *args, **kwargs):
        self._eager_graph = None
        self._initial_obj = obj
        self._initial_args = args
        self._initializer = initializer
        self._initial_kwargs = kwargs
        self._initialized = False

        if isinstance(obj, FunctionType):
            self.__name__ = obj.__name__
        elif isinstance(obj, object):
            self.__name__ = type(obj).__name__

    def _initialize(self, *args, **kwargs):
        if not self._initialized:
            if "source" in self._initial_kwargs:
                ivy.set_backend(self._initial_kwargs["source"])
            self._initial_args = nest_array_to_new_backend(
                self._initial_args, to_ignore=tvp._to_ignore
            )
            self._initial_kwargs = nest_array_to_new_backend(
                self._initial_kwargs, to_ignore=tvp._to_ignore
            )
            self._eager_graph = self._initializer(
                self._initial_obj,
                *self._initial_args,
                args=args,
                kwargs=kwargs,
                **self._initial_kwargs,
            )
            self._initialized = True
            if "source" in self._initial_kwargs:
                ivy.previous_backend()

    def _check_if_initialized(self):
        if not self._initialized:
            raise ValueError(
                "A LazyGraph instance must be initialized before calling a Graph method."
            )

    def __call__(self, *args, **kwargs):
        if not self._initialized:
            self._initialize(*args, **kwargs)
            to = self._initial_kwargs["to"]
            if to not in [None, "ivy"]:
                ivy.set_backend(to)
            args = nest_array_to_new_backend(args, to_ignore=tvp._to_ignore)
            kwargs = nest_array_to_new_backend(kwargs, to_ignore=tvp._to_ignore)
            if to not in [None, "ivy"]:
                ivy.previous_backend()
        return self._eager_graph(*args, **kwargs)

    def __repr__(self):
        return f"{object.__repr__(self)} ({'not ' if not self._initialized else ''}initialized)"

    def list_missing_frontends(self, *args, **kwargs):
        self._check_if_initialized()
        return self._eager_graph.list_missing_frontends(*args, **kwargs)

    list_missing_frontends.__doc__ = Graph.list_missing_frontends.__doc__

    def list_function_frequencies(self, *args, **kwargs):
        self._check_if_initialized()
        return self._eager_graph.list_function_frequencies(*args, **kwargs)

    list_function_frequencies.__doc__ = Graph.list_function_frequencies.__doc__

    def show(self, *args, **kwargs):
        self._check_if_initialized()
        return self._eager_graph.show(*args, **kwargs)

    show.__doc__ = Graph.show.__doc__
