import importlib
import builtins
from typing import Callable, List

import ivy
import graph_compiler.tracked_var_proxy as tvp
from graph_compiler.graph import Graph
from graph_compiler.param import Param
from graph_compiler.wrapping import FUNC_TO_PATH, Node


def _get_backend_fn_from_path(fn_path: List[str]):
    split_path = fn_path.split(".")
    if hasattr(tvp, split_path[0]):
        backend_fn = getattr(tvp, split_path[0])
    elif hasattr(builtins, split_path[0]):
        backend_fn = getattr(builtins, split_path[0])
    elif fn_path == "vmap":
        backend_fn = Node()
        backend_fn.__name__ = "vmap"
    else:
        backend_fn = importlib.import_module(split_path[0])
    for p in split_path[1:]:
        backend_fn = getattr(backend_fn, p)
    return backend_fn


def _convert_graph_to_dict(graph: Graph, is_subgraph: bool = False):
    """
    Serializes a Graph object to a simple dict.

    NOTE:
    1. currently memorisation of array objects is not supported.
    2. stateful objects are not supported

    Parameters
    ----------
    graph

    Returns
    -------
    ret
        a dict containing all the essential details to construct/transpile a graph
    """
    graph_dict = dict()

    graph_dict["_to_ivy"] = graph._to_ivy

    # graph_dict["_args"] = graph._args
    graph_dict["_arg_tracked_idxs"] = graph._arg_tracked_idxs
    graph_dict["_arg_param_ids"] = graph._arg_param_ids

    # graph_dict["_kwargs"] = graph._kwargs
    graph_dict["_kwarg_tracked_idxs"] = graph._kwarg_tracked_idxs
    graph_dict["_kwarg_param_ids"] = graph._kwarg_param_ids

    graph_dict["_output_tracked_idxs"] = graph._output_tracked_idxs
    graph_dict["_output_param_ids"] = graph._output_param_ids

    # add parameters as well
    graph_dict["parameters"] = [id_ for id_ in graph._id_to_parameter.keys()]

    # set tracked outputs to be None
    graph_dict["_output"] = ivy.map_nest_at_indices(
        graph._output, graph._output_tracked_idxs, lambda x: None
    )
    if not is_subgraph:
        graph_dict["constants"] = graph.constants
    # if graph._sub_graphs:
    graph_dict["_sub_graphs"] = {
        k: _convert_graph_to_dict(v, True) for k, v in graph._sub_graphs.items()
    }

    function_list = list()
    for wrapped_fn in graph._functions:
        func_dict = dict()
        try:
            func_dict["backend_fn"] = FUNC_TO_PATH[wrapped_fn.backend_fn]
        except KeyError:
            if wrapped_fn.__name__ == "vmap":
                func_dict["backend_fn"] = "vmap"
                func_dict["id_"] = wrapped_fn.id_
                func_dict["vmap_args"] = wrapped_fn.vmap_args
                func_dict["vmap_kwargs"] = wrapped_fn.vmap_kwargs
            else:
                func_dict["backend_fn"] = wrapped_fn.backend_fn.__qualname__

        func_dict["args"] = wrapped_fn.args
        func_dict["arg_tracked_idxs"] = wrapped_fn.arg_tracked_idxs
        func_dict["arg_param_ids"] = wrapped_fn.arg_param_ids

        func_dict["kwargs"] = wrapped_fn.kwargs
        func_dict["kwarg_tracked_idxs"] = wrapped_fn.kwarg_tracked_idxs
        func_dict["kwarg_param_ids"] = wrapped_fn.kwarg_param_ids

        func_dict["output"] = wrapped_fn.output
        func_dict["output_tracked_idxs"] = wrapped_fn.output_tracked_idxs
        func_dict["output_param_ids"] = wrapped_fn.output_param_ids

        func_dict["remove_output_tuple"] = wrapped_fn.remove_output_tuple
        func_dict["with_tracked_slices"] = wrapped_fn.with_tracked_slices
        func_dict["from_tracked_var"] = wrapped_fn.from_tracked_var
        func_dict["inplace_fn"] = wrapped_fn.inplace_fn
        func_dict["timestamp"] = wrapped_fn.timestamp

        function_list.append(func_dict)

    graph_dict["_all_functions"] = function_list

    try:
        # return json.dumps(graph_dict, skipkeys=True)
        return graph_dict
    except TypeError:
        raise ivy.exceptions.IvyException(
            "Cannot convert the given graph to dict, the graph might contain non-dependent variables or stateful classes in its creation..."
        )


def wrap_backend_fn(backend_fn: Callable, func_dict: dict):
    wrapped_fn = Node()

    wrapped_fn.args = func_dict["args"]
    wrapped_fn.kwargs = func_dict["kwargs"]

    wrapped_fn.arg_param_ids = func_dict["arg_param_ids"]
    wrapped_fn.kwarg_param_ids = func_dict["kwarg_param_ids"]

    wrapped_fn.arg_tracked_idxs = func_dict["arg_tracked_idxs"]
    wrapped_fn.kwarg_tracked_idxs = func_dict["kwarg_tracked_idxs"]

    wrapped_fn.output = func_dict["output"]
    wrapped_fn.output_param_ids = func_dict["output_param_ids"]
    wrapped_fn.output_tracked_idxs = func_dict["output_tracked_idxs"]

    wrapped_fn.remove_output_tuple = func_dict["remove_output_tuple"]
    wrapped_fn.with_tracked_slices = func_dict["with_tracked_slices"]
    wrapped_fn.from_tracked_var = func_dict["from_tracked_var"]
    wrapped_fn.timestamp = func_dict["timestamp"]
    wrapped_fn.inplace_fn = func_dict["inplace_fn"]

    wrapped_fn.backend_fn = backend_fn
    wrapped_fn.__name__ = backend_fn.__name__

    if wrapped_fn.__name__ == "vmap":
        wrapped_fn.id_ = func_dict["id_"]
        wrapped_fn.vmap_args = func_dict["vmap_args"]
        wrapped_fn.vmap_kwargs = func_dict["vmap_kwargs"]

    return wrapped_fn


# TODO: support Stateful and cached arrays
def _convert_dict_to_graph(graph_dict: dict, is_subgraph_dict: bool = False):
    # create an empty graph object
    graph = Graph.empty()

    graph._outer_connected = True
    graph._to_ivy = graph_dict["_to_ivy"]
    if graph_dict["_sub_graphs"]:
        graph._sub_graphs = {
            k: _convert_dict_to_graph(v, True)
            for k, v in graph_dict["_sub_graphs"].items()
        }

    # TODO: infer ptype as well
    for id_ in graph_dict["parameters"]:
        graph._id_to_parameter[id_] = Param(ptype=None)

    graph._arg_tracked_idxs = graph_dict["_arg_tracked_idxs"]
    graph._arg_param_ids = graph_dict["_arg_param_ids"]

    graph._kwarg_tracked_idxs = graph_dict["_kwarg_tracked_idxs"]
    graph._kwarg_param_ids = graph_dict["_kwarg_param_ids"]

    graph._output_tracked_idxs = graph_dict["_output_tracked_idxs"]
    graph._output_param_ids = graph_dict["_output_param_ids"]

    # wrap functions again in graph_dict["_all_functions"]
    wrapped_fn_list = list()

    for func_dict in graph_dict["_all_functions"]:
        backend_fn_path = func_dict["backend_fn"]
        backend_fn = _get_backend_fn_from_path(backend_fn_path)

        # wrap the backend function so that it works with graph _call method
        wrapped_fn = wrap_backend_fn(backend_fn, func_dict)
        wrapped_fn_list.append(wrapped_fn)

    graph._functions = wrapped_fn_list

    # TODO: we can just replace with placeholders
    graph._output = graph_dict["_output"]

    # ToDo: This won't get correctly serialzed (same as before)
    if not is_subgraph_dict:
        graph.constants = graph_dict["constants"]
    graph.compiled()
    return graph
