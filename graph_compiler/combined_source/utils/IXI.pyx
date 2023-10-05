import importlib
from typing import Callable, List

import ivy
from IIV import Graph
from IVI import Param
import VIV as sg


def _get_backend_fn_from_path(fn_path: List[str]):
    backend_fn = importlib.import_module(fn_path[0])
    for p in fn_path[1:]:
        backend_fn = getattr(backend_fn, p)
    return backend_fn


def _convert_graph_to_dict(graph: Graph):
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

    graph_dict["_scripted_call"] = graph._scripted_call
    graph_dict["constants"] = graph.constants

    function_list = list()
    not_implemented_frontend_fns = list()
    fw = graph.backend

    for wrapped_fn in graph._functions:
        func_dict = dict()
        try:
            func_dict["backend_fn"] = sg.FUNC_TO_PATH[fw]["native"][
                wrapped_fn.backend_fn
            ]
        except KeyError:
            not_implemented_frontend_fns.append(wrapped_fn.backend_fn)

        func_dict["args"] = wrapped_fn.args
        func_dict["arg_tracked_idxs"] = wrapped_fn.arg_tracked_idxs
        func_dict["arg_param_ids"] = wrapped_fn.arg_param_ids

        func_dict["kwargs"] = wrapped_fn.kwargs
        func_dict["kwarg_tracked_idxs"] = wrapped_fn.kwarg_tracked_idxs
        func_dict["kwarg_param_ids"] = wrapped_fn.kwarg_param_ids

        func_dict["output_tracked_idxs"] = wrapped_fn.output_tracked_idxs
        func_dict["output_param_ids"] = wrapped_fn.output_param_ids

        func_dict["timestamp"] = wrapped_fn.timestamp

        function_list.append(func_dict)

    if not_implemented_frontend_fns:
        missing_frontends = graph.list_missing_frontends(
            return_str=True, fw=graph.backend
        )
        raise ivy.exceptions.IvyException(missing_frontends)

    graph_dict["_all_functions"] = function_list

    try:
        # return json.dumps(graph_dict, skipkeys=True)
        return graph_dict
    except TypeError:
        raise ivy.exceptions.IvyException(
            "Cannot convert the given graph to dict, the graph might contain non-dependent variables or stateful classes in its creation..."
        )


def wrap_backend_fn(backend_fn: Callable, func_dict: dict):
    def wrapped_fn(args, kwargs):
        args_writeable = ivy.copy_nest(func_dict["args"])
        kwargs_writeable = ivy.copy_nest(func_dict["kwargs"])

        args_writeable = ivy.set_nest_at_indices(
            args_writeable, func_dict["arg_tracked_idxs"], args
        )
        kwargs_writeable = ivy.set_nest_at_indices(
            kwargs_writeable, func_dict["kwarg_tracked_idxs"], kwargs
        )

        return backend_fn(*args_writeable, **kwargs_writeable)

    wrapped_fn.args = func_dict["args"]
    wrapped_fn.kwargs = func_dict["kwargs"]

    wrapped_fn.arg_param_ids = func_dict["arg_param_ids"]
    wrapped_fn.kwarg_param_ids = func_dict["kwarg_param_ids"]

    wrapped_fn.arg_tracked_idxs = func_dict["arg_tracked_idxs"]
    wrapped_fn.kwarg_tracked_idxs = func_dict["kwarg_tracked_idxs"]

    wrapped_fn.output_param_ids = func_dict["output_param_ids"]
    wrapped_fn.output_tracked_idxs = func_dict["output_tracked_idxs"]

    wrapped_fn.timestamp = func_dict["timestamp"]

    wrapped_fn.backend_fn = backend_fn
    wrapped_fn.__name__ = backend_fn.__name__

    return wrapped_fn


# TODO: support Stateful and cached arrays
def _convert_dict_to_graph(graph_dict: dict):
    # create an empty graph object
    graph = Graph.empty()

    graph._outer_connected = True

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

    graph._scripted_call = graph_dict["_scripted_call"]
    # ToDo: This won't get correctly serialzed (same as before)
    graph.constants = graph_dict["constants"]

    return graph
