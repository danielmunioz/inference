# global
import sys
import importlib
import importlib.util
from typing import Callable, List
import textwrap

# local
import ivy
from graph_compiler import helpers
import graph_compiler.globals as glob
from graph_compiler.wrapping import FUNC_TO_PATH
from graph_compiler.special_ops.vmap_helpers import generate_vmap_subgraph


class SourceGenerator:
    def __init__(self, graph):
        self.graph = graph
        # Constants cached during compilation, these are passed to the compiled call.
        self.constants = {}
        # Count of references to intermediate variables
        self.reference_count = {}
        self.count_references(graph)
        self.vmap_imported = False

    def count_references(self, graph=None):
        for f in graph._functions:
            for a in f.arg_param_ids:
                self.reference_count[a] = self.reference_count.get(a, 0) + 1
            for kw in f.kwarg_param_ids:
                self.reference_count[kw] = self.reference_count.get(kw, 0) + 1
        for o in graph._output_param_ids:
            self.reference_count[o] = self.reference_count.get(o, 0) + 1

    def get_constants(self) -> dict:
        return self.constants

    def generate_vars_from(self, from_: str, graph=None) -> List[str]:
        """Creates references to the tracked params to extract their values from the args and kwargs
        of the original function. A list of lines will be returned, where each line will have the format:

        "p{tracked_param_id} = {args|kwargs}[i1][i2]["keyword"][...]"
        """
        if from_ not in ["args", "kwargs"]:
            raise ValueError("from_ must be one of 'args' or 'kwargs'.")
        extracted_params = []
        param_ids = graph._arg_param_ids if from_ == "args" else graph._kwarg_param_ids
        tracked_idxs = (
            graph._arg_tracked_idxs if from_ == "args" else graph._kwarg_tracked_idxs
        )
        key_from = lambda x: "'" + x + "'"
        for tracked_id, idx in zip(param_ids, tracked_idxs):
            indices_str = f"[{']['.join(key_from(i) if isinstance(i, str) else str(i) for i in idx)}]"
            extracted_var = f"p{tracked_id} = {from_}{indices_str}\n    "
            extracted_params.append(extracted_var)
        return extracted_params

    def generate_fn_output(self, graph=None) -> str:
        """Generates the final line of the compiled function. This will have the format:

        "return p1234, p4321, ..."
        """
        output_reprs = graph._output
        ids = graph._output_param_ids
        idxs = graph._output_tracked_idxs

        # replace tracked parameters with their ids
        output_reprs = ivy.set_nest_at_indices(output_reprs, idxs, ids, shallow=False)
        # Replace non tracked vars with a variable that stores their value
        output_reprs = self.replace_nontracked_from_reprs(output_reprs, ids, idxs)
        glob.logging_paused = True
        # Replace tracked args with the corresponding intermediate variable
        output_reprs = ivy.map_nest_at_indices(
            output_reprs, idxs, lambda x: f"p{x}", shallow=False
        )

        # store subclasses as context variables
        output_reprs_list = [output_reprs]
        subclass_idx = ivy.nested_argwhere(
            output_reprs_list,
            lambda x: isinstance(x, (list, tuple, dict))
            and type(x) not in [list, tuple, dict],
            check_nests=True,
        )
        subclass_values = ivy.multi_index_nest(output_reprs_list, subclass_idx)
        subclass_names = [f"init_{type(x).__name__}" for x in subclass_values]
        subclass_values = [type(x) for x in subclass_values]
        # add variables to function context
        self.constants.update(dict(zip(subclass_names, subclass_values)))
        # replace args with the new variables
        output_reprs = nest_to_str(output_reprs, subclass_names, subclass_values)
        return f"return {output_reprs}"

    def generate_source(
        self,
        frontend: str = None,
        graph=None,
        fn_name: str = "compiled_fn",
        indent: int = 0,
    ) -> str:
        """Generates a string containing the source file for the compiled call."""
        # source file header
        fn_header = ""
        if indent == 0:
            if frontend:
                # ToDo: Only update the header if we are transpiling
                fn_header += (
                    f"import ivy.functional.frontends.{frontend} as {frontend}\n"
                )
            elif not graph._to_ivy:
                fn_header += f"import {ivy.current_backend_str()}\n"
                if ivy.current_backend_str() == "jax":
                    fn_header += f"import jaxlib\n"
            else:
                fn_header += f"import ivy\n"
            if graph._with_numpy:
                if frontend:
                    fn_header += f"import ivy.functional.frontends.numpy as numpy\n"
                else:
                    fn_header += f"import numpy\n"
            if glob.check_frontend_vs_original_differences:
                fn_header += f"import graph_compiler.globals as glob\n"
        # function signature

        fn_signature = f"\ndef {fn_name}(*args, **kwargs):\n"
        # function output
        fn_output = ""
        graph_output_str = self.generate_fn_output(graph)
        fn_output += graph_output_str
        # function variables
        fn_variables = ""
        # graph args and kwargs
        tracked_in_args = self.generate_vars_from("args", graph)
        tracked_in_kwargs = self.generate_vars_from("kwargs", graph)
        # add the tracked params from iterables in args
        for tracked_var in tracked_in_args + tracked_in_kwargs:
            fn_variables += tracked_var
        # stateful
        self.constants.update(
            dict(
                zip(
                    [f"p{sid}" for sid in graph._stateful_param_ids],
                    graph._stateful,
                )
            )
        )
        # function body
        fn_body = ""
        nested_fn_body = ""
        # Iterate over every registered fn
        for f in graph._functions:
            if f.__name__ == "vmap":
                vectorized_fn_str, nested_fn_body, fn_header = generate_vmap_subgraph(
                    self, graph, f, frontend, indent, nested_fn_body, fn_header
                )
                inner_fn_body = self.generate_inner_fn(
                    f, frontend=frontend, graph=graph
                )
                inner_fn_body = get_unindented_stmts(inner_fn_body)
                fn_body += vectorized_fn_str + inner_fn_body + "\n"
            else:
                inner_fn_body = self.generate_inner_fn(
                    f, frontend=frontend, graph=graph
                )
                get_unindented_stmts(inner_fn_body)
                fn_body += inner_fn_body
        # add constants to variables definition block
        self.clean_constants()

        if indent == 0:  # no need to add args & kwargs in scalar fn
            for c_id in self.constants:
                fn_variables += f"{c_id} = kwargs['{c_id}']\n    "
        # fn_str stores the compiled function definition
        fn_variables = get_unindented_stmts(fn_variables)
        fn_output = get_unindented_stmts(fn_output)
        fn_body = get_unindented_stmts(fn_body)
        if "scipy" in fn_body:
            fn_header += f"import scipy\n"
        if "jax._src" in fn_body and frontend is None:
            fn_header += "import jax._src as _src\n"
            fn_body = fn_body.replace("jax._src", "_src")
        fn_str = (
            fn_header
            + fn_signature
            + textwrap.indent(
                nested_fn_body + fn_variables + fn_body + fn_output, " " * 4
            )
        )
        return fn_str

    def clean_constants(self):
        """Removes integers and certain strings from the constants dictionary"""
        for k, v in list(self.constants.items()):
            if (
                v is None
                or isinstance(v, (bool, int))
                or (isinstance(k, str) and "'" in k)
            ):
                del self.constants[k]

    def generate_inner_fn(self, fn: Callable, frontend: str = None, graph=None) -> str:
        """Generates a string which corresponds to a single functional node of the graph."""
        # function name -> needs to be correctly formatted
        if fn.__name__ == "vmap":
            fn_name = "vectorized_fn"
        else:
            fn_name = get_fn_name(fn, to_ivy=graph._to_ivy)
        if frontend:
            paths_to_replace = glob.NATIVE_TO_FRONTEND_PATH
            if any([p in fn_name for p in paths_to_replace]):
                old_path, new_path = [
                    (k, v) for k, v in paths_to_replace.items() if k in fn_name
                ][0]
                fn_name = fn_name.replace(old_path, new_path)
        # function type -> can be a method or a function
        is_getattr = fn_name in ["__getattr__", "__getattribute__"]
        is_input_to_output = fn_name.split(".")[-1] == "input_to_output"
        is_method = (
            (fn.inplace_fn or fn_name[:2] == "__" or fn.from_tracked_var)
            and not is_input_to_output
            and "tvp" not in fn.__name__
        )
        is_inplace = fn.inplace_fn
        # args -> we may have to remove the first one if calling a method
        args_str = self.generate_inner_fn_args(fn, from_="args")
        kwargs_str = self.generate_inner_fn_args(fn, from_="kwargs")
        args_n_kwargs_str = join_args_n_kwargs(args_str, kwargs_str)
        if (
            is_method or is_inplace
        ) and fn.__name__ in glob.INPLACE_FUNCTIONS_WITHOUT_RET[
            ivy.current_backend_str()
        ]:
            instance, _ = method_args_from(args_n_kwargs_str)
        elif is_method or is_inplace:
            instance, args_n_kwargs_str = method_args_from(args_n_kwargs_str)
            fn_name = f"{instance}.{fn_name}"
        if is_input_to_output:
            args_n_kwargs_str = args_n_kwargs_str.split(",")[0]
        # output -> can be inplace or not
        output_str = self.generate_inner_fn_output(fn)
        # indexing -> we may need to remove a 1-item tuple
        indexing = "[0]" if hasattr(fn.backend_fn, "tuple_subclass_output") else ""
        indexing = "[0]" if fn.remove_output_tuple and not fn.terminal else indexing
        # delete any intermediate var that won't be used again
        del_statement = ""
        to_delete = []
        for a in fn.arg_param_ids + fn.kwarg_param_ids:
            self.reference_count[a] = self.reference_count[a] - 1
            if self.reference_count[a] == 0:
                to_delete.append(f"p{a}")
        if to_delete:
            del_statement = "del " + ", ".join(to_delete) + "\n    "
        # return the function string
        if is_inplace:
            fn_str = f"{fn_name}({args_n_kwargs_str}){indexing}\n    "
            fn_str += f"{output_str} = {instance}\n    "
        elif is_getattr:
            fn_str = f"{output_str} = getattr({instance}, {args_n_kwargs_str})\n    "
        elif is_input_to_output:
            fn_str = f"{output_str} = {args_n_kwargs_str}\n    "
        else:
            fn_str = f"{output_str} = {fn_name}({args_n_kwargs_str}){indexing}\n    "
        if glob.check_frontend_vs_original_differences:
            if frontend:
                fn_str += f"glob.frontend_results.append({output_str})\n    "
            else:
                fn_str += f"glob.original_results.append({output_str})\n    "
        return fn_str + del_statement

    def generate_inner_fn_args(self, f: Callable, from_: str) -> str:
        """Generates a string which contains the args of the specified function.
        This function also stores any constant variable in the self.constant dict."""
        if from_ not in ["args", "kwargs"]:
            raise ValueError("from_ must be one of 'args' or 'kwargs'.")
        from_args = from_ == "args"
        args = f.args if from_args else f.kwargs
        idxs = f.arg_tracked_idxs if from_args else f.kwarg_tracked_idxs
        ids = f.arg_param_ids if from_args else f.kwarg_param_ids
        # Replace tracked args with their ids
        args_reprs = ivy.set_nest_at_indices(args, idxs, ids, shallow=False)
        # Replace non tracked args with a variable that stores their value
        args_reprs = self.replace_nontracked_from_reprs(args_reprs, ids, idxs)
        # Replace tracked args with the corresponding intermediate variable
        args_reprs = ivy.map_nest_at_indices(args_reprs, idxs, lambda x: f"p{x}")
        # store subclasses as context variables
        subclass_idx = ivy.nested_argwhere(
            args_reprs,
            lambda x: isinstance(x, (list, tuple, dict))
            and type(x) not in [list, tuple, dict],
            check_nests=True,
        )
        subclass_values = ivy.multi_index_nest(args_reprs, subclass_idx)
        subclass_names = [f"init_{type(x).__name__}" for x in subclass_values]
        subclass_types = [type(x) for x in subclass_values]
        # add variables to function context
        self.constants.update(dict(zip(subclass_names, subclass_values)))
        # Fabricate the args string
        _slice_idxs = f.with_tracked_slices
        args_reprs = nest_to_str(
            args_reprs, subclass_names, subclass_types, _slice_idxs
        )
        return args_reprs

    def replace_nontracked_from_reprs(self, reprs, param_ids, tracked_idxs):
        # get indices of the values in args that are not tracked
        nontracked_idxs = ivy.nested_argwhere(
            reprs,
            lambda x: type(x) is not int or x not in param_ids,
            check_nests=True,
        )
        nestables_idxs = ivy.nested_argwhere(
            reprs,
            lambda x: isinstance(x, (list, tuple, dict)),
            check_nests=True,
        )
        # remove nestables from constants if not all inner values are constants
        nontracked_idxs = remove_non_constant_nests_idxs(
            nontracked_idxs, nestables_idxs, tracked_idxs
        )
        # get values at said indices
        nontracked_values = ivy.multi_index_nest(reprs, nontracked_idxs)
        # generate a unique id for each value
        nontracked_ids = [
            f"c{helpers._get_unique_id(x)}"
            if not isinstance(x, (int, str)) and x is not None
            else f"'{x}'"
            if isinstance(x, str)
            else x
            if x is not None
            else "None"
            for x in nontracked_values
        ]
        glob.logging_paused = True
        # add variables to function context
        context = dict(zip(nontracked_ids, nontracked_values))
        self.constants.update(context)
        # replace reprs with the new variables
        reprs = ivy.set_nest_at_indices(reprs, nontracked_idxs, nontracked_ids)
        return reprs

    def generate_inner_fn_output(self, f: Callable) -> str:
        output_reprs = ivy.set_nest_at_indices(
            f.output, f.output_tracked_idxs, f.output_param_ids, shallow=False
        )
        # output_reprs can be nested --> [[...], [...]] while
        # output_param_ids will always be flat --> [<param_id_1>, <param_id_2>]
        # so need to compare the two lists keeping the nesting in mind
        reprs = []
        flat_reprs = helpers.flatten(output_reprs)
        for o in flat_reprs:
            reprs.append(f"p{o}" if o in f.output_param_ids else "_")
        output_reprs = ", ".join(reprs) if reprs else "_"
        return output_reprs


# Helpers


def get_unindented_stmts(text):
    lines = text.splitlines()
    unindented_lines = [line.lstrip() for line in lines]
    unindented_text = "\n".join(unindented_lines)
    return unindented_text


def join_args_n_kwargs(args_str: str, kwargs_str: str) -> str:
    """Generates a string which contains args and kwargs correctly formatted to script a function.
    Parameters
    ----------
    args_str : str
        String containing the arguments of a function. (i.e. "x1, x2").
    kwargs_str : str
        String containing the keyword arguments of a function. (i.e. "kw1=v1, kw2=v2").
    Returns
    -------
    str
        Correctly formatted arguments and keyword arguments.
        (i.e. "x1, x2, kw1=v1, kw2=v2").
    """
    valid_args_n_kwargs = [i for i in [args_str, kwargs_str] if i]
    return ", ".join(valid_args_n_kwargs)


def nest_to_str(
    nest, _inits=None, _init_types=None, _slice_idxs=None, _base=True, _index=None
):
    """Takes a nestable which holds strings and integers and correctly formats the nested structure into
    a final string."""
    # all arguments should have been already converted to strings
    _inits = ivy.default(_inits, [])
    _init_types = ivy.default(_init_types, [])
    _slice_idxs = ivy.default(_slice_idxs, [])
    _index = [] if _base else _index
    is_subclass_init = type(nest) in _init_types
    is_slice = _index in _slice_idxs
    if isinstance(nest, (tuple, list)):
        if isinstance(nest, list):
            opening, closing = "[", "]"
        elif isinstance(nest, tuple) and not hasattr(nest, "_fields"):
            opening, closing = "(", ")"
        _args = [
            nest_to_str(item, _inits, _init_types, _slice_idxs, False, _index + [i])
            for i, item in enumerate(nest)
        ]
        if len(_args) == 1 and not _base:
            _args[0] += ","
    elif isinstance(nest, dict):
        opening, closing = "{", "}"
        union = ": " if not _base else "="
        union = "=" if is_subclass_init else union
        _args = {
            k: nest_to_str(v, _inits, _init_types, _slice_idxs, False, _index + [k])
            for k, v in nest.items()
        }
        if isinstance(nest, ivy.Container):  # ToDo: Move to _inits?
            opening, closing = "ivy.Container(", ")"
            union = "="
        # Wrap raw dict keys in string quotations
        _args = [
            f"'{k}'{union}{v}"
            if isinstance(k, str) and union == ": "
            else f"{k}{union}{v}"
            for k, v in _args.items()
        ]
    else:
        if not isinstance(nest, str):
            return str(nest)
        return nest
    if _base:
        opening, closing = "", ""
    if is_slice:
        opening, closing = "slice(", ")"
    if is_subclass_init:
        idx = _init_types.index(type(nest))
        subclass_name = _inits[idx]
        opening, closing = f"{subclass_name}(", ")"
        # TODO: Add check for normal namedtuples
        if "torch.return_types" in str(type(nest)):
            opening = opening + "["
            closing = "]" + closing
    repr = f"{opening}{', '.join(_args)}{closing}"
    return repr


def get_fn_name(fn: Callable, to_ivy: bool = False) -> str:
    "Gets the correct function name from a given function."
    if fn.__name__ in ["__getattribute__", "__getattr__", "__getitem__", "__setitem__"]:
        return fn.__name__
    if to_ivy:
        return (
            FUNC_TO_PATH[fn.backend_fn] if not fn.from_tracked_var else f"{fn.__name__}"
        )
        return fn_name
    if (
        fn.inplace_fn
        and fn.__name__
        not in glob.INPLACE_FUNCTIONS_WITHOUT_RET[ivy.current_backend_str()]
    ):
        return fn.backend_fn.__name__
    try:
        fn_path = FUNC_TO_PATH[fn.backend_fn]
    except KeyError:
        fn_path = fn.backend_fn.__name__
    if "tvp__" in fn_path:
        fn_path = fn_path[5:-2]
    elif fn.from_tracked_var:
        return fn.__name__
    return fn_path


def method_args_from(args_n_kwargs):
    args = args_n_kwargs.split(", ")
    instance = args[0]
    method_args = ", ".join(args[1:])
    return instance, method_args


def load_fn_from_str(source_str):
    """Executes the source code passed as arguments and returns the defined "compiled_fn" """
    namespace = {}
    exec(source_str, namespace)
    compiled_fn = namespace["compiled_fn"]
    return compiled_fn


def load_fn_from_file(source_str):
    """Saves the generated source code into a file and imports said file as a module.
    This allows the user to step into the (scripted) compiled function."""
    # ToDo: fix path of intermediate file
    file_path = "ivy_temp_script.py"
    module_name = "ivy_compiled_fn"
    with open(file_path, "w") as f:
        f.write(source_str)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module at {file_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        compiled_fn = module.__dict__["compiled_fn"]
    except:
        raise ImportError("Error while loading the compiled function as a module.")
    return compiled_fn


def remove_non_constant_nests_idxs(nontracked_idxs, nestables_idxs, tracked_idxs):
    # remove any nested types idxs from nontracked if there is any tracked param inside it
    to_remove = []
    for nest_idx in nestables_idxs:
        for idx in tracked_idxs:
            if idx[: len(nest_idx)] == nest_idx:
                to_remove.append(nest_idx)
                break
    for nidx in to_remove:
        nontracked_idxs.remove(nidx)
        nestables_idxs.remove(nidx)
    # remove any inner idxs from nontracked if its outer nest is constant
    redundant = []
    for nidx in nontracked_idxs:
        for nest_idx in nestables_idxs:
            if nidx != nest_idx and nidx[: len(nest_idx)] == nest_idx:
                redundant.append(nidx)
                break
    [nontracked_idxs.remove(nidx) for nidx in redundant]
    return nontracked_idxs
