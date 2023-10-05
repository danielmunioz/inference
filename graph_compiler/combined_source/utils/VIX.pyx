import base64
import pickle

from . import VIV as sg
from .IIV import Graph

from .XVV import CacheDB
from .VVX import (detect_differences, parse_all_lines,
                                    retrieve_nested_source_raw, trace_obj)


class Cacher():
    def __init__(self) -> None:
        self.db = CacheDB()

    def get_trace_result(self, obj, args, kwargs, compile_kwargs):
        obj_call, code_loc, code_line, func_def, args_str, kwargs_str, compile_kwargs_str = trace_obj(
            obj, args, kwargs, compile_kwargs)
        source_code = retrieve_nested_source_raw(obj, obj_call)

        return code_loc, code_line, func_def, args_str, kwargs_str, compile_kwargs_str, source_code

    def get_cached_traced_data(self, obj, args, kwargs, compile_kwargs):
        # get trace result
        code_loc, code_line, func_def, args_str, kwargs_str, compile_kwargs_str, source_code = self.get_trace_result(
            obj, args, kwargs, compile_kwargs)

        # find matching cache if it exists
        cache_list = self.db.select_matching_cache(
            code_loc, code_line, func_def, args_str, kwargs_str, compile_kwargs_str, source_code)

        # store traced results
        traced_data = {
            # these values are stored in their original form
            # and are used to re-create the graph
            "fn": obj,
            "args": args,
            "kwargs": kwargs,
            "compile_kwargs": compile_kwargs,
            # these values are stored in their string form
            "code_loc": code_loc,
            "code_line": code_line,
            "func_def": func_def,
            "args_str": args_str,
            "kwargs_str": kwargs_str,
            "compile_kwargs_str": compile_kwargs_str,
            "source_code": source_code,
        }

        cached_data = None
        if cache_list is not None and len(cache_list) != 0:
            cached_data = {
                "code_loc": cache_list[0][1],
                "code_line": cache_list[0][2],
                "func_def": cache_list[0][3],
                "args_str": cache_list[0][4],
                "kwargs_str": cache_list[0][5],
                "compile_kwargs_str": cache_list[0][6],
                "source_code": cache_list[0][7],
                "graph_fn_str": cache_list[0][8],
                "constants": cache_list[0][9],
            }

        return traced_data, cached_data

    def store_cache(self, traced_data, graph):
        # retrieve traced data
        code_loc = traced_data.get("code_loc")
        code_line = traced_data.get("code_line")
        func_def = traced_data.get("func_def")
        args_str = traced_data.get("args_str")
        kwargs_str = traced_data.get("kwargs_str")
        compile_kwargs_str = traced_data.get("compile_kwargs_str")
        source_code_str = traced_data.get("source_code")

        # retrieve graph data
        graph_fn_str, constants = graph.obtain_sourcecode()
        graph_fn_str = str(graph_fn_str).replace("'", '"')
        constants = self._serialize_object(constants)

        # store cache
        self.db.insert_cache(
            code_loc, code_line, func_def, args_str, kwargs_str, compile_kwargs_str, source_code_str, graph_fn_str, constants)

    def return_all_cache(self):
        caches = self.db.select_all_cache()
        return caches

    def get_matching_graph(self, traced_data, cached_data):
        if cached_data is None:
            return None

        traced_code_src = traced_data.get("source_code")
        cached_code_src = cached_data.get("source_code")
        # check if the source code is the same, return the graph
        if not self.verify_execution_changed(traced_code_src, cached_code_src):
            fn = traced_data.get("fn")
            args = traced_data.get("args")
            kwargs = traced_data.get("kwargs")
            compile_kwargs = traced_data.get("compile_kwargs")

            compiled_fn = sg.load_fn_from_str(
                cached_data.get("graph_fn_str"))
            constants = self._deserialize_object(
                cached_data.get("constants"))

            cached_graph = Graph(fn, *args, **kwargs, **compile_kwargs,)
            cached_graph.initialize_from_cache(compiled_fn=compiled_fn, constants=constants)
            return cached_graph

    def verify_execution_changed(self, func_a_src, func_b_src):
        changed_lines, added_lines, removed_lines, diff_orig = detect_differences(
            func_a_src, func_b_src)
        total_execution_changed = parse_all_lines(
            added_lines, removed_lines, diff_orig)
        return total_execution_changed

    def _serialize_object(self, obj):
        for key, value in obj.items():  # paddle tensors are not serializable - convert them to numpy
            if "paddle.Tensor" in str(value.__class__):
                obj[key] = value.numpy()
        pickled = pickle.dumps(obj)
        return base64.b64encode(pickled).decode('utf-8')

    def _deserialize_object(self, obj):
        decoded = base64.b64decode(obj)
        return pickle.loads(decoded)
