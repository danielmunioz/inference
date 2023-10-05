# global
import os
import collections
from itertools import zip_longest
import csv
import dill
import gc
import ast
import argparse
import inspect
import importlib
import signal
from multiprocessing import Process, Manager, set_start_method
from queue import Empty
from multiprocessing.managers import ListProxy, DictProxy
import time
import psutil
from typing import Union, List, Tuple, Dict, Callable
import builtins
from builtins import *
import sys
import subprocess
import functools

try:
    from astunparse import unparse
except Exception:
    from ast import unparse

# local
sys.path.append(".")
sys.path.append("lib_scraping/scrape/")
import graph_compiler as ic
from graph_compiler import globals as globs
from graph_compiler import source_gen as sg
from graph_compiler.wrapping import _wrap_or_unwrap_module, _load_modules_from
from graph_compiler.helpers import _find_missing_frontends
from graph_compiler.conversion import nest_array_to_new_backend
from transpiler.transpiler import transpile
import ivy

# so we only have to add a new framework in scrape.py
from scrape import *

target_frameworks = {
    "function": tensor_types.keys(),
    "class": module_types.keys(),
}

# extra settings to help with compilation
ivy.compile = ic.compile
torch.set_grad_enabled(False)
torch.jit._state.disable()
jax.config.update("jax_enable_x64", True)
import jax.random
import ivy.functional.backends.jax as ivy_jax


def _compile_args_kwargs(arg: ast.AST) -> object:
    """
    Helper to convert an AST node to an object,
    mostly used to get arguments and keyword arguments from a string call.
    """
    expr = ast.Expression(body=arg)
    ast.fix_missing_locations(expr)
    return builtins.eval(builtins.compile(expr, filename="", mode="eval"))


def parse_function_call(func_str: str) -> Tuple[List[object], Dict[str, object]]:
    """
    In the form of func(args, kwargs)
    Parse the function call string into an AST node to extract objects.
    """
    call = ast.parse(func_str).body[-1].value

    args = [_compile_args_kwargs(arg) for arg in call.args]
    kwargs = {kwarg.arg: _compile_args_kwargs(kwarg.value) for kwarg in call.keywords}

    return args, kwargs


def parse_class_call(
    class_str: str,
) -> Tuple[str, List[object], Dict[str, object], str, List[object], Dict[str, object]]:
    """
    In the form of
    class.init_method(init_args, init_kwargs).call_method(call_args, call_kwargs)
    Parse the class call string into an AST node to extract objects and
    methods used to initialize and call the class.
    """
    call = ast.parse(class_str).body[-1].value

    try:
        # class().call_method()
        call_method = call.func.attr
        init = call.func.value
    except Exception:
        # class()()
        call_method = None
        init = call.func
    try:
        # class.init_method()()
        init_method = init.func.attr
    except Exception:
        # class()()
        init_method = None

    call_args = [_compile_args_kwargs(arg) for arg in call.args]
    call_kwargs = {
        kwarg.arg: _compile_args_kwargs(kwarg.value) for kwarg in call.keywords
    }
    init_args = [_compile_args_kwargs(arg) for arg in init.args]
    init_kwargs = {
        kwarg.arg: _compile_args_kwargs(kwarg.value) for kwarg in init.keywords
    }

    return init_method, init_args, init_kwargs, call_method, call_args, call_kwargs


def is_tensor_only_in_attribute(
    args: List[object],
    kwargs: Dict[str, object],
    result: object,
    native_tensor_type: Tuple[type, ...],
) -> bool:
    """
    We used a modified ivy.nested_any with an extra flag check_attributes=True
    to also check for tensor in an object's attributes
    i.e. some_lib.some_container.Box(torch.tensor(1), torch.tensor(2))
    but this attracted many false positives, especially optimizers
    i.e. SGD(model.parameters(), lr=0.1)
    """
    return not (
        ivy.nested_any(
            args,
            lambda x: builtins.isinstance(x, native_tensor_type),  # noqa
        )
        or ivy.nested_any(
            kwargs,
            lambda x: builtins.isinstance(x, native_tensor_type),  # noqa
        )
    ) or not ivy.nested_any(
        result,
        lambda x: builtins.isinstance(x, native_tensor_type),  # noqa
    )


def _to_multiprocessing_friendly_and_back(
    done: Union[List[str], ListProxy],
    control_flow: Union[List[str], ListProxy],
    function_paths: Union[List[str], ListProxy],
    missing_frontends: Union[List[str], ListProxy],
    time_benchmark: Union[Dict, DictProxy],
    failed: Union[Dict, DictProxy],
    manager: Manager,
    to_multiprocessing: bool,
) -> Tuple[
    Union[ListProxy, List[str]],
    Union[ListProxy, List[str]],
    Union[ListProxy, List[str]],
    Union[ListProxy, List[str]],
    Union[DictProxy, Dict],
    Union[DictProxy, Dict],
]:
    """
    We must use a shared manager object to communicate between processes
    but dill can't pickle them, so we convert to a native object
    before saving checkpoint and vice versa.
    """
    if to_multiprocessing:
        convert_list = manager.list
        convert_dict = manager.dict
    else:
        convert_list = list
        convert_dict = dict
    done = convert_list(done)
    control_flow = convert_list(control_flow)
    function_paths = convert_list(function_paths)
    missing_frontends = convert_list(missing_frontends)
    time_benchmark = convert_dict(time_benchmark)
    failed = convert_dict(failed)
    for metric in failed:
        failed[metric] = convert_list(failed[metric])
    return done, control_flow, function_paths, missing_frontends, time_benchmark, failed


def compile_functions_classes(
    queue: object,
    process: int,
    library: str,
    done: ListProxy,
    control_flow: ListProxy,
    function_paths: ListProxy,
    missing_frontends: ListProxy,
    time_benchmark: DictProxy,
    failed: DictProxy,
    fw: str,
    time_counter: ListProxy,
    is_benchmark: bool,
) -> None:
    """
    Parameters
    ----------
    queue
        A manager.Queue object, contains the scraped callables
        where each item has the format (fn_or_cls, callable_path, code_imports)
        fn_or_cls: str
            Either "function" or "class"
        callable_path: str
            The path of the original callable
        code_imports: Dict
            key "code": The scraped call as a string
            key "imports": A list of imports required to evaluate the scraped call
    process
        From 0 to num_processes-1, to help with debugging
    library
        mod_name of a library
    done
        List of completed callable strings
    control_flow
        List of callable strings that has control flow when compiled
    function_paths
        List of functions inside the compiled graph of a callable
    missing_frontends
        List of missing frontends found after compiling a callable
    time_benchmark
        Dictionary to record time performance of a callable in native, compiled
        and transpiled
    failed
        Dictionary to record if a callable failed to eval/compile/transpile
        or has unmatching outputs
    fw
        The framework that we're currently compiling for
    time_counter
        List of time.time() for each process to poll if a process is stuck
        while eval/compile/transpile
    is_benchmark
        Whether we want to benchmark or not
    """

    def process_print(*msg):
        """Prepend a process number to each print for easier debugging."""
        builtins.print(process, *msg, flush=True)

    def benchmark_call(
        benchmark_dict: DictProxy,
        callable_path: str,
        metric: str,
        lambda_call: Callable,
        iterations=10,
    ) -> None:
        """
        Calculate mean and standard deviation when calling a function/model
        for n iterations.

        Parameters
        ----------
        benchmark_dict
            A time benchmark dictionary that will be used to save to .csv
        callable_path
            The path of the original function/model call
        metric
            Either "native", "native compiled", "ivy compile", "fw" or "fw compiled"
            for each framework that we want to transpile to
        lambda_call
            The call that we want to benchmark, passed in as a lambda call
            so we don't have to add framework-specific logic to benchmarking
        iterations
            The number of iterations to benchmark, excluding the first warmup call
        """
        try:
            timing = []
            for iteration in builtins.range(iterations + 1):
                start = time.perf_counter()
                lambda_call()
                if iteration == 0:
                    # warmup
                    continue
                timing.append(time.perf_counter() - start)

            if callable_path not in benchmark_dict:
                benchmark_dict[callable_path] = builtins.dict()
            benchmark_result = benchmark_dict[callable_path]
            benchmark_result[metric] = builtins.dict()
            benchmark_result[metric]["mean"] = np.mean(timing)
            benchmark_result[metric]["std"] = np.std(timing)
            benchmark_dict[callable_path] = benchmark_result
        except Exception as e:
            process_print(
                callable_path, f"- Benchmark {metric} error -", builtins.repr(e)
            )

    def find_control_flow(f):
        """
        Find functions/classes failing while having control flow.
        Doesn't mean that they are failing due to control flow.
        Used to wrap ivy.if_else, while_loop and for_loop.
        """

        @functools.wraps(f)
        def control_flow_wrapper(*args, **kwargs):
            nonlocal control_flow_detected
            control_flow_detected = True
            return f(*args, **kwargs)

        return control_flow_wrapper

    control_flow_detected = False
    builtins.exec(f"import {library}", builtins.globals())
    native_tensor_type = tensor_types[fw]

    while True:
        time_counter[process] = time.time()
        ivy.set_backend(fw)
        ivy_jax.RNG.key = jax.random.PRNGKey(0)

        # reset control flow flag
        ivy.if_else = find_control_flow(ivy.if_else)
        ivy.while_loop = find_control_flow(ivy.while_loop)
        ivy.for_loop = find_control_flow(ivy.for_loop)
        control_flow_detected = False

        try:
            fn_or_cls, callable_path, code_imports = queue.get(block=False)
        except Empty:
            process_print("Queue is empty, stopping process")
            time_counter[process] = None
            return

        code, imports = code_imports["code"], code_imports["imports"]
        full_code = callable_path + code[code.index("(") :]
        if (
            full_code in done
            or full_code in failed["oom"]
            or full_code in failed["import"]
            or full_code in failed["tensor_only_in_attribute"]
        ):
            # skip if already done
            continue

        # TODO: https://github.com/unifyai/graph-compiler/issues/237
        if "register" in full_code or "lambda" in full_code:
            continue

        process_print(full_code)
        globals_copy = builtins.globals().copy()
        try:
            # import other libs first (torch, numpy...)
            for lib in imports:
                if not lib.startswith(library):
                    builtins.exec(f"from {lib} import *", builtins.globals())
            # import the target library last to overwrite
            # in case they both have a function of the same name
            for lib in imports:
                if lib.startswith(library):
                    builtins.exec(f"from {lib} import *", builtins.globals())

            # get function/class object
            if "." in callable_path:
                obj = builtins.eval(callable_path.split(".")[-1])
            else:
                obj = builtins.eval(callable_path)

            # get args/kwargs and try to eval the call natively before compiling
            if fn_or_cls == "function":
                args, kwargs = parse_function_call(code)
                call_method = None
                ret = obj(*args, **kwargs)
                if is_benchmark:
                    # native
                    if fw == "jax":
                        lambda_call = lambda: jax.block_until_ready(
                            obj(*args, **kwargs)
                        )
                    else:
                        lambda_call = lambda: obj(*args, **kwargs)
                    benchmark_call(time_benchmark, callable_path, "native", lambda_call)

                    # native compiled
                    if fw == "torch":
                        native_compiled_obj = torch.compile(obj)
                        lambda_call = lambda: native_compiled_obj(*args, **kwargs)
                        benchmark_call(
                            time_benchmark,
                            callable_path,
                            "native compiled",
                            lambda_call,
                        )
                    elif fw == "jax":
                        native_compiled_obj = jax.jit(obj)
                        lambda_call = lambda: jax.block_until_ready(
                            native_compiled_obj(*args, **kwargs)
                        )
                        benchmark_call(
                            time_benchmark,
                            callable_path,
                            "native compiled",
                            lambda_call,
                        )
                        del native_compiled_obj, lambda_call
            elif fn_or_cls == "class":
                (
                    init_method,
                    init_args,
                    init_kwargs,
                    call_method,
                    args,
                    kwargs,
                ) = parse_class_call(code)
                if not init_method:
                    obj = obj(*init_args, **init_kwargs)
                else:
                    obj = builtins.getattr(obj, init_method)(*init_args, **init_kwargs)

                if call_method is None:
                    call_method = "__call__"

                # remove dropout, batchnorm...
                try:
                    obj.eval()
                except Exception:
                    pass
                if fw in ("tensorflow", "jax"):
                    kwargs["training"] = False

                ret = builtins.getattr(obj, call_method)(*args, **kwargs)
                if is_benchmark:
                    # native
                    if fw == "jax":
                        lambda_call = lambda: jax.block_until_ready(
                            builtins.getattr(obj, call_method)(*args, **kwargs)
                        )
                    else:
                        lambda_call = lambda: builtins.getattr(obj, call_method)(
                            *args, **kwargs
                        )
                    benchmark_call(time_benchmark, callable_path, "native", lambda_call)

                    # native compiled
                    if fw == "torch":
                        native_compiled_obj = torch.compile(obj)
                        lambda_call = lambda: builtins.getattr(
                            native_compiled_obj, call_method
                        )(*args, **kwargs)
                        benchmark_call(
                            time_benchmark,
                            callable_path,
                            "native compiled",
                            lambda_call,
                        )
                    elif fw == "jax":
                        native_compiled_obj = jax.jit(
                            builtins.getattr(obj, call_method)
                        )
                        lambda_call = lambda: jax.block_until_ready(
                            native_compiled_obj(*args, **kwargs)
                        )
                        benchmark_call(
                            time_benchmark,
                            callable_path,
                            "native compiled",
                            lambda_call,
                        )
                        del native_compiled_obj, lambda_call

        except Exception as e:
            process_print(full_code, "- Import error -", builtins.repr(e))
            failed["import"].append(full_code)
            continue

        if is_tensor_only_in_attribute(args, kwargs, ret, native_tensor_type):
            process_print(full_code, " - Tensor only in Attribute")
            failed["tensor_only_in_attribute"].append(full_code)
            continue

        # preemptively mark as OOM so we can skip it when a new process is created
        failed["oom"].append(full_code)
        gc.collect()
        try:
            # compile
            if call_method is None:
                compiled = ic.compile(obj)
            else:
                compiled = ic.compile(builtins.getattr(obj, call_method))
            comp_ret = compiled(*args, **kwargs)
            if is_benchmark:
                lambda_call = lambda: compiled(*args, **kwargs)
                benchmark_call(
                    time_benchmark, callable_path, "ivy compile", lambda_call
                )

            if control_flow_detected:
                control_flow.append(full_code)

            eager_graph = compiled._eager_graph

            # get functions in graph
            fn_paths = [
                sg.FUNC_TO_PATH[fn]
                for fn in eager_graph.list_function_frequencies(return_raw=True)
                if fn in sg.FUNC_TO_PATH
            ]
            function_paths.append(fn_paths)

            # get missing frontends
            missing = builtins.dict(_find_missing_frontends(eager_graph))
            missing_frontends.append(missing)

            # if there is a random function, we don't compare the results
            has_random = builtins.any(
                fn.split(".")[-1] in globs.GENERATOR_FUNCTIONS[fw] for fn in fn_paths
            )
            if not has_random:
                try:
                    if not check_same_output(ret, comp_ret):
                        failed[f"compile_{fn_or_cls}_output"].append(full_code)
                        if control_flow_detected:
                            failed["cf_output"].append(full_code)
                except Exception as e:
                    process_print(
                        full_code, "- Compile output error -", builtins.repr(e)
                    )
                    failed[f"compile_{fn_or_cls}_output"].append(full_code)
                    if control_flow_detected:
                        failed["cf_output"].append(full_code)

            # transpile
            if len(missing) == 0:  # only transpile if there's no missing frontends
                failed_transpile_fw = []
                failed_transpile_output_fw = []

                native_params_v = None
                if fw == "jax" and call_method == "apply":
                    native_params_v = args[0]
                    args = args[1:]

                for to in target_frameworks[fn_or_cls]:
                    try:
                        ivy.set_backend(fw)

                        # transpile eagerly
                        transpiled = transpile(
                            obj,
                            source=fw,
                            to=to,
                            params_v=native_params_v,
                            args=args,
                            kwargs=kwargs,
                        )

                        # flax, haiku special handling
                        target_backend = to
                        if to in ("flax", "haiku"):
                            target_backend = "jax"

                        # convert arguments into target framework
                        ivy.set_backend(target_backend)
                        new_args = nest_array_to_new_backend(
                            args, native=True, shallow=False
                        )
                        new_kwargs = nest_array_to_new_backend(
                            kwargs, native=True, shallow=False
                        )

                        # TODO: when CF is done, revise this for dropout
                        if "training" in new_kwargs:
                            del new_kwargs["training"]

                        if to == "haiku":
                            haiku_module = hk.transform(
                                lambda *arg, **kwarg: transpiled()(*arg, **kwarg)
                            )
                            params_v = haiku_module.init(
                                None,
                                *new_args,
                                **new_kwargs,
                            )
                            trans_ret = haiku_module.apply(
                                params_v,
                                None,
                                *new_args,
                                **new_kwargs,
                            )
                            if is_benchmark:
                                # transpiled
                                lambda_call = lambda: jax.block_until_ready(
                                    haiku_module.apply(
                                        params_v, None, *new_args, **new_kwargs
                                    )
                                )
                                benchmark_call(
                                    time_benchmark, callable_path, to, lambda_call
                                )

                                # transpiled compiled
                                native_compiled_module = jax.jit(haiku_module.apply)
                                lambda_call = lambda: jax.block_until_ready(
                                    native_compiled_module(
                                        params_v, None, *new_args, **new_kwargs
                                    )
                                )
                                benchmark_call(
                                    time_benchmark,
                                    callable_path,
                                    to + " compiled",
                                    lambda_call,
                                )
                        elif to == "flax":
                            params_v = transpiled.init(
                                jax.random.PRNGKey(0),
                                *new_args,
                                **new_kwargs,
                            )
                            trans_ret = transpiled.apply(
                                params_v,
                                *new_args,
                                **new_kwargs,
                            )
                            if is_benchmark:
                                # transpiled
                                lambda_call = lambda: jax.block_until_ready(
                                    transpiled.apply(params_v, *new_args, **new_kwargs)
                                )
                                benchmark_call(
                                    time_benchmark, callable_path, to, lambda_call
                                )

                                # transpiled compiled
                                native_compiled_module = jax.jit(transpiled.apply)
                                lambda_call = lambda: jax.block_until_ready(
                                    native_compiled_module(
                                        params_v, *new_args, **new_kwargs
                                    )
                                )
                                benchmark_call(
                                    time_benchmark,
                                    callable_path,
                                    to + " compiled",
                                    lambda_call,
                                )
                        else:
                            trans_ret = transpiled(*new_args, **new_kwargs)
                            if is_benchmark:
                                # transpiled
                                lambda_call = lambda: transpiled(
                                    *new_args, **new_kwargs
                                )
                                benchmark_call(
                                    time_benchmark, callable_path, to, lambda_call
                                )

                                # transpiled compiled
                                if to == "torch":
                                    # torch.compile doesn't like Graph
                                    native_compiled_module = torch.compile(
                                        lambda: transpiled
                                    )
                                    lambda_call = lambda: native_compiled_module()(
                                        *new_args, **new_kwargs
                                    )
                                    benchmark_call(
                                        time_benchmark,
                                        callable_path,
                                        to + " compiled",
                                        lambda_call,
                                    )
                                elif to == "jax":
                                    # only a function transpiles to jax
                                    native_compiled_module = jax.jit(transpiled)
                                    lambda_call = lambda: jax.block_until_ready(
                                        native_compiled_module(*new_args, **new_kwargs)
                                    )
                                    benchmark_call(
                                        time_benchmark,
                                        callable_path,
                                        to + " compiled",
                                        lambda_call,
                                    )

                        # use ivy's implicit backend to use ivy.to_numpy
                        # between different frameworks
                        ivy.unset_backend()
                        try:
                            if not has_random and not check_same_output(ret, trans_ret):
                                failed_transpile_output_fw.append(to)
                                if control_flow_detected:
                                    failed["cf_output"].append(full_code)
                        except Exception as e:
                            process_print(
                                full_code,
                                f"- Transpile to {to} output error -",
                                builtins.repr(e),
                            )
                            failed_transpile_output_fw.append(to)
                            if control_flow_detected:
                                failed["cf_output"].append(full_code)

                    except Exception as e:
                        process_print(
                            full_code, f"- Transpile to {to} error -", builtins.repr(e)
                        )
                        failed_transpile_fw.append(to)
                        if control_flow_detected:
                            failed["cf"].append(full_code)

                if len(failed_transpile_fw):
                    failed[f"transpile_{fn_or_cls}"].append(
                        f"{full_code}: {', '.join(failed_transpile_fw)}"
                    )

                if len(failed_transpile_output_fw):
                    failed[f"transpile_{fn_or_cls}_output"].append(
                        f"{full_code}: {', '.join(failed_transpile_output_fw)}"
                    )

        except Exception as e:
            process_print(full_code, "- Compile error -", builtins.repr(e))
            # TODO: https://github.com/unifyai/graph-compiler/issues/237
            if "WrappedCallable" in repr(e):
                builtins.exit(9)
            failed[f"compile_{fn_or_cls}"].append(full_code)
            if control_flow_detected:
                failed["cf"].append(full_code)

        builtins.globals().update(globals_copy)
        failed["oom"].remove(full_code)
        done.append(full_code)


def compile_framework(
    library: str,
    callables: Dict,
    fw: str,
    manager: Manager,
    num_processes: int,
    is_benchmark: bool,
) -> Tuple[Dict, Dict, Dict]:
    """
    Compile a framework by putting scraped functions / classes calls into a queue,
    then spin up processes to consume that queue.
    This has timeout management so if a call is stuck for too long for unknown reasons,
    it will be marked as OOM and the process will be revived then skip it.
    """
    queue = (
        manager.Queue()
    )  # create a queue to distribute functions/classes to processes
    time_counter = (
        manager.list()
    )  # keep track of which process is hanging / stuck for too long
    stuck_timeout = 1800

    for fn_or_cls in callables.keys():
        for callable_path, code_imports in callables[fn_or_cls].items():
            queue.put((fn_or_cls, callable_path, code_imports))

    if os.path.exists(
        f"lib_scraping/compile/result/{library}/checkpoint/temp_{fw}.dil"
    ):
        # reload from checkpoint
        with open(
            f"lib_scraping/compile/result/{library}/checkpoint/temp_{fw}.dil", "rb"
        ) as handle:
            (
                done,
                control_flow,
                function_paths,
                missing_frontends,
                time_benchmark,
                failed,
            ) = dill.load(handle)
            (
                done,
                control_flow,
                function_paths,
                missing_frontends,
                time_benchmark,
                failed,
            ) = _to_multiprocessing_friendly_and_back(
                done,
                control_flow,
                function_paths,
                missing_frontends,
                time_benchmark,
                failed,
                manager,
                to_multiprocessing=True,
            )
    else:
        # first time, setup as multiprocessing friendly objects
        done = manager.list()
        control_flow = manager.list()
        function_paths = manager.list()
        missing_frontends = manager.list()
        time_benchmark = manager.dict()
        failed = manager.dict()
        failed["compile_function"] = manager.list()
        failed["compile_class"] = manager.list()
        failed["compile_function_output"] = manager.list()
        failed["compile_class_output"] = manager.list()
        failed["transpile_function"] = manager.list()
        failed["transpile_class"] = manager.list()
        failed["transpile_function_output"] = manager.list()
        failed["transpile_class_output"] = manager.list()
        failed["oom"] = manager.list()
        failed["import"] = manager.list()
        failed["tensor_only_in_attribute"] = manager.list()
        failed["cf"] = manager.list()
        failed["cf_output"] = manager.list()

    processes = []
    for process_num in range(num_processes):
        time_counter.append(None)
        process = Process(
            target=compile_functions_classes,
            args=(
                queue,
                process_num,
                library,
                done,
                control_flow,
                function_paths,
                missing_frontends,
                time_benchmark,
                failed,
                fw,
                time_counter,
                is_benchmark,
            ),
            daemon=False,
        )
        processes.append(process)
        process.start()

    try:
        while any([p.exitcode != 0 for p in processes]):
            # keep polling for the processes' exit code.
            # if one died with a non-zero exit code,
            # create a new one in its place.
            time.sleep(5)

            for process_num in range(len(processes)):
                process = processes[process_num]
                # check if process died (most probably OOM)
                # check if a function got stuck
                if (process.exitcode is not None and process.exitcode != 0) or (
                    time_counter[process_num] is not None
                    and time.time() - time_counter[process_num] > stuck_timeout
                ):
                    if process.exitcode is not None:
                        print(
                            f"Detect process {process_num} died "
                            f"with exit code {process.exitcode}. Restarting..."
                        )
                    else:
                        print(
                            f"Detect process {process_num} hanging "
                            f"for more than {stuck_timeout} seconds. Restarting..."
                        )
                        process.kill()

                    time_counter[process_num] = None
                    new_process = Process(
                        target=compile_functions_classes,
                        args=(
                            queue,
                            process_num,
                            library,
                            done,
                            control_flow,
                            function_paths,
                            missing_frontends,
                            time_benchmark,
                            failed,
                            fw,
                            time_counter,
                            is_benchmark,
                        ),
                        daemon=False,
                    )
                    processes[process_num] = new_process
                    new_process.start()
    except KeyboardInterrupt:
        # save to checkpoint
        with builtins.open(
            f"lib_scraping/compile/result/{library}/checkpoint/temp_{fw}.dil", "wb"
        ) as handle:
            dill.dump(
                _to_multiprocessing_friendly_and_back(
                    done,
                    control_flow,
                    function_paths,
                    missing_frontends,
                    time_benchmark,
                    failed,
                    manager,
                    to_multiprocessing=False,
                ),
                handle,
                protocol=dill.HIGHEST_PROTOCOL,
            )
        builtins.print("Finished cleaning up")
        builtins.exit()

    print(f"Summarizing results for {fw}")
    (
        _,
        control_flow,
        function_paths,
        missing_frontends,
        time_benchmark,
        failed,
    ) = _to_multiprocessing_friendly_and_back(
        done,
        control_flow,
        function_paths,
        missing_frontends,
        time_benchmark,
        failed,
        manager,
        to_multiprocessing=False,
    )

    # remove duplicates in control flow
    failed["cf"] = list(set(failed["cf"]))
    failed["cf_output"] = list(set(failed["cf_output"]))

    # aggregate frequencies and sort by most common
    function_frequencies = collections.Counter()
    missing_functions = collections.Counter()
    for i in function_paths:
        function_frequencies += collections.Counter(i)
    for i in missing_frontends:
        missing_functions += collections.Counter(i)
    frequencies = {
        "function_frequencies": [
            [fw] + list(freq) for freq in function_frequencies.most_common()
        ],
        "missing_functions": [
            [fw] + list(freq) for freq in missing_functions.most_common()
        ],
    }

    # find ratio of failed to total for classes and functions
    failed["transpile_function"].insert(
        0,
        f"## {len(failed['transpile_function'])}"
        f"/"
        f"{len(callables['function']) - len(failed['compile_function'])} "
        f"Functions compiled in {fw} but can't be transpiled to",
    )
    failed["transpile_class"].insert(
        0,
        f"## {len(failed['transpile_class'])}"
        f"/"
        f"{len(callables['class']) - len(failed['compile_class'])} "
        f"Classes compiled in {fw} but can't be transpiled to",
    )
    failed["transpile_function_output"].insert(
        0,
        f"## {len(failed['transpile_function_output'])}"
        f"/"
        f"{len(callables['function']) - len(failed['compile_function'])} "
        f"Function transpile outputs unmatched between {fw} and",
    )
    failed["transpile_class_output"].insert(
        0,
        f"## {len(failed['transpile_class_output'])}"
        f"/"
        f"{len(callables['class']) - len(failed['compile_class'])} "
        f"Class transpile outputs unmatched between {fw} and",
    )

    failed["compile_function"].insert(
        0,
        f"## {len(failed['compile_function'])}"
        f"/"
        f"{len(callables['function'])} "
        f"Functions failed to compile",
    )
    failed["compile_class"].insert(
        0,
        f"## {len(failed['compile_class'])}"
        f"/"
        f"{len(callables['class'])} "
        f"Classes failed to compile",
    )
    failed["compile_function_output"].insert(
        0,
        f"## {len(failed['compile_function_output'])}"
        f"/"
        f"{len(callables['function'])} "
        f"Function compile outputs unmatched",
    )
    failed["compile_class_output"].insert(
        0,
        f"## {len(failed['compile_class_output'])}"
        f"/"
        f"{len(callables['class'])} "
        f"Class compile outputs unmatched",
    )
    failed["cf"].insert(
        0,
        f"## {len(failed['cf'])}"
        f"/"
        f"{len(control_flow)} "
        f"Have control flow but failed to compile/transpile",
    )
    failed["cf_output"].insert(
        0,
        f"## {len(failed['cf_output'])}"
        f"/"
        f"{len(control_flow)} "
        f"Have control flow but outputs unmatched",
    )

    if is_benchmark:
        # weird error that native fails in benchmark
        # even though we verified that it ran fine
        time_benchmark = {
            k: v
            for k, v in time_benchmark.items()
            if "native" in v and "mean" in v["native"]
        }

        # sort by native mean time
        time_benchmark = dict(
            sorted(
                time_benchmark.items(),
                key=lambda item: item[1]["native"]["mean"],
                reverse=True,
            )
        )

    # save fw checkpoint
    with open(
        f"lib_scraping/compile/result/{library}/checkpoint/compiled_{fw}.dil", "wb"
    ) as handle:
        dill.dump(
            [frequencies, failed, time_benchmark],
            handle,
            protocol=dill.HIGHEST_PROTOCOL,
        )

    # delete temp checkpoint
    try:
        os.remove(f"lib_scraping/compile/result/{library}/checkpoint/temp_{fw}.dil")
    except Exception:
        pass

    return frequencies, failed, time_benchmark


def check_same_output(func_ret: object, comp_ret: object) -> bool:
    """Compare two objects containing arrays"""
    if builtins.hasattr(func_ret, "__array__"):
        # compare 2 arrays
        return np.allclose(
            ivy.to_numpy(func_ret), ivy.to_numpy(comp_ret), equal_nan=True
        )
    else:
        # compare 2 nested objects with arrays
        idx = ivy.nested_argwhere(func_ret, lambda x: builtins.hasattr(x, "__array__"))
        np_func_ret = ivy.map_nest_at_indices(
            func_ret, idx, ivy.to_numpy, shallow=False
        )
        np_func_ret = ivy.multi_index_nest(np_func_ret, idx)
        np_comp_ret = ivy.map_nest_at_indices(
            comp_ret, idx, ivy.to_numpy, shallow=False
        )
        np_comp_ret = ivy.multi_index_nest(np_comp_ret, idx)
        same = []
        for f_ret, c_ret in builtins.zip(np_func_ret, np_comp_ret):
            if builtins.isinstance(f_ret, np.ndarray):
                same.append(np.allclose(f_ret, c_ret, equal_nan=True))
        return builtins.all(same)


def compile_library(
    library: str, scraped_lib: Dict, num_processes: int, is_benchmark: bool
) -> Tuple[Dict, Dict, Dict]:
    """
    Compile a library by going through each framework in it,
    also create a Manager object here so we don't create / destroy
    too many times between frameworks.
    """
    frequencies = dict()
    failed = dict()
    time_benchmark = dict()
    manager = Manager()

    for fw, callables in scraped_lib.items():
        if len(callables["function"]) > 0 or len(callables["class"]) > 0:
            if not os.path.exists(
                f"lib_scraping/compile/result/{library}/checkpoint/compiled_{fw}.dil"
            ):
                # start compiling
                frequencies[fw], failed[fw], time_benchmark[fw] = compile_framework(
                    library, callables, fw, manager, num_processes, is_benchmark
                )
            else:
                # reload completed checkpoint
                with open(
                    f"lib_scraping/compile/result/{library}/checkpoint/compiled_{fw}.dil",
                    "rb",
                ) as f:
                    frequencies[fw], failed[fw], time_benchmark[fw] = dill.load(f)

            # populate sg.FUNC_TO_PATH[fw] so we can later scrape for source frequencies
            modules = _load_modules_from(globs.MODULES_TO_WRAP[fw])
            for module in modules:
                _wrap_or_unwrap_module(lambda x: x, module, framework=fw, to_ivy=False)

            print(f"Done with framework {fw} of library {library}")

    return frequencies, failed, time_benchmark


def save_results(
    library: str,
    frequencies: Dict,
    failed: Dict,
    time_benchmark: Dict,
    is_benchmark: bool,
) -> None:
    """Save results to frequency.csv, failed.txt and (if specified) benchmark.csv"""
    # frequency.csv
    frequencies_by_framework = [
        frequencies[framework]["function_frequencies"] for framework in frequencies
    ]
    frequencies_by_framework = [
        fn for framework in frequencies_by_framework for fn in framework
    ]
    missing_by_framework = [
        frequencies[framework]["missing_functions"] for framework in frequencies
    ]
    missing_by_framework = [
        fn for framework in missing_by_framework for fn in framework
    ]
    data = [
        [fn[0] for fn in frequencies_by_framework],  # fw
        [fn[1] for fn in frequencies_by_framework],  # fn
        [fn[2] for fn in frequencies_by_framework],  # compile_freq
        [fn[3] for fn in frequencies_by_framework],  # source_freq
        [fn[0] for fn in missing_by_framework],      # ^
        [fn[1] for fn in missing_by_framework],      # |
        [fn[2] for fn in missing_by_framework],      # |
        [fn[3] for fn in missing_by_framework],      # |
    ]
    export_data = zip_longest(*data, fillvalue="")

    with open(f"lib_scraping/compile/result/{library}/frequency.csv", "w") as fp:
        csvwriter = csv.writer(fp)
        csvwriter.writerow(
            (
                "Framework",
                "Functions",
                "Compile Frequencies",
                "Source Frequencies",
                "Framework",
                "Missing Functions",
                "Compile Frequencies",
                "Source Frequencies",
            )
        )
        csvwriter.writerows(export_data)

    # failed.txt
    with open(f"lib_scraping/compile/result/{library}/failed.txt", "w") as fp:
        # requirements
        fp.write(
            "<details><summary>requirements.txt "
            "(install directly from unifyai/ivy:latest Docker)</summary><code>"
        )
        with open(f"lib_scraping/scrape/result/{library}/requirements.txt", "r") as f:
            fp.write(f.read().strip())
        fp.write("</code></details>\n\n")

        # which commit was used to compile
        fp.write("Compiled at commit ")
        with open(f"lib_scraping/compile/result/{library}/start_commit.txt", "r") as f:
            fp.write(f.read().strip())
        fp.write("\n\n")

        # import bug notice
        fp.write(
            "**Note**: if the full function path is wrong, please try\n"
            "```python\n"
            "from path.to import fn\n"
            "fn(args, kwargs)\n"
            "```\n"
            "instead of\n"
            "```python\n"
            "path.to.fn(args, kwargs)\n"
            "```\n"
        )
        fp.write(
            "There's a bug in most libraries that a function gets star-imported "
            "and replaces its father module with the same name.\n\n"
        )

        # result
        for fw in failed:
            fp.write(f"# {fw.capitalize()}\n")
            for metric in (
                "compile_function",
                "compile_class",
                "compile_function_output",
                "compile_class_output",
                "transpile_function",
                "transpile_class",
                "transpile_function_output",
                "transpile_class_output",
                "cf",
                "cf_output",
            ):
                fp.write("\n- ".join(failed[fw][metric]))
                fp.write("\n\n")

        # extra metrics
        fp.write("# Not related to compiling\n")
        extras = {
            "oom": [],
            "import": [],
            "tensor_only_in_attribute": [],
        }
        # aggregate extra metrics between multiple frameworks
        for fw in failed:
            for extra_metric in extras.keys():
                extras[extra_metric].extend(failed[fw][extra_metric])
        extras["oom"].insert(
            0, f"## {len(extras['oom'])} " f"went Out-Of-Memory while compiling"
        )
        extras["import"].insert(0, f"## {len(extras['import'])} had an Import issue")
        extras["tensor_only_in_attribute"].insert(
            0,
            f"## {len(extras['tensor_only_in_attribute'])} "
            f"only had a Tensor in Attribute",
        )
        for extra_metric in extras.keys():
            fp.write("\n- ".join(extras[extra_metric]))
            fp.write("\n\n")

    # benchmark.csv
    if is_benchmark:
        with open(
            f"lib_scraping/compile/result/{library}/benchmark.csv", "w"
        ) as handle:
            # setup metrics to write
            possible_transpile_to = set(target_frameworks["function"]).union(
                set(target_frameworks["class"])
            )
            possible_transpile_to = sorted(
                list(possible_transpile_to)
            )  # to keep ordering consistent
            possible_transpile_to_compiled = [
                fw + " compiled" for fw in possible_transpile_to
            ]
            possible_transpile_to = sorted(
                possible_transpile_to + possible_transpile_to_compiled
            )
            metrics = [
                "native",
                "native compiled",
                "ivy compile",
            ] + possible_transpile_to

            # header
            handle.write("Framework,Callable")
            for metric in metrics:
                handle.write(f",{metric.title()} Mean,{metric.title()} Std")
            handle.write("\n")

            # rows
            for fw in time_benchmark.keys():
                for callable, benchmark_result in time_benchmark[fw].items():
                    handle.write(f"{fw},{callable}")
                    for metric in metrics:
                        if metric in benchmark_result:
                            # scientific notation
                            handle.write(
                                f",{benchmark_result[metric]['mean']:.2E}"
                                f",{benchmark_result[metric]['std']:.2E}"
                            )
                        else:
                            handle.write(",,")
                    handle.write("\n")


class FunctionCallVisitor(ast.NodeVisitor):
    """To aid with source frequency collecting."""

    def __init__(self):
        self.framework_calls = dict()

    def visit_Import(self, node):
        """Import any module at the start of a file."""
        try:
            for alias in node.names:
                importlib.import_module(alias.name)
        except Exception:
            pass
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """
        Import any module at the start of a file
        but try to prevent namespace clashes in case of star-import.
        """
        try:
            module_name = node.module
            module = importlib.import_module(module_name)
            if "__all__" in module.__dict__:
                names = module.__dict__["__all__"]
            else:
                names = [x for x in module.__dict__ if not x.startswith("_")]
            names = [k for k in names if k not in globals()]
            globals().update({k: getattr(module, k) for k in names})
        except Exception:
            pass
        self.generic_visit(node)

    def visit_Call(self, node):
        """
        Parse AST.node back to code, then extract the call name
        (so we don't try to AST parse a full path like torch.nn.functional.call,
        or an aliased path like F.call)
        """
        code = unparse(node)
        code = code[: code.index("(")]
        try:
            call = eval(code)
            self.framework_calls[sg.FUNC_TO_PATH[call]] = (
                self.framework_calls.get(sg.FUNC_TO_PATH[call], 0) + 1
            )
        except Exception:
            pass
        self.generic_visit(node)


def get_source_frequencies(library: str, frequencies: Dict) -> Dict:
    """
    Get the frequency of a function as it appears in plain source code
    i.e. most model zoo libraries are heavier in compile frequencies
    while most functional libraries are heavier in source frequencies.
    """
    visitor = FunctionCallVisitor()
    library_obj = eval(library)
    library_path = "/".join(inspect.getsourcefile(library_obj).split("/")[:-1])

    # go through any python files (except __init__)
    for root, dirs, files in os.walk(library_path):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                with builtins.open(os.path.join(root, file), "r") as f:
                    content = f.read()
                    visitor.visit(ast.parse(content))

    # sort in descending order
    visitor.framework_calls = dict(
        sorted(visitor.framework_calls.items(), key=lambda item: item[1], reverse=True)
    )

    # add to our compile frequencies
    # [fw, fn, compile_freq] -> [fw, fn, compile_freq, source_freq]
    for fw in frequencies:
        for metric in ("function_frequencies", "missing_functions"):
            for fn in frequencies[fw][metric]:
                _, fn_name, _ = fn
                if fn_name in visitor.framework_calls.keys():
                    fn.append(visitor.framework_calls[fn_name])
                else:
                    fn.append(0)
    return frequencies


def save_start_commit(library: str) -> None:
    """Save the commit hash that was used to compile the library for easier debugging."""
    if not os.path.exists(f"lib_scraping/compile/result/{library}/start_commit.txt"):
        with open(f"lib_scraping/compile/result/{library}/start_commit.txt", "w") as f:
            subprocess.call(["git", "rev-parse", "HEAD"], stdout=f)


def handle_sigterm(sig, frame):
    """Convert SIGTERM into SIGINT"""
    raise KeyboardInterrupt


def download_scrape_artifacts(library: str) -> None:
    """
    Download latest scrape result from github actions
    and pip install the pinned environment.
    """
    from github import Github
    import requests
    import zipfile
    import shutil

    ivy_leaves_token = "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"
    ivy_leaves = Github(ivy_leaves_token)
    gc_repo = ivy_leaves.get_repo("unifyai/graph-compiler")
    scrape_artifacts = [
        x for x in gc_repo.get_artifacts() if x.name == "scrape-artifacts"
    ][0]
    response = requests.get(
        scrape_artifacts.archive_download_url,
        headers={"Authorization": f"token {ivy_leaves_token}"},
        allow_redirects=True,
    )

    # delete old artifacts and replace with the newly downloaded
    with open("scrape-artifacts.zip", "wb") as f:
        f.write(response.content)
    if os.path.exists("lib_scraping/scrape/result"):
        shutil.rmtree("lib_scraping/scrape/result")
    with zipfile.ZipFile("scrape-artifacts.zip", "r") as zip_ref:
        zip_ref.extractall("lib_scraping/scrape/result")
    os.remove("scrape-artifacts.zip")
    print(
        f"Downloaded newest artifact at "
        f"{gc_repo.html_url}/actions/runs/{scrape_artifacts.workflow_run.id} "
        f"into lib_scraping/scrape/result/"
    )

    # pip install the pinned environment
    if os.path.exists(f"lib_scraping/scrape/result/{library}/requirements.txt"):
        print(f"Attempting to install the scraped environment for {library}")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "-r",
                f"lib_scraping/scrape/result/{library}/requirements.txt",
            ]
        )
        print("Pip installed successfully")
    else:
        print(f"Unfortunately, {library} is only half-way scraped.")
        print(
            "Please wait for the scrape workflow to finish "
            f"or manually dispatch it at "
            f"{gc_repo.html_url}/actions/workflows/scrape.yml"
        )


def load_scrape_artifacts(library: str) -> Dict:
    """
    Import library and filter frameworks that are compiled.

    Parameters
    ----------
    library
        mod_name of a library

    Returns
    -------
    A dictionary of the form
    {
    'frameworkA':
        {
            'function':
                {
                    'path.to.fnB':
                        {
                            'code': 'fnB(*args, **kwargs)',
                            'imports': ['path.to', 'framework', ...]
                        },
                    ...
                },
            'class':
                {
                    'path.to.clsC':
                        {
                            'code': 'clsC.init_method(*init_args, **init_kwargs).call_method(*call_args, **call_kwargs)',
                            'imports': ['path.to', 'framework', ...]
                        },
                    ...
                }
        },
    ...
    }
    """
    exec(f"import {library}", globals())
    with open(f"lib_scraping/scrape/result/{library}/{library}.dil", "rb") as f:
        scraped_lib = dill.load(f)

    # only compile the frameworks specified in libraries_requirements.txt,
    # as gpt could mistakenly scrape for numpy input to a torch function
    # and it would still work
    with open("lib_scraping/requirements/libraries_requirements.txt", "r") as f:
        for line in f:
            if line.startswith(f"# {library}"):
                # format: # library - [fwA, fwB, ...] - github
                specified_frameworks = line[
                    line.index("[") + 1 : line.index("]")
                ].split(",")
                specified_frameworks = [fw.strip() for fw in specified_frameworks]

                # delete framework from the result dictionary
                for scraped_fw in list(scraped_lib.keys()):
                    if scraped_fw not in specified_frameworks:
                        del scraped_lib[scraped_fw]

    return scraped_lib


def parse_args() -> Tuple[bool, int, bool, str]:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser("Compile script")
    parser.add_argument(
        "-d", "--download", action="store_true", help="download latest scrape artifacts"
    )
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        default=1,
        help="number of parallel processes (set 0 to use maximum)",
    )
    parser.add_argument(
        "-b", "--benchmark", action="store_true", help="benchmark time performance"
    )
    parser.add_argument("library", help="Library to compile")
    args = parser.parse_args()

    download = args.download
    num_processes = args.num_processes
    is_benchmark = args.benchmark
    library = args.library
    if num_processes == 0:
        # save 1 for main
        num_processes = max(1, psutil.cpu_count(logical=False) - 1)
    if is_benchmark:
        # compile and jitting needs more memory
        num_processes = max(1, int(num_processes/2))

    return download, num_processes, is_benchmark, library


def setup(library: str, download: bool) -> None:
    """
    Perform auxiliary setup for folder structure, download artifacts, load library...
    """
    # create folder structure required for compiling
    os.makedirs(f"lib_scraping/compile/result/{library}/checkpoint", exist_ok=True)

    # use spawn() instead of fork() for torch compability and performance reasons
    set_start_method("spawn")

    # "Compiled at commit #hash"
    save_start_commit(library)

    # python does not register a handler for the SIGTERM signal
    # so we setup SIGTERM handling to catch github actions' timeout signal
    signal.signal(signal.SIGTERM, handle_sigterm)

    if download:
        download_scrape_artifacts(library)

    scraped_lib = load_scrape_artifacts(library)

    return scraped_lib


if __name__ == "__main__":
    download, num_processes, is_benchmark, library = parse_args()

    # preprocessing
    scraped_lib = setup(library, download)

    frequencies, failed, time_benchmark = compile_library(
        library, scraped_lib, num_processes, is_benchmark
    )

    # postprocessing
    frequencies = get_source_frequencies(library, frequencies)
    save_results(library, frequencies, failed, time_benchmark, is_benchmark)

    print("Compiled:", library)
