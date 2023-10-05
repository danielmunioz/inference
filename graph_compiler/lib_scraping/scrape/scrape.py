# global
import inspect
import ast
import subprocess
import sys
import resource
import time
import traceback
import dill
import os
import gc
import psutil
import signal
from multiprocessing import Queue, Process, Manager, set_start_method
from multiprocessing.managers import ListProxy, DictProxy
from typing import Union, List, Tuple, Dict
from types import ModuleType
import builtins
from builtins import *
import argparse
import poe
import ivy
from tqdm.autonotebook import tqdm

# extra settings to help with scraping
# https://github.com/pytorch/pytorch/issues/29893
os.environ["LRU_CACHE_CAPACITY"] = "1"
os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "1"
# help collect callables
resource.setrlimit(resource.RLIMIT_STACK, [0x10000000, resource.RLIM_INFINITY])
sys.setrecursionlimit(0x100000)


# -------- ADD A FRAMEWORK TO SCRAPE FOR HERE --------
import numpy
import numpy as np
import torch
import torch.nn.functional as F
import tensorflow
import tensorflow as tf
import jax
import jax.numpy as jnp
import flax
import haiku
import haiku as hk
import paddle

module_types = {
    "torch": (torch.nn.Module,),
    "tensorflow": (tf.Module,),
    "flax": (flax.linen.Module,),
    "haiku": (haiku.Module, haiku.Transformed),
    # "paddle": (paddle.nn.Layer,),
}
tensor_types = {
    "torch": (torch.Tensor,),
    "tensorflow": (tf.Tensor, tf.Variable),
    "jax": (jax.numpy.ndarray,),
    "numpy": (numpy.ndarray,),
    # "paddle": (paddle.Tensor,),
}
frameworks = list(tensor_types.keys())
# ----------------------------------------------------

torch.set_grad_enabled(False)
bot = "capybara"
context = """You are a deep learning framework expert. 
You are very good at torch, tensorflow, jax,... 
You are given a function or class definition with type hints and docstrings. 
Your task is to generate valid arguments and call that function or initiate that class in one line of code. 
Only generate arguments in the following frameworks: [{}] . 
Do not repeat the definition. 
Do not give any explanation. 
Just call the function or initiate the class with your arguments as a one-liner. 
Pay close attention to the valid shape of the tensor arguments. 
If you can't figure out the value of tensors, use 0 or 1 as example data. 
Only provide arguments for the arguments that don't have a default value. 
If there is an examples section with >>> format, combine the >>> lines into 1 line, then give me the result. 
Do not ask for clarification. 
Remember, your answer must be one line of code, and if it is a class, you must not call the class method, just initiate the class. 
I do not need any library imports. 
Use the following format: ```command```. 
You cannot say anything else. 
------------------------------- 
"""
follow_up = """Congratulations! Calling {} returned a model! 
Now you must generate a valid input for the model. 
Your answer has to follow this format: ```{}(tensors)```. 
Use random values as data. 
I don't know the shape of the input tensor, 
so I suggest using random values of shape 
(1, 3, 224, 224) for computer vision models, 
(1, 512) for natural language processing models or 
(1, 10, 10) for tabular models. 
If the shape is wrong, you'll have to correct it 
by using the traceback I give in subsequent prompts. 
Do not give any explanation. 
I don't need any library imports. 
Your answer must be a one-liner. 
Do not say anything else.
"""
reminder = """Give me another call. Do not apologize. 
Remember, your answer must be one line of code. 
I do not need any library imports. 
Use the following format: ```{}```. 
You cannot say anything else.
"""


# ------- Temporary until this PR gets merged
# https://github.com/unifyai/ivy/pull/13904
def nested_any(
    nest,
    fn,
    check_nests=False,
    check_attributes=False,
    _base=True,
    extra_nest_types=None,
) -> bool:
    extra_nest_types = extra_nest_types or builtins.tuple()
    if builtins.isinstance(
        nest, (builtins.tuple, builtins.list)
    ) or builtins.isinstance(nest, extra_nest_types):
        if fn(nest):
            return True
        for i, item in builtins.enumerate(nest):
            if nested_any(
                item, fn, check_nests, check_attributes, False, extra_nest_types
            ):
                return True
        if check_nests and fn(nest):
            return True
    elif builtins.isinstance(nest, builtins.dict):
        for k, v in nest.items():
            if nested_any(
                v, fn, check_nests, check_attributes, False, extra_nest_types
            ):
                return True
        if check_nests and fn(nest):
            return True
    elif fn(nest):
        return True
    elif (
        check_attributes
        and builtins.hasattr(nest, "__dict__")
        and not builtins.hasattr(nest, "__array__")
    ):
        if nested_any(
            builtins.list(nest.__dict__.values()),
            fn,
            check_nests,
            check_attributes,
            False,
            extra_nest_types,
        ):
            return True
    return False


# --------------------


def is_native_module(obj: object) -> bool:
    # check if a class is inheritted from a Model/Module class in native framework
    module_types_flattened = [
        module_type for framework in module_types.values() for module_type in framework
    ]
    return builtins.any([x in module_types_flattened for x in obj.__mro__])


def collect_callables(module: ModuleType) -> None:
    global callables, processed_modules, other_modules
    if module in processed_modules:
        return
    processed_modules.add(module)

    if "." in module.__name__:
        module_root = ".".join(module.__name__.split(".")[:-1])
    else:
        module_root = module.__name__

    for thing in dir(module):
        try:
            obj = getattr(module, thing)
        except Exception:
            continue

        # if a module has __all__, import their __all__ only
        try:
            if (
                not inspect.ismodule(obj)
                and hasattr(module, "__all__")
                and thing not in getattr(module, "__all__")
            ):
                continue
        except Exception:
            continue

        # prevents going to torch, tensorflow etc...
        # prevents importing `_` prefixed objects
        if (
            callable(obj)
            and hasattr(obj, "__module__")
            and obj.__module__ is not None
            and obj.__module__.startswith(module_root)
            and hasattr(obj, "__name__")
            and obj.__name__ is not None
            and not obj.__name__.startswith("_")
        ):
            if inspect.isclass(obj) and is_native_module(obj):
                callables.add(f"{obj.__module__}.{obj.__name__}")
            elif inspect.isfunction(obj):
                callables.add(f"{obj.__module__}.{obj.__name__}")
        elif inspect.ismodule(obj):
            if obj.__name__.startswith(module_root):
                collect_callables(obj)
            else:
                other_modules.add(obj)


def exists_collect_callables(library: ModuleType, use_cached: bool = False) -> None:
    # if we already collected once, then load
    global callables, processed_modules, other_modules
    callables_path = "lib_scraping/scrape/result/{}/callables.dil".format(
        library.__name__
    )
    if use_cached and os.path.exists(callables_path):
        with builtins.open(callables_path, "rb") as f:
            callables = dill.load(f)
        callables = list(callables)
    else:
        collect_callables(library)
        # convert to list so order is preserved for subsequent scrapes,
        # makes debugging easier
        callables = list(callables)
        os.makedirs(os.path.dirname(callables_path), exist_ok=True)
        with builtins.open(callables_path, "wb") as f:
            dill.dump(callables, f, protocol=dill.HIGHEST_PROTOCOL)
    print("Num callables of {}: {}".format(library.__name__, len(callables)))


def compile_args_kwargs(arg: ast.AST) -> object:
    expr = ast.Expression(body=arg)
    ast.fix_missing_locations(expr)
    return builtins.eval(builtins.compile(expr, filename="", mode="eval"))


def parse_function_call(
    func_str: str, result: object, module: str
) -> Tuple[str, Dict[str, Union[str, List[str]]]]:
    # in the form of func(args, kwargs)
    # parse the function call string into an AST node
    call = ast.parse(func_str).body[-1].value

    args = [compile_args_kwargs(arg) for arg in call.args]
    kwargs = {kwarg.arg: compile_args_kwargs(kwarg.value) for kwarg in call.keywords}

    # gpt's answer may not have the accompanying module
    imports = [module]
    ivy.nested_map(
        args,
        lambda x: imports.append(x.__module__) if hasattr(x, "__module__") else None,
        shallow=False,
    )
    ivy.nested_map(
        kwargs,
        lambda x: imports.append(x.__module__) if hasattr(x, "__module__") else None,
        shallow=False,
    )
    imports = [x for x in imports if x is not None]

    framework = None
    try:
        for native_framework, native_tensor_type in tensor_types.items():
            if (
                nested_any(
                    args,
                    lambda x: builtins.isinstance(x, native_tensor_type),
                    check_attributes=True,
                )
                or nested_any(
                    kwargs,
                    lambda x: builtins.isinstance(x, native_tensor_type),
                    check_attributes=True,
                )
            ) and nested_any(
                result,
                lambda x: builtins.isinstance(x, native_tensor_type),
                check_attributes=True,
            ):
                framework = native_framework
                break
    except Exception as e:
        print(repr(e))
        pass

    return framework, {"code": func_str, "imports": imports}


def parse_class_call(
    class_str: str, result: object, module: str
) -> Tuple[str, Dict[str, Union[str, List[str]]]]:
    # in the form of
    # class.init_method(init_args, init_kwargs).call_method(call_args, call_kwargs)
    # parse the class call string into an AST node
    call = ast.parse(class_str).body[-1].value

    try:
        # class.init_method().call_method()
        init = call.func.value
    except Exception:
        # class.init_method()()
        init = call.func

    call_args = [compile_args_kwargs(arg) for arg in call.args]
    call_kwargs = {
        kwarg.arg: compile_args_kwargs(kwarg.value) for kwarg in call.keywords
    }
    init_args = [compile_args_kwargs(arg) for arg in init.args]
    init_kwargs = {
        kwarg.arg: compile_args_kwargs(kwarg.value) for kwarg in init.keywords
    }

    imports = [module]
    ivy.nested_map(
        init_args,
        lambda x: imports.append(x.__module__) if hasattr(x, "__module__") else None,
        shallow=False,
    )
    ivy.nested_map(
        call_args,
        lambda x: imports.append(x.__module__) if hasattr(x, "__module__") else None,
        shallow=False,
    )
    ivy.nested_map(
        init_kwargs,
        lambda x: imports.append(x.__module__) if hasattr(x, "__module__") else None,
        shallow=False,
    )
    ivy.nested_map(
        call_kwargs,
        lambda x: imports.append(x.__module__) if hasattr(x, "__module__") else None,
        shallow=False,
    )
    imports = [x for x in imports if x is not None]

    framework = None
    try:
        for native_framework, native_tensor_type in tensor_types.items():
            if (
                nested_any(
                    call_args,
                    lambda x: builtins.isinstance(x, native_tensor_type),
                    check_attributes=True,
                )
                or nested_any(
                    call_kwargs,
                    lambda x: builtins.isinstance(x, native_tensor_type),
                    check_attributes=True,
                )
            ) and nested_any(
                result,
                lambda x: builtins.isinstance(x, native_tensor_type),
                check_attributes=True,
            ):
                framework = native_framework
                break
    except Exception:
        pass

    return framework, {"code": class_str, "imports": imports}


def _to_multiprocessing_friendly_and_back(
    good: Union[List[str], ListProxy],
    bad: Union[List[str], ListProxy],
    callables_with_args_kwargs: Union[Dict, DictProxy],
    callables_with_args_kwargs_non_tensor: Union[Dict, DictProxy],
    manager: Manager,
    to_multiprocessing: bool,
) -> Tuple[
    Union[ListProxy, List[str]],
    Union[ListProxy, List[str]],
    Union[DictProxy, Dict],
    Union[DictProxy, Dict],
]:
    if to_multiprocessing:
        convert_list = manager.list
        convert_dict = manager.dict
    else:
        convert_list = list
        convert_dict = dict
    good = convert_list(good)
    bad = convert_list(bad)
    callables_with_args_kwargs = convert_dict(callables_with_args_kwargs)
    for fw in callables_with_args_kwargs:
        callables_with_args_kwargs[fw] = convert_dict(callables_with_args_kwargs[fw])
        for func_cls in ("function", "class"):
            callables_with_args_kwargs[fw][func_cls] = convert_dict(
                callables_with_args_kwargs[fw][func_cls]
            )
    callables_with_args_kwargs_non_tensor = convert_dict(
        callables_with_args_kwargs_non_tensor
    )
    for func_cls in ("function", "class"):
        callables_with_args_kwargs_non_tensor[func_cls] = convert_dict(
            callables_with_args_kwargs_non_tensor[func_cls]
        )
    return good, bad, callables_with_args_kwargs, callables_with_args_kwargs_non_tensor


def gen_args_kwargs_gpt3(
    callables: List[str],
    token_queue: Queue,
    process: int,
    good: ListProxy,
    bad: ListProxy,
    callables_with_args_kwargs: DictProxy,
    callables_with_args_kwargs_non_tensor: DictProxy,
    time_counter: ListProxy,
    lib_name: str,
    debug_print: bool,
) -> None:
    def process_print(*msg):
        builtins.print(process, *msg, flush=True)
        # tqdm.write(" ".join([str(process)] + [str(m) for m in msg]))

    client = poe.Client(token_queue.get(timeout=600))
    callables = [
        x for x in callables if x not in builtins.set(good).union(builtins.set(bad))
    ]
    pbar = tqdm(
        callables,
        desc="Process {}".format(process),
        dynamic_ncols=True,
        position=process + 1,
        leave=False,
    )
    globals_copy = globals().copy()
    for i, func_str in builtins.enumerate(callables):
        gc.collect()
        time_counter[process] = time.time()
        # preemptively mark the function as bad, then add it to good later
        # in case we have to deal with giant models (process is sigkill-ed immediately)
        # then we skip the model when we revive the process
        if func_str not in bad:
            bad.append(func_str)
        if process == 0 and i != 0 and i % 5 == 0:  # checkpointing
            with builtins.open(
                "lib_scraping/scrape/result/{}/temp.dil".format(lib_name), "wb"
            ) as f:
                try:
                    # convert to regular list, dict
                    dill.dump(
                        _to_multiprocessing_friendly_and_back(
                            good,
                            bad,
                            callables_with_args_kwargs,
                            callables_with_args_kwargs_non_tensor,
                            manager,
                            to_multiprocessing=False,
                        ),
                        f,
                        protocol=dill.HIGHEST_PROTOCOL,
                    )
                except Exception:
                    # error: changed size while saving due to multiprocessing
                    pass

        try:
            if "." in func_str:
                module = ".".join(func_str.split(".")[:-1])
                # import everything from the module of our function
                # so that gpt doesn't have to care about namespaces
                import_command = "from {} import *".format(module)
                if debug_print:
                    process_print(import_command)
                builtins.exec(import_command, globals())
                func = builtins.eval(func_str.split(".")[-1])
            else:
                module = None
                func = builtins.eval(func_str)
        except Exception as e:
            process_print("Weird error that should've been caught", builtins.repr(e))
            continue

        # get definition in source code to prompt
        try:
            code = inspect.getsource(func)
        except Exception:
            process_print("Found error: Cant find source code for", func.__name__)
            continue

        # we could use tiktoken to accurately get token length but yea slow.
        # we have to make space for subsequent debug prompts, so
        # prefer init -> forward -> docstring
        token_limit = 6000
        if len(code) > token_limit and inspect.isclass(func):
            code = "class " + func.__name__
            # init
            try:
                code += "\n\n" + inspect.getsource(func.__init__)
            except Exception:
                pass

            # forward
            for forward_method in ("forward", "call", "__call__"):
                try:
                    code += "\n\n" + inspect.getsource(
                        builtins.getattr(func, forward_method)
                    )
                    break
                except Exception:
                    pass

            # docstring
            try:
                code += "\n\n" + inspect.getdoc(func)
            except Exception:
                pass

        prompt = context + code
        if debug_print:
            process_print(code)
        # TODO: wrap this into a decorator
        try:
            client.send_chat_break(bot)
        except RuntimeError:
            client = poe.Client(token_queue.get(timeout=600))

        for attempt in builtins.range(10):
            time.sleep(5)
            time_counter[process] = time.time()
            ee = False
            try:
                for chunk in client.send_message(
                    bot, prompt[:token_limit], with_chat_break=False
                ):
                    pass
            except RuntimeError:
                client = poe.Client(token_queue.get(timeout=600))
                ee = True
            except Exception as e:
                process_print(builtins.repr(e))
                ee = True
            if ee:
                process_print("Retrying sending message")
                continue

            # parse reply
            reply = chunk["text"]
            if "#" in reply:
                reply = reply[: reply.index("#")]
            reply = reply.strip("` \n")
            process_print("Answer: {}".format(reply))

            # check if reply is not gibberish
            if not reply.startswith(func.__name__):
                if func.__name__ in reply:
                    reply = reply[reply.rindex(func.__name__) :]
                else:
                    process_print("Found error: Original is", func.__name__)
                    prompt = f"""Your answer has to start with {func.__name__}. 
                       {reminder.format("command")}
                    """
                    continue
            if ")(" in reply:
                reply = reply[: reply.index(")(") + 1]
            if ")." in reply:
                reply = reply[: reply.index(").") + 1]
            if not reply.endswith(")"):
                process_print("Found error: Doesn't end with a call")
                prompt = f"""Your answer must be a single call of {func.__name__}. 
                     {reminder.format("command")}
                  """
                continue

            # try to run the reply
            try:
                result = builtins.eval(reply)
            except Exception as e:
                process_print("Found error:", builtins.repr(e))
                tb = "\n".join(traceback.format_exc().splitlines()[2:])
                prompt = f"""Running this found an error. The traceback is:{{{tb}}}.
                    {reminder.format("command")}
                  """
                if attempt >= 5:
                    # initially tried with .zeros and .ones, now try with .rand
                    # or encourage chatgpt to steal code from a random git repo :)
                    prompt += """ If you can't figure out the value of tensors, 
                        try random values as example data."""
                continue

            # if calling the given function or class returns a callable
            # we can assume it's a model, and try to pass tensor inputs to it
            if builtins.callable(result):
                prompt = follow_up.format(reply, reply, reply)
                for calling_attempt in builtins.range(10 - attempt):
                    time.sleep(5)
                    time_counter[process] = time.time()
                    # TODO: this is repeated but with small modifications
                    # maybe I can group into a function
                    try:
                        for chunk in client.send_message(
                            bot, prompt[:token_limit], with_chat_break=False
                        ):
                            pass
                    except RuntimeError:
                        client = poe.Client(token_queue.get(timeout=600))
                        ee = True
                    except Exception as e:
                        print(builtins.repr(e))
                        ee = True
                    if ee:
                        process_print("Retrying sending message")
                        continue

                    # parse reply
                    reply = chunk["text"]
                    if "#" in reply:
                        reply = reply[: reply.index("#")]
                    reply = reply.strip("` \n")
                    process_print(
                        "Answer in calling the model returned by function: {}".format(
                            reply
                        )
                    )

                    # check if reply is not gibberish
                    if not reply.startswith(func.__name__):
                        if func.__name__ in reply:
                            reply = reply[reply.rindex(func.__name__) :]
                        else:
                            process_print("Found error: Original is", func.__name__)
                            prompt = f"""Your answer has to start with {func.__name__}. 
                          {reminder.format("command(tensors)")}
                        """
                            continue

                    # try to run the call
                    try:
                        result = builtins.eval(reply)
                    except Exception as e:
                        process_print("Found error:", builtins.repr(e))
                        tb = "\n".join(traceback.format_exc().splitlines()[2:])
                        prompt = f"""Running this found an error. The traceback is:{{{tb}}}. 
                         You must try another tensor shape or batch size. 
                         {reminder.format("command(tensors)")}
                      """
                        continue

                    # parse the reply with ast
                    try:
                        framework, code_imports = parse_class_call(
                            reply, result, module
                        )
                    except Exception as e:
                        process_print(builtins.repr(e))
                        # we urge gpt to include at least a tensor in the reply
                        process_print("Found error: No tensor in input")
                        prompt = f"""You have to call with a tensor input! 
                         {reminder.format("command(tensors)")}
                      """
                        continue
                    if framework:
                        callables_with_args_kwargs[framework]["class"][
                            func_str
                        ] = code_imports
                    else:
                        callables_with_args_kwargs_non_tensor["class"][
                            func_str
                        ] = code_imports
                    break
                else:
                    process_print("-" * 7, func_str)
                    break
            else:
                # parse the reply with ast
                try:
                    framework, code_imports = parse_function_call(reply, result, module)
                except Exception as e:
                    process_print(builtins.repr(e))
                    # we urge gpt to include at least a tensor in the reply
                    process_print("Found error: No tensor in input")
                    prompt = f"""You have to call with a tensor input! 
                         {reminder.format("command(tensors)")}
                      """
                    continue
                if framework:
                    callables_with_args_kwargs[framework]["function"][
                        func_str
                    ] = code_imports
                else:
                    callables_with_args_kwargs_non_tensor["function"][
                        func_str
                    ] = code_imports

            if func_str not in good:
                good.append(func_str)
            if func_str in bad:
                bad.remove(func_str)
            process_print("+" * 7, func_str)
            break
        else:
            process_print("-" * 7, func_str)
        pbar.update(1)
    globals().update(globals_copy)
    time_counter[process] = None
    pbar.close()


def import_subpackages(library: ModuleType) -> None:
    subpackages = os.listdir("/".join(inspect.getsourcefile(library).split("/")[:-1]))
    subpackages = [
        package[: package.index(".py")] if ".py" in package else package
        for package in subpackages
    ]
    # prevent self-importing __init__ or other non-regular packages
    subpackages = [package for package in subpackages if not package.startswith("_")]
    for package in subpackages:
        try:
            exec(f"import {library.__name__}.{package}", globals())
        except Exception:
            pass


def handle_sigterm(sig, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser("Scrape script")
    parser.add_argument("--debug", action="store_true", help="print gpt queries")
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        default=1,
        help="number of parallel processes (set 0 to use maximum)",
    )
    parser.add_argument("library", help="Library to scrape for")
    args = parser.parse_args()
    debug_print = args.debug
    num_processes = args.num_processes
    library = args.library
    if num_processes == 0:
        # we are mostly IO bound, so logical cores are fine
        # save 1 for main, 1 for token
        num_processes = max(1, psutil.cpu_count(logical=True) - 2)
    os.makedirs("lib_scraping/scrape/result/{}".format(library), exist_ok=True)

    # only scrape specified frameworks
    if os.path.exists(f"lib_scraping/requirements/requirements_{library}.txt"):
        requirements_file = f"lib_scraping/requirements/requirements_{library}.txt"
    else:
        requirements_file = "lib_scraping/requirements/libraries_requirements.txt"
    with open(requirements_file, "r") as f:
        # format: # library - [fwA, fwB, ...] - github
        for line in f:
            if line.startswith(f"# {library}"):
                specified_frameworks = line[line.index("[") + 1 : line.index("]")]
                context = context.format(specified_frameworks)
                frameworks = [fw for fw in frameworks if fw in specified_frameworks]

    # get gpt tokens in another process
    set_start_method("spawn")
    from get_poe_tokens import get_tokens

    subprocess.run(["pkill", "chrome"])
    processes = []
    token_queue = Queue(maxsize=1)
    Process(target=get_tokens, args=(token_queue,), daemon=True).start()

    # import the libraries (and maybe it's hidden subpackages, i.e. fastai)
    exec(f"import {library}", globals())
    library = eval(library)
    import_subpackages(library)

    # collect functions and classes
    callables = set()  # all callables
    other_modules = set()  # all modules found that are not in lib
    processed_modules = set()  # avoid recursive imports
    exists_collect_callables(library, use_cached=False)

    manager = Manager()
    time_counter = manager.list()
    # continue from checkpoint / continue from previous scrape / first time
    if os.path.exists(
        "lib_scraping/scrape/result/{}/temp.dil".format(library.__name__)
    ):
        # load from checkpoint
        with builtins.open(
            "lib_scraping/scrape/result/{}/temp.dil".format(library.__name__), "rb"
        ) as f:
            (
                good,
                bad,
                callables_with_args_kwargs,
                callables_with_args_kwargs_non_tensor,
            ) = dill.load(f)
        callables = [x for x in callables if x not in set(good).union(set(bad))]
        # convert to multiprocessing friendly
        (
            good,
            bad,
            callables_with_args_kwargs,
            callables_with_args_kwargs_non_tensor,
        ) = _to_multiprocessing_friendly_and_back(
            good,
            bad,
            callables_with_args_kwargs,
            callables_with_args_kwargs_non_tensor,
            manager,
            to_multiprocessing=True,
        )
    elif os.path.exists(
        "lib_scraping/scrape/result/{}/{}.dil".format(
            library.__name__, library.__name__
        )
    ):
        # load from previous scrape
        # good is always appended, as we reuse good calls
        # but bad should always be empty here, as we try to redo bad calls
        good = list()
        bad = list()
        with builtins.open(
            "lib_scraping/scrape/result/{}/{}.dil".format(
                library.__name__, library.__name__
            ),
            "rb",
        ) as f:
            callables_with_args_kwargs = dill.load(f)
        with builtins.open(
            "lib_scraping/scrape/result/{}/{}_non_tensor.dil".format(
                library.__name__, library.__name__
            ),
            "rb",
        ) as f:
            callables_with_args_kwargs_non_tensor = dill.load(f)
        for fw in callables_with_args_kwargs:
            for func_cls in ("function", "class"):
                for full_name, code_imports in list(
                    callables_with_args_kwargs[fw][func_cls].items()
                ):
                    code, imports = code_imports["code"], code_imports["imports"]
                    globals_copy = globals().copy()
                    try:
                        # import other libs first (torch, numpy...)
                        for lib in imports:
                            if not lib.startswith(library.__name__):
                                builtins.exec(
                                    f"from {lib} import *", builtins.globals()
                                )
                        # import the target library last to overwrite
                        for lib in imports:
                            if lib.startswith(library.__name__):
                                builtins.exec(
                                    f"from {lib} import *", builtins.globals()
                                )
                        builtins.exec(code)
                        if full_name in callables:
                            good.append(full_name)
                    except Exception:
                        callables_with_args_kwargs[fw][func_cls].pop(full_name)
                    globals().update(globals_copy)
        for func_cls in ("function", "class"):
            for full_name, code_imports in list(
                callables_with_args_kwargs_non_tensor[func_cls].items()
            ):
                code, imports = code_imports["code"], code_imports["imports"]
                globals_copy = globals().copy()
                try:
                    # import other libs first (torch, numpy...)
                    for lib in imports:
                        if not lib.startswith(library.__name__):
                            builtins.exec(f"from {lib} import *", builtins.globals())
                    # import the target library last to overwrite
                    for lib in imports:
                        if lib.startswith(library.__name__):
                            builtins.exec(f"from {lib} import *", builtins.globals())
                    builtins.exec(code)
                    if full_name in callables:
                        good.append(full_name)
                except Exception:
                    callables_with_args_kwargs_non_tensor[func_cls].pop(full_name)
                globals().update(globals_copy)
        callables = [x for x in callables if x not in set(good)]
        print(f"Reused {len(good)} calls from the previous scrape")
        # convert to multiprocessing friendly
        (
            good,
            bad,
            callables_with_args_kwargs,
            callables_with_args_kwargs_non_tensor,
        ) = _to_multiprocessing_friendly_and_back(
            good,
            bad,
            callables_with_args_kwargs,
            callables_with_args_kwargs_non_tensor,
            manager,
            to_multiprocessing=True,
        )
    else:
        # first time
        # init as multiprocessing friendly objects
        good = manager.list()  # we nail those
        bad = manager.list()  # we fail those
        callables_with_args_kwargs = manager.dict()
        for fw in frameworks:
            callables_with_args_kwargs[fw] = manager.dict()
            for func_cls in ("function", "class"):
                callables_with_args_kwargs[fw][func_cls] = manager.dict()
        callables_with_args_kwargs_non_tensor = manager.dict()
        for func_cls in ("function", "class"):
            callables_with_args_kwargs_non_tensor[func_cls] = manager.dict()

    # TODO: delete this
    for x in good[:]:
        if "__init__" in x:
            good.remove(x)
    for x in bad[:]:
        if "__init__" in x:
            bad.remove(x)
    for fw in callables_with_args_kwargs:
        for func_cls in ("function", "class"):
            for x in list(callables_with_args_kwargs[fw][func_cls].keys()):
                if "__init__" in x:
                    callables_with_args_kwargs[fw][func_cls].pop(x)
    for func_cls in ("function", "class"):
        for x in list(callables_with_args_kwargs_non_tensor[func_cls].keys()):
            if "__init__" in x:
                callables_with_args_kwargs_non_tensor[func_cls].pop(x)

    # mental check
    print("good calls:", len(set(good)))
    print("bad calls:", len(set(bad)))

    for i in range(num_processes):
        # split callables into processes
        sublist_size = len(callables) // num_processes
        start_index = i * sublist_size
        end_index = (i + 1) * sublist_size
        if i == num_processes - 1:
            end_index = len(callables)
        c = callables[start_index:end_index]
        time_counter.append(None)
        p = Process(
            target=gen_args_kwargs_gpt3,
            args=(
                c,
                token_queue,
                i,
                good,
                bad,
                callables_with_args_kwargs,
                callables_with_args_kwargs_non_tensor,
                time_counter,
                library.__name__,
                debug_print,
            ),
            daemon=True,
        )
        processes.append((p, c))
        p.start()

    stuck_timeout = 300
    signal.signal(signal.SIGTERM, handle_sigterm)
    try:
        # exitcode is:
        # - None when not completed
        # - 0 when completed
        # - >0 when failed as POSIX exitcode
        # - <0 when failed as negative POSIX exitcode for subprocesses
        while any([p.exitcode != 0 for p, _ in processes]):
            # keep polling for the processes' exit code.
            # if one died with a non-zero exit code,
            # create a new one in its place.
            time.sleep(5)
            for i in range(len(processes)):
                p, c = processes[i]
                # check if process died (most probably OOM)
                # check if a function got stuck
                if (p.exitcode is not None and p.exitcode != 0) or (
                    time_counter[i] is not None
                    and time.time() - time_counter[i] > stuck_timeout
                ):
                    if p.exitcode is not None:
                        print(
                            f"Detect process {i} died "
                            f"with exit code {p.exitcode}. Restarting..."
                        )
                    else:
                        print(
                            f"Detect process {i} hanging "
                            f"for more than {stuck_timeout} seconds. Restarting..."
                        )
                        p.kill()
                    time_counter[i] = None
                    new_p = Process(
                        target=gen_args_kwargs_gpt3,
                        args=(
                            c,
                            token_queue,
                            i,
                            good,
                            bad,
                            callables_with_args_kwargs,
                            callables_with_args_kwargs_non_tensor,
                            time_counter,
                            library.__name__,
                            debug_print,
                        ),
                        daemon=True,
                    )
                    processes[i] = (new_p, c)
                    new_p.start()

        # ignore shutdown request
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        gc.collect()
        # converting back to regular dict when finished
        (
            good,
            bad,
            callables_with_args_kwargs,
            callables_with_args_kwargs_non_tensor,
        ) = _to_multiprocessing_friendly_and_back(
            good,
            bad,
            callables_with_args_kwargs,
            callables_with_args_kwargs_non_tensor,
            manager,
            to_multiprocessing=False,
        )

        # save result
        with builtins.open(
            "lib_scraping/scrape/result/{}/{}.dil".format(
                library.__name__, library.__name__
            ),
            "wb",
        ) as f:
            dill.dump(callables_with_args_kwargs, f, protocol=dill.HIGHEST_PROTOCOL)
        with builtins.open(
            "lib_scraping/scrape/result/{}/{}_non_tensor.dil".format(
                library.__name__, library.__name__
            ),
            "wb",
        ) as f:
            dill.dump(
                callables_with_args_kwargs_non_tensor,
                f,
                protocol=dill.HIGHEST_PROTOCOL,
            )

        # remove temp files
        try:
            os.remove(
                "lib_scraping/scrape/result/{}/callables.dil".format(library.__name__)
            )
        except Exception:
            pass
        try:
            os.remove("lib_scraping/scrape/result/{}/temp.dil".format(library.__name__))
        except Exception:
            pass

        # save current environment for compiling
        full_requirements = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "-r", requirements_file],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        requirements = []
        for line in full_requirements.split("\n"):
            if line.startswith("##"):
                break
            requirements.append(line)
        with builtins.open(
            "lib_scraping/scrape/result/{}/requirements.txt".format(library.__name__),
            "w",
        ) as f:
            f.write("\n".join(requirements))

        print("Scraped:", library.__name__)
    except KeyboardInterrupt:
        # you can stop this, it automatically saves to a checkpoint
        print("Detect interrupt, terminating processes and saving to checkpoint...")
        gc.collect()
        for p, _ in processes:
            # don't want to .kill() as they could be saving to checkpoint
            p.terminate()
        time.sleep(5)
        with builtins.open(
            "lib_scraping/scrape/result/{}/temp.dil".format(library.__name__), "wb"
        ) as f:
            dill.dump(
                _to_multiprocessing_friendly_and_back(
                    good,
                    bad,
                    callables_with_args_kwargs,
                    callables_with_args_kwargs_non_tensor,
                    manager,
                    to_multiprocessing=False,
                ),
                f,
                protocol=dill.HIGHEST_PROTOCOL,
            )
        print("Finished cleaning up")
    finally:
        subprocess.run(["pkill", "chrome"])
