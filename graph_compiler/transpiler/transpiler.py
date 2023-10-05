from typing import Callable, Optional, Union, Tuple, Any
from types import ModuleType
import importlib
import gc
import copy

import ivy
from module import module
import graph_compiler.globals as glob
from graph_compiler import compile
from graph_compiler.conversion import (
    _dtype_and_dev_to_ivy,
    native_array_to_frontend,
    _to_ND,
    _from_ND,
)
from graph_compiler.graph import Graph, LazyGraph
from graph_compiler.helpers import (
    _find_missing_frontends,
    _format_missing_frontends_msg,
    _apply_fn_to_module,
)
import graph_compiler.tracked_var_proxy as tvp
from graph_compiler.wrapping import FUNC_TO_PATH

# Helpers #
# ------- #


def _is_submodule(obj, kw):
    cls_str = {
        "torch": "torch.nn.modules.module.Module",
        "keras": "keras.engine.training.Model",
        "haiku": "haiku._src.transform.Transformed",
        "paddle": "paddle.fluid.dygraph.layers.Layer",
        "flax": "flax.linen.module.Module",
    }[kw]
    try:
        for bc in type(obj).mro():
            if cls_str in str(bc):
                return True
    except TypeError:
        pass
    return False


def _native_fn_to_frontend(fn):
    if hasattr(fn, "wrapped_for_compiling"):
        native_path = FUNC_TO_PATH[fn.__wrapped__]
        frontend_path = "ivy.functional.frontends." + native_path
        mod_name, fn_name = frontend_path.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        frontend_fn = getattr(mod, fn_name)
        return frontend_fn
    return fn

    # Transpiling #
    # ----------- #


def _to_target_framework(
    obj,
    *args: Any,
    source: Optional[str] = None,
    to: Optional[str] = None,
    with_numpy: bool = False,
    **kwargs: Any,
) -> Graph:
    """
    Converts the function to be transpiled into a graph of functions
    from the target framework. The target may be ivy or one of ivy's
    supported backends.
    """
    if glob.check_frontend_vs_original_differences:
        glob.original_results = []
        glob.frontend_results = []
    ivy.set_backend(source)
    graph = compile(obj, args=args, kwargs=kwargs, with_numpy=with_numpy)
    if glob.check_frontend_vs_original_differences:
        graph(*args, **kwargs)  # run original graph to store result for each node
    kwargs["source"] = source
    kwargs["target"] = to
    missing_frontends = _find_missing_frontends(graph)
    if missing_frontends:
        msg = _format_missing_frontends_msg(missing_frontends)
        raise ivy.exceptions.IvyException(msg)
    # ToDo: this can be optimized if:
    # - No need to reload constants as they will be reloaded below
    graph.reload_sourcecode(frontend=source)
    using_reloader = glob.use_reloader
    glob.use_reloader = False
    graph.constants = ivy.nested_map(
        graph.constants,
        _dtype_and_dev_to_ivy,
        to_ignore=tvp._to_ignore,
    )
    graph.constants = ivy.nested_map(
        graph.constants,
        _native_fn_to_frontend,
        to_ignore=tvp._to_ignore,
    )
    if to != "ivy":
        ivy.set_backend(to)
    args = ivy.nested_map(
        args,
        native_array_to_frontend,
        include_derived={dict: True},
        to_ignore=tvp._to_ignore,
        shallow=False,
    )
    kwargs = ivy.nested_map(
        kwargs,
        native_array_to_frontend,
        include_derived={dict: True},
        to_ignore=tvp._to_ignore,
        shallow=False,
    )
    graph.constants = ivy.nested_map(
        graph.constants,
        native_array_to_frontend,
        to_ignore=tvp._to_ignore,
    )
    if with_numpy or ivy.current_backend_str == "numpy":
        graph.constants = ivy.nested_map(
            graph.constants,
            _to_ND,
            to_ignore=tvp._to_ignore,
        )
    if to == "ivy":
        graph = compile(graph, to="ivy", args=args, kwargs=kwargs)
    else:
        graph = compile(graph, args=args, kwargs=kwargs)
        glob.use_reloader = using_reloader
        del graph._fn
        gc.collect()
        ivy.previous_backend()
    if with_numpy or ivy.current_backend_str == "numpy":
        graph.constants = ivy.nested_map(
            graph.constants,
            _from_ND,
            to_ignore=tvp._to_ignore,
        )
    ivy.previous_backend()
    return graph


# ToDo: Add module transpilation docs
def transpile(
    *objs: Callable,
    source: Optional[str] = None,
    to: Optional[str] = None,
    with_numpy: bool = True,
    args: Optional[Tuple] = None,
    kwargs: Optional[dict] = None,
    params_v=None,
    v=None,  # Make this cleaner
) -> Union[Graph, LazyGraph]:
    """Transpiles Callable objects passed as arguments.
    If args and kwargs are specified, transpilation is performed eagerly,
    otherwise, transpilation will happen lazily.

    Parameters
    ----------
    objs
        The native Callables to be transpiled
    source
        The framework that `obj` is from.
    to
        The target framework to transpile `obj` to.
    args
        If specified, arguments that will be used to transpile eagerly.
    kwargs
        If specified, keyword arguments that will be used to transpile eagerly.

    Returns
    -------
    Either a transpiled Graph or a non-initialized LazyGraph.
    """
    if source not in [
        "torch",
        "jax",
        "tensorflow",
        "numpy",
        "paddle",
        "flax",
        "haiku",
        "keras",
        None,
    ]:
        raise ivy.exceptions.IvyException(
            "source must be one of 'torch', 'jax', 'tensorflow', 'numpy', "
            "'paddle', 'flax', 'haiku' or 'keras'."
        )

    if to not in [
        "ivy",
        "torch",
        "jax",
        "tensorflow",
        "numpy",
        "paddle",
        "flax",
        "haiku",
        "keras",
    ]:
        raise ivy.exceptions.IvyException(
            "to must be one of 'ivy', 'torch', 'jax', 'tensorflow', 'numpy', "
            "'paddle', 'flax', 'haiku' or 'keras'."
        )

    source_mod = None
    to_mod = None
    if source in ["flax", "haiku"]:
        source_mod = source
        source = "jax"
    elif source == "keras":
        source = "tensorflow"

    if to in ["flax", "haiku", "jax"]:
        to_mod = to
        if to_mod == "jax":
            if importlib.util.find_spec("haiku"):
                to_mod = "haiku"
            elif importlib.util.find_spec("flax"):
                to_mod = "flax"
            else:
                raise ModuleNotFoundError("Couldn't find haiku or flax installed in the system !")
        to = "jax"
    elif to == "keras":
        to = "tensorflow"

    _transpile_kwargs = {
        "source": source,
        "to": to,
        "with_numpy": with_numpy,
    }
    # this is being used as a decorator, only if there are no positional args
    if len(objs) == 0:

        def decorator(func):
            return transpile(
                func,
                args=args,
                kwargs=kwargs,
                **_transpile_kwargs,
            )

        return decorator

    if len(objs) > 1:
        return tuple(
            transpile(
                o,
                args=args,
                kwargs=kwargs,
                **_transpile_kwargs,
            )
            for o in objs
        )

    obj = objs[0]

    # check if fn is a module or a function
    if isinstance(obj, ModuleType):
        return _apply_fn_to_module(
            obj, fn=transpile, source=source, to=to, with_numpy=with_numpy
        )

    is_trainable_module = False
    if source == "torch" or _is_submodule(obj, "torch"):
        import torch

        if isinstance(obj, torch.nn.Module):
            is_trainable_module = True
            source = "torch"
    elif source == "tensorflow" or _is_submodule(obj, "keras"):
        import tensorflow as tf

        if isinstance(obj, tf.keras.Model):
            is_trainable_module = True
            source = "tensorflow"
    elif source == "paddle" or _is_submodule(obj, "paddle"):
        import paddle

        if isinstance(obj, paddle.nn.Layer):
            is_trainable_module = True
            source = "paddle"
    elif source in ("jax", "flax", "haiku", None): 
        if _is_submodule(obj, "haiku"):
            import haiku as hk

            if isinstance(obj, hk.Transformed):
                is_trainable_module = True
                source = "jax"
        elif _is_submodule(obj, "flax"):
            import flax

            if isinstance(obj, flax.linen.Module):
                is_trainable_module = True
                source = "jax"
    if isinstance(obj, ivy.Module):
        source = "ivy"
        is_trainable_module = True

    if is_trainable_module:
        return module._transpile_trainable_module(
            obj,
            source=source,
            to=to,
            source_mod=source_mod,
            to_mod=to_mod,
            args=args,
            kwargs=kwargs,
            params_v=params_v,
        )

    # return eager graph if args or kwargs are supplied
    if (args is not None) or (kwargs is not None):
        args = ivy.default(args, [])
        kwargs = ivy.default(kwargs, {})
        if ivy.exists(v):
            kwargs = copy.copy(kwargs)
            kwargs["v"] = v
        return _to_target_framework(
            obj,
            *args,
            **kwargs,
            **_transpile_kwargs,
        )

    # untranspiled
    awaiting = LazyGraph(
        obj,
        initializer=transpile,
        source=source,
        to=to,
        v=v,
        with_numpy=with_numpy,
    )

    return awaiting


def unify(
    *objs: Callable,
    source: Optional[str] = None,
    args: Optional[Tuple] = None,
    kwargs: Optional[dict] = None,
    with_numpy: bool = True,
    **transpile_kwargs,
) -> Callable:
    return transpile(
        *objs,
        source=source,
        to="ivy",
        **transpile_kwargs,
        args=args,
        kwargs=kwargs,
        with_numpy=with_numpy,
    )
