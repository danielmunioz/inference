# global
from typing import Optional, Dict, Tuple
import re

# local
import ivy
from graph_compiler.conversion import _to_native
import transpiler.transpiler as transpiler


# Transpilation functions


def to_torch_module(module, lazy):
    import torch

    class TranspiledTorchModule(torch.nn.Module):
        def __init__(self, ivy_module, lazy=False):
            torch.nn.Module.__init__(self)
            self._ivy_module = ivy_module
            self.lazy = lazy
            if not lazy:
                self._assign_variables()
                self._parameters_converted = False

        def _assign_variables(self):
            from graph_compiler.conversion import array_to_new_backend

            # TODO: use local ivy.backends.torch here
            ivy.set_backend("torch")
            # Again assuming backend is torch when running this function
            self._ivy_module.v = self._ivy_module.v.cont_map(
                lambda x, kc: _to_native(x, inplace=True)
            )
            ivy_module_weights_in_torch_tensor = self._ivy_module.v.cont_map(
                lambda x, kc: array_to_new_backend(x, native=True)
            )

            ivy_module_weights_in_torch_tensor.cont_map(
                lambda x, kc: self.register_parameter(
                    name=kc, param=torch.nn.Parameter(x)
                )
            )
            ivy.previous_backend()

        def forward(self, *args, **kwargs):
            if self.lazy:
                # Convert to ivy first
                self._ivy_module._module_graph._initialize(*args, **kwargs)
                self.lazy = False
                self._assign_variables()
                self._parameters_converted = False
            # inputs should be only in native tensors
            if self._ivy_module._module_graph and not self._parameters_converted:
                self._ivy_module.v = self._ivy_module.v.cont_map(
                    lambda _, kc: self._parameters[kc]
                )
                self._parameters_converted = True
            # can only use ivy.Module's __call__ only since it has been compiled to be used with torch
            ret = self._ivy_module(*args, **kwargs, v=self._ivy_module.v)
            # Output however could be in ivy.Array form (when ivy_module has not been compiled)
            # So converting to native tensor again
            return ivy.to_native(ret, nested=True)

    torch_module = TranspiledTorchModule(module, lazy=lazy)

    # set compilation flags
    torch_module._ivy_module._lazy_compiled = lazy
    torch_module._ivy_module._target = "torch"

    return torch_module


def to_haiku_module(module, lazy):
    import haiku as hk

    ivy_module = module

    class TranspiledHaikuModule(hk.Module):
        def __init__(self):
            super(TranspiledHaikuModule, self).__init__()
            self._ivy_module = ivy_module
            self._parameters_converted = False
            self.lazy = lazy

        def __call__(self, *args, **kwargs):
            from graph_compiler.conversion import array_to_new_backend

            if (
                self.lazy
                and hasattr(self._ivy_module._module_graph, "_initialized")
                and not self._ivy_module._module_graph._initialized
            ):
                # Convert to ivy first
                self._ivy_module._module_graph._initialize(*args, **kwargs)
                self.lazy = False
                self._ivy_module.v.cont_map(
                    lambda x, kc: hk.get_parameter(
                        name=kc,
                        shape=x.shape,
                        dtype=x.dtype,
                        init=lambda shape, dtype: array_to_new_backend(
                            _to_native(self._ivy_module.v[kc], inplace=True),
                            native=True,
                        ),
                    )
                )
            # assuming backend is set to JAX when using the call method
            # We do not want to interfere with already set ivy_module.v
            # right now it is a hacky fix.
            if self._ivy_module._module_graph is None:
                # this is only called during init
                self._ivy_module.v.cont_map(
                    lambda x, kc: hk.get_parameter(
                        name=kc,
                        shape=x.shape,
                        dtype=x.dtype,
                        init=lambda shape, dtype: array_to_new_backend(
                            _to_native(self._ivy_module.v[kc], inplace=True),
                            native=True,
                        ),
                    )
                )
            elif not self._parameters_converted:
                # if we are using all parameters, we would eventually have to call `hk.get_parameter` for every param,
                # so it's okay to call here, won't result in slowdowns
                # TODO: see if we can remove `array_to_new_backend` from here.
                prev_backend = ivy.current_backend_str()
                ivy.set_backend("jax")
                self._ivy_module.v = self._ivy_module.v.cont_map(
                    lambda x, kc: hk.get_parameter(
                        name=kc,
                        shape=x.shape,
                        dtype=x.dtype,
                        init=lambda shape, dtype: array_to_new_backend(
                            _to_native(self._ivy_module.v[kc], inplace=True),
                            native=True,
                        ),  # this won't be used here tho
                    )
                )
                if prev_backend:
                    ivy.set_backend(prev_backend)
                self._parameters_converted = True

            args, kwargs = ivy.args_to_native(*args, **kwargs)
            ret = self._ivy_module(*args, v=self._ivy_module.v, **kwargs)
            if isinstance(ret, tuple):
                return ivy.args_to_native(*ret)
            return ivy.to_native(ret)

    # set compilation flags
    ivy_module._lazy_compiled = lazy
    ivy_module._target = "jax"

    return TranspiledHaikuModule


def to_flax_module(module, lazy):
    import flax

    class TranspiledFlaxModule(flax.linen.Module):
        ivy_module: ivy.Module 
        lazy: bool = False

        def setup(self):
            self._ivy_module = self.ivy_module
            self._ivy_module._parameters_converted = False
            self._ivy_module.lazy = self.lazy
            if not lazy:
                self._assign_variables()
        
        def _assign_variables(self):
            from graph_compiler.conversion import array_to_new_backend

            ivy.set_backend("jax")

            self._ivy_module.v.cont_map(
                lambda x, kc: self.param(
                    # "vars",
                    kc,
                    lambda _, shape, dtype: array_to_new_backend(
                        _to_native(self._ivy_module.v[kc], inplace=True),
                        native=True,
                    ),
                    x.shape,
                    x.dtype,
                )
            )
            ivy.previous_backend()
        
        @flax.linen.compact
        def __call__(self, *args, **kwargs):

            if self._ivy_module.lazy:
                # Convert to ivy first
                self._ivy_module._module_graph._initialize(*args, **kwargs)
                self._ivy_module.lazy = False 
                self._assign_variables()
                self._ivy_module._parameters_converted = False

            # inputs should be only in native arrays
            if self._ivy_module._module_graph and not self._ivy_module._parameters_converted:
                self._ivy_module.v = self._ivy_module.v.cont_map(
                    lambda _, kc: self.variables["params"][kc]
                )
                self._ivy_module._parameters_converted = True
            args, kwargs = ivy.args_to_native(*args, **kwargs)
            ret = self._ivy_module(*args, v=self._ivy_module.v, **kwargs)
            if isinstance(ret, tuple):
                return ivy.args_to_native(*ret)
            return ivy.to_native(ret)
    
    flax_module = TranspiledFlaxModule(module, lazy=lazy)

    # set compilation flags
    flax_module._lazy_compiled = lazy
    flax_module._target = "jax"

    return flax_module


def to_keras_module(module, lazy):
    import tensorflow as tf

    class TranspiledKerasModel(tf.keras.Model):
        def __init__(self, ivy_module, lazy):
            super(TranspiledKerasModel, self).__init__()
            self._ivy_module = ivy_module
            self._parameters_converted = False
            self.lazy = lazy
            if not lazy:
                self._assign_variables()

        def _assign_variables(self):
            from graph_compiler.conversion import array_to_new_backend

            # TODO: use local ivy.backends.tensorflow here
            ivy.set_backend("tensorflow")

            self._ivy_module.v = self._ivy_module.v.cont_map(
                lambda x, kc: _to_native(x, inplace=True)
            )

            self._ivy_module.v = self._ivy_module.v.cont_map(
                lambda x, kc: array_to_new_backend(x, native=True)
            )
            self._ivy_module.v.cont_map(
                lambda x, kc: self.add_weight(
                    name=kc, shape=x.shape, dtype=x.dtype, trainable=True
                )
            )
            model_weights = list()
            self._ivy_module.v.cont_map(
                lambda x, kc: model_weights.append(ivy.to_numpy(x))
            )
            self.set_weights(model_weights)

            ivy.previous_backend()

        def call(self, *args, **kwargs):
            # set model_weights in self._ivy_module.v, so that the
            # graph uses the trainable weights in the computation;
            if self.lazy:
                # Convert to ivy first
                kwargs_ = dict(kwargs)
                del kwargs_["training"]
                self._ivy_module._module_graph._initialize(*args, **kwargs_)
                self.lazy = False
                self._assign_variables()
                self._parameters_converted = False
            if self._ivy_module._module_graph and not self._parameters_converted:
                params = {
                    re.sub(r":([0-9]+)$", "", param.name).replace(
                        f"{self.name}/", ""
                    ): param
                    for param in self.variables
                }
                self._ivy_module.v = self._ivy_module.v.cont_map(
                    lambda _, kc: params[kc]
                )
                self._parameters_converted = True
            # need to call with the weights passed since compilation was done with it
            ret = self._ivy_module(*args, **kwargs, v=self._ivy_module.v)
            if isinstance(ret, tuple):
                return ivy.args_to_native(*ret)
            return ivy.to_native(ret)

        def __call__(self, *args, **kwargs):
            from graph_compiler.conversion import nest_array_to_new_backend

            ivy.set_backend("tensorflow")
            args = nest_array_to_new_backend(args, native=True)
            kwargs = nest_array_to_new_backend(kwargs, native=True)
            ivy.previous_backend()

            return super(TranspiledKerasModel, self).__call__(*args, **kwargs)

    keras_module = TranspiledKerasModel(module, lazy=lazy)

    # set compilation flags
    keras_module._ivy_module._lazy_compiled = lazy
    keras_module._ivy_module._target = "tensorflow"

    return keras_module


def to_paddle_module(module, lazy):
    import paddle

    class TranspiledPaddleModule(paddle.nn.Layer):
        def __init__(self, ivy_module, lazy=False):
            super(TranspiledPaddleModule, self).__init__()
            self._ivy_module = ivy_module
            self.lazy = lazy
            if not lazy:
                self._assign_variables()
                self._parameters_converted = False

        def _assign_variables(self):
            from graph_compiler.conversion import array_to_new_backend

            # TODO: use local ivy.backends.paddle here
            ivy.set_backend("paddle")

            self._ivy_module.v = self._ivy_module.v.cont_map(
                lambda x, kc: _to_native(x, inplace=True)
            )
            self._ivy_module.v = self._ivy_module.v.cont_map(
                lambda x, kc: array_to_new_backend(x, native=True)
            )
            self._ivy_module.v = self._ivy_module.v.cont_map(
                lambda x, kc: self.create_parameter(
                    shape=x.shape,
                    dtype=x.dtype,
                    default_initializer=paddle.nn.initializer.Assign(x),
                )
            )
            ivy.previous_backend()

        def forward(self, *args, **kwargs):
            if self.lazy:
                self._ivy_module._module_graph._initialize(*args, **kwargs)
                self.lazy = False
                self._assign_variables()
                self._parameters_converted = False
            # inputs should be only in native tensors
            if self._ivy_module._module_graph and not self._parameters_converted:
                self._ivy_module.v = self._ivy_module.v.cont_map(
                    lambda _, kc: self._parameters[kc]
                )
                self._parameters_converted = True

            ret = self._ivy_module(*args, **kwargs, v=self._ivy_module.v)
            return ivy.to_native(ret, nested=True)

    paddle_module = TranspiledPaddleModule(module, lazy=lazy)

    # set compilation flags
    paddle_module._ivy_module._lazy_compiled = lazy
    paddle_module._ivy_module._target = "paddle"

    return paddle_module


def _transpile_trainable_module(
    source_module,
    source,
    to,
    source_mod: Optional[str] = None,
    to_mod: Optional[str] = None,
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict] = None,
    params_v=None,
):
    """
    Converts module in source backend to the target backend.
    Returns a lazily compilable module (in target backend) if no instance args and kwargs are provided.

    params_v : Required for creation of ivy.Module from some source modules (e.g. Haiku)
    """

    if to == "numpy":
        raise ValueError(
            "A module can not be fully transpiled to NumPy. To get an equivalent NumPy function, "
            + "transpile the forward pass instead."
        )

    BACKEND_TO_MODULE_FROM_BACKEND = {
        "torch": ivy.ModuleConverters.from_torch_module,
        "jax": {
            "haiku": ivy.ModuleConverters.from_haiku_module,
            "flax": ivy.ModuleConverters.from_flax_module,
        },
        "tensorflow": ivy.ModuleConverters.from_keras_module,
        "paddle": ivy.ModuleConverters.from_paddle_module,
    }

    if source == "jax" and source_mod is None:
        import flax
        source_mod = "flax" if isinstance(source_module, flax.linen.Module) else "haiku"

    if source != "ivy":  # Probably there is a cleaner way of doing this
        # first let's construct a ivy.Module from the source module
        fw_kwargs = {} 
        if params_v is not None:
            params_key = "params_hk" if source_mod == "haiku" else "params_fx"
            fw_kwargs[params_key] = params_v
        module_converter = BACKEND_TO_MODULE_FROM_BACKEND[source]
        if source == "jax":
            module_converter = module_converter[source_mod]
        ivy_module = module_converter(
            source_module,
            instance_args=args,
            instance_kwargs=kwargs,
            **fw_kwargs,
        )
    else:
        ivy_module = source_module
        source = ivy.current_backend_str()

    # transpile the inner graph
    ivy_module._module_graph = transpiler.transpile(
        ivy_module._call, source=source, to=to, args=args, kwargs=kwargs, v=ivy_module.v
    )
    # if the target is ivy, return an ivy.Module, otherwise convert into corresponding module
    if to == "ivy":
        return ivy_module

    TO_NATIVE_MODULE = {
        "torch": to_torch_module,
        "jax": {
            "haiku": to_haiku_module,
            "flax": to_flax_module,
        },
        "tensorflow": to_keras_module,
        "paddle": to_paddle_module,
    }
    lazy_transpile = args is None and kwargs is None
    to_converter = TO_NATIVE_MODULE[to]
    if to == "jax":
        to_converter = to_converter[to_mod]
    target_module = to_converter(ivy_module, lazy=lazy_transpile)

    return target_module
