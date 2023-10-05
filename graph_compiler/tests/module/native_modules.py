# global
import haiku as hk
import flax
import jax.numpy as jnp
import paddle
import jaxlib
import tensorflow as tf
import torch
from typing import Sequence


# local
import ivy


# ivy
class TestIvyModule(ivy.Module):
    def __init__(
        self,
        in_size,
        out_size,
        device=None,
        hidden_size=64,
        v=None,
        with_partial_v=False,
    ):
        self._linear0 = ivy.Linear(in_size, hidden_size, device=device)
        self._linear1 = ivy.Linear(hidden_size, hidden_size, device=device)
        self._linear2 = ivy.Linear(hidden_size, out_size, device=device)
        ivy.Module.__init__(self, device=device, v=v, with_partial_v=with_partial_v)

    def _forward(self, x):
        x = ivy.expand_dims(x, axis=0)
        x = ivy.tanh(self._linear0(x))
        x = ivy.tanh(self._linear1(x))
        return ivy.tanh(self._linear2(x))[0]


# jax haiku module


class TestHaikuLinear(hk.Module):
    def __init__(self, out_size):
        super(TestHaikuLinear, self).__init__()
        self._linear = hk.Linear(out_size)

    def __call__(self, x):
        return self._linear(x)


class TestHaikuModule(hk.Module):
    def __init__(self, in_size, out_size, device=None, hidden_size=4):
        super(TestHaikuModule, self).__init__()
        self._linear0 = TestHaikuLinear(hidden_size)
        self._linear1 = TestHaikuLinear(hidden_size)
        self._linear2 = TestHaikuLinear(out_size)

    def __call__(self, x):
        x = jnp.expand_dims(x, 0)
        x = jnp.tanh(self._linear0(x))
        x = jnp.tanh(self._linear1(x))
        return jnp.tanh(self._linear2(x))[0]


def forward_fn(x):
    model = TestHaikuModule(2, 2)
    return model(x)


jax_module = hk.transform(forward_fn)


# jax flax module


class TestFlaxLinear(flax.linen.Module):
    out_size: Sequence[int]

    def setup(self):
        self._linear = flax.linen.Dense(self.out_size)

    def __call__(self, x):
        return self._linear(x)


class TestFlaxModule(flax.linen.Module):
    in_size: Sequence[int]
    out_size: Sequence[int]
    device: jaxlib.xla_extension.Device = None
    hidden_size: Sequence[int] = 4

    def setup(self):
        self._linear0 = TestFlaxLinear(self.hidden_size)
        self._linear1 = TestFlaxLinear(self.hidden_size)
        self._linear2 = TestFlaxLinear(self.out_size)

    def __call__(self, x):
        x = jnp.expand_dims(x, 0)
        x = jnp.tanh(self._linear0(x))
        x = jnp.tanh(self._linear1(x))
        return jnp.tanh(self._linear2(x))[0]


flax_module = TestFlaxModule(in_size=2, out_size=2)


# tensorflow keras model


class TestKerasLinear(tf.keras.Model):
    def __init__(self, out_size):
        super(TestKerasLinear, self).__init__()
        self._linear = tf.keras.layers.Dense(out_size)
        self.tanh = tf.keras.activations.tanh

    def build(self, input_shape):
        super(TestKerasLinear, self).build(input_shape)

    def call(self, x):
        return self.tanh(self._linear(x))


class TestKerasModel(tf.keras.Model):
    def __init__(self, in_size, out_size, device=None, hidden_size=4):
        super(TestKerasModel, self).__init__()
        self._linear0 = TestKerasLinear(hidden_size)
        self._linear1 = TestKerasLinear(hidden_size)
        self._linear2 = TestKerasLinear(out_size)

    def call(self, x):
        x = tf.expand_dims(x, 0)
        x = self._linear0(x)
        x = self._linear1(x)
        return self._linear2(x)


tf_module = TestKerasModel(2, 2)
tf_module.build((2,))


# torch module


class TestTorchModule(torch.nn.Module):
    def __init__(self, in_size, out_size, intermediate=3):
        super(TestTorchModule, self).__init__()
        self.fc1 = torch.nn.Linear(in_size, intermediate)
        self.fc2 = torch.nn.Linear(intermediate, out_size)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        output = self.tanh(self.fc1(x))
        output = self.tanh(self.fc2(output))
        return output


torch_module = TestTorchModule(in_size=2, out_size=2)


# paddle module


class TestPaddleModule(paddle.nn.Layer):
    def __init__(self, in_size, out_size, intermediate=3):
        super(TestPaddleModule, self).__init__()
        self.fc1 = paddle.nn.Linear(in_size, intermediate)
        self.fc2 = paddle.nn.Linear(intermediate, out_size)
        self.tanh = paddle.nn.Tanh()

    def forward(self, x):
        output = self.tanh(self.fc1(x))
        output = self.tanh(self.fc2(output))
        return output


paddle_module = TestPaddleModule(in_size=2, out_size=2)


# globals

NATIVE_MODULES = {
    "paddle": paddle_module,
    "jax": {
        "haiku": jax_module,
        "flax": flax_module,
    },
    "tensorflow": tf_module,
    "torch": torch_module,
}
