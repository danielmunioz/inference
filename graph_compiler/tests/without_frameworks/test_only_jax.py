import ivy
import jax.numpy as jnp
from graph_compiler import compile
from transpiler.transpiler import transpile


def _jax_fn(x):
    return jnp.add(x, x)


def test_compile_only_jax():
    ivy.set_jax_backend()
    x = jnp.array([1.0, 2.0, 3.0])
    graph = compile(_jax_fn, args=(x,))
    graph(x)


def test_transpile_only_jax():
    x = jnp.array([1.0, 2.0, 3.0])
    graph = transpile(_jax_fn, source="jax", to="jax", args=(x,))
    graph(x)
