import ivy
import tensorflow as tf
from graph_compiler import compile
from transpiler.transpiler import transpile


def _tf_fn(x):
    return tf.add(x, x)


def test_compile_only_tf():
    ivy.set_tensorflow_backend()
    x = tf.constant([1.0, 2.0, 3.0])
    graph = compile(_tf_fn, args=(x,))
    graph(x)


def test_transpile_only_tf():
    x = tf.constant([1.0, 2.0, 3.0])
    graph = transpile(_tf_fn, source="tensorflow", to="tensorflow", args=(x,))
    graph(x)
