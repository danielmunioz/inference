import ivy
import paddle
from graph_compiler import compile
from transpiler.transpiler import transpile


def _paddle_fn(x):
    return paddle.add(x, x)


def test_compile_only_paddle():
    ivy.set_paddle_backend()
    x = paddle.to_tensor([1.0, 2.0, 3.0])
    graph = compile(_paddle_fn, args=(x,))
    graph(x)


# def test_transpile_only_paddle():
#     x = paddle.to_tensor([1., 2., 3.])
#     graph = transpile(_paddle_fn, source='paddle', to='paddle', args=(x,))
#     graph(x)
