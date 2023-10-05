from time import perf_counter
import ivy
from graph_compiler import compile
from transpiler.transpiler import transpile
import numpy as np


def benchmark(function, args, kwargs={}, n_iterations=100):
    backend = ivy.current_backend_str()

    compiled_graph = compile(function, args=args, kwargs=kwargs)
    transpiled_graph = transpile(compiled_graph, source=backend, to=backend, args=args, kwargs=kwargs)

    # results_are_same = np.allclose(original_ret, transpiled_ret, atol=1e-4)

    # if not results_are_same:
    #     print(np.max(np.abs(np.array(np.array(original_ret) - transpiled_ret))))
    #     raise Exception("Results are not the same!")

    print("Compiled graph:")
    compiled_graph.list_function_frequencies()
    print("number of functions =", len(compiled_graph._functions))

    print("Transpiled graph:")
    transpiled_graph.list_function_frequencies()
    print("number of functions =", len(transpiled_graph._functions))

    compiled_times = np.zeros(shape=n_iterations)
    transpiled_times = np.zeros(shape=n_iterations)
    for i in range(n_iterations):
        start = perf_counter()
        compiled_graph(*args, **kwargs)
        end = perf_counter() - start
        compiled_times[i] = end

        start = perf_counter()
        transpiled_graph(*args, **kwargs)
        end = perf_counter() - start
        transpiled_times[i] = end

    print("mean compiled time =", np.mean(compiled_times))
    print("compiled time std =", np.std(compiled_times)) 
    print("mean transpiled time =", np.mean(transpiled_times))
    print("transpiled time std =", np.std(transpiled_times))