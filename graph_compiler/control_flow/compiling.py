"""
    AST recompiling
"""

import ast
import inspect
import textwrap

from .name_transform import (
    SplitNonlocals,
    TransformVariables,
)
from .variable_handler import VariableHandler
import control_flow.control_functions as cff


global_variable_handlers = {}  # map functions to their handler

temp_globals = {
    "global_variable_handlers": global_variable_handlers,
    "if_else":cff.if_else,
}


def ast_and_back(fn, transformer=lambda x, _: x):
    # parse the function to AST, apply some modification, then compile again.
    tree = fn_to_tree(fn)

    tree = transformer(tree, fn)

    name_replacer = TransformVariables(fn)
    tree = SplitNonlocals().visit(tree)
    tree = name_replacer.visit(tree)

    ns = fn.__globals__
    new_vh(fn)

    ast.fix_missing_locations(tree)

    compiled = compile(tree, filename=inspect.getfile(fn), mode="exec")
    exec(compiled, ns)

    def wrapped(*args, **kwargs):
        # toDo: make the process of attaching variableHandlers more elegant
        insert_globals(ns[fn.__name__].__globals__)
        ret = ns[fn.__name__](*args, **kwargs)
        # toDo: clean variableHandler out of global namespace when done
        return ret

    wrapped.__name__ = fn.__name__

    return wrapped


def new_vh(fn):
    vh = VariableHandler(fn.__globals__, cell_dict(fn))
    global_variable_handlers[id(fn)] = vh
    return vh


def cell_dict(fn):
    # dict of references to cells in closure
    if fn.__closure__ is None:
        return {}
    return {k[0]: k[1] for k in zip(fn.__code__.co_freevars, fn.__closure__)}


def print_tree(fn):
    # visualise AST tree
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)

    print(ast.dump(tree))


def fn_to_tree(fn):
    src = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(src)
    return tree


def insert_globals(g):
    for k in temp_globals:
        if k in g:
            continue
        g[k] = temp_globals[k]


def remove_globals(g):
    for k in temp_globals:
        g.pop(k)
