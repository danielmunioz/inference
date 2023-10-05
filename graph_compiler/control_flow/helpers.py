import ast
import sys

from random import randint

# helper functions


def rand_id():
    return str(randint(0, sys.maxsize))


def empty_args():
    return ast.arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )

# helper classes


class GetNames(ast.NodeVisitor):
    """
    Collect the list of Names under a node. Doesn't enter function defs.
    """

    def __init__(self, to_ignore=[]):
        self.to_ignore = to_ignore
        self.names = []
        self.depth = 0
        return super().__init__()

    def visit_Name(self, node):
        if node.id in self.to_ignore:
            return
        self.names.append(node.id)

    def visit(self, node):
        self.depth += 1
        return super().visit(node)

    def visit_FunctionDef(self, node):
        if self.depth > 2:
            return
        return self.generic_visit(node)


def get_names(*nodes, to_ignore=[]):
    if len(nodes) > 1:
        return sum([get_names(node, to_ignore=to_ignore) for node in nodes], [])
    node = nodes[0]
    getnames = GetNames(to_ignore=to_ignore)
    getnames.visit(node)
    names = getnames.names
    return names


def declare_names(names):
    declare_names = [
        ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=ast.Name(id=name, ctx=ast.Load()),
        )
        for name in names
    ]
    return declare_names


def declare_nonlocal(names):
    return ast.Nonlocal(names=names)


def flip_context(node):
    if not isinstance(node, ast.Name):
        return node
    if isinstance(node.ctx, ast.Load):
        return ast.Name(id=node.id, ctx=ast.Store())
    return ast.Name(id=node.id, ctx=ast.Load())
