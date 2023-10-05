import ast

import control_flow.helpers as helpers


class SplitNonlocals(ast.NodeTransformer):
    """
    For simpler logic, split multi-name nonlocal statements into multiple statements
    """

    def visit_Nonlocal(self, node):
        nodes = [ast.Nonlocal(names=[name]) for name in node.names]
        return nodes


class TransformVariables(ast.NodeTransformer):
    """
    Replace closure and nonlocal expressions in a function with calls to the function's VariableHandler.
    """

    def __init__(self, fn):
        self._id = id(fn)
        self.closures = fn.__code__.co_freevars
        self.nonlocals = []  # names declared with nonlocal keyword
        self.inside_definition = False
        self.depth = 0
        return super().__init__()

    def visit_Name(self, node):
        if node.id not in self.closures:
            return node
        if isinstance(node.ctx, ast.Load):
            from_handler = ast.Call(
                func=ast.Attribute(
                    value=ast.Subscript(
                        value=ast.Name(id="global_variable_handlers", ctx=ast.Load()),
                        slice=ast.Index(value=ast.Constant(value=self._id, kind=None)),
                        ctx=ast.Load(),
                    ),
                    attr="get_var",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Constant(value=node.id, kind=None),
                    ast.Call(
                        func=ast.Name(id="locals", ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    ),
                ],
                keywords=[],
            )
            return from_handler
        return node

    def visit_Assign(self, node):
        if len(node.targets) == 1:
            target = node.targets[0]
            assert isinstance(target, ast.Name)
            if self.inside_definition and target.id not in self.closures:
                return self.generic_visit(node)
            if target.id not in self.nonlocals:
                return self.generic_visit(node)
            if target.id not in self.closures:
                return self.generic_visit(node)
            to_handler = ast.Call(
                func=ast.Attribute(
                    value=ast.Subscript(
                        value=ast.Name(id="global_variable_handlers", ctx=ast.Load()),
                        slice=ast.Index(value=ast.Constant(value=self._id, kind=None)),
                        ctx=ast.Load(),
                    ),
                    attr="set_var",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=target.id, kind=None), self.visit(node.value)],
                keywords=[],
            )
            return ast.Expr(value=to_handler)
        else:
            raise Exception("no support for unpacking in assignment yet")

    def visit_AugAssign(self, node):
        if self.inside_definition and node.target.id not in self.closures:
            return self.generic_visit(node)
        if node.target.id not in self.nonlocals:
            return self.generic_visit(node)
        if node.target.id not in self.closures:
            return self.generic_visit(node)

        assign_node = ast.Assign(
            targets=[node.target],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Subscript(
                        value=ast.Name(id="global_variable_handlers", ctx=ast.Load()),
                        slice=ast.Index(value=ast.Constant(value=self._id, kind=None)),
                        ctx=ast.Load(),
                    ),
                    attr="aug_var",
                    ctx=ast.Load(),
                ),
                args=[
                    helpers.flip_context(node.target),
                    ast.Constant(value=node.op.__class__.__name__, kind=None),
                    self.visit(node.value),
                ],
                keywords=[],
            ),
        )

        return self.visit(assign_node)

    def visit_Nonlocal(self, node):
        assert len(node.names) == 1  # statement should be split first
        if self.inside_definition and node.names[0] not in self.closures:
            # this is a local variable
            return self.generic_visit(node)
        for name in node.names:
            # this is a closure variable, track it
            self.nonlocals.append(name)
        return ast.Pass()

    def visit_FunctionDef(self, node):
        if self.depth == 0:
            # this is the outer function def, which we ignore
            self.depth += 1
            return self.generic_visit(node)
        if self.inside_definition:
            return self.generic_visit(node)
        self.inside_definition = True
        ret = self.generic_visit(node)
        self.inside_definition = False
        return ret
