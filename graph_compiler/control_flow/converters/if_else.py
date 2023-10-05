import ast
import control_flow.helpers as helpers


class TransformIfElse(ast.NodeTransformer):
    """
    Replaces if-else control statements with functional equivalents.

    One step of return nesting is done like so:

    if cond:
        return(?) x()
    else:
        return(?) y()

    return fn()

    ->

    def rest():
        return fn()
    def a():
        return(?) x()
        return rest()
    def b():
        return(?) y()
        return rest()

    return if_else(a,b)

    This accounts for all cases of return statements appearing in the if-else body.
    """

    def __init__(self, glob={}):
        self.glob = glob
        return super().__init__()

    def expand_If(self, node, rest_body):
        if_id = "if" + helpers.rand_id()
        orelse_id = "orelse" + helpers.rand_id()
        rest_id = "rest" + helpers.rand_id()
        if_body = node.body
        else_body = node.orelse
        names = helpers.get_names(node, to_ignore=self.glob)

        # place return statements in bodies
        if_body.append(
            ast.Return(
                value=ast.Call(
                    func=ast.Name(id=rest_id, ctx=ast.Load()), args=[], keywords=[]
                )
            )
        )
        else_body.append(
            ast.Return(
                value=ast.Call(
                    func=ast.Name(id=rest_id, ctx=ast.Load()), args=[], keywords=[]
                )
            )
        )

        # in case the rest body is empty
        rest_body.append(ast.Pass())

        # declare all names beforehand as themselves so they can be nonlocal'd
        # and persist outside of the functiondefs
        # in the fix-up pass, these get turned into VariableHandler calls
        declares = helpers.declare_names(names)
        if_names = helpers.get_names(*if_body, to_ignore=self.glob)
        else_names = helpers.get_names(*else_body, to_ignore=self.glob)
        rest_names = helpers.get_names(*rest_body, to_ignore=self.glob)

        # adding nonlocal statement to allow escape of functiondef scope
        if_body.insert(0, helpers.declare_nonlocal(if_names))
        else_body.insert(0, helpers.declare_nonlocal(else_names))
        rest_body.insert(0, helpers.declare_nonlocal(rest_names))

        rest_body_def = ast.FunctionDef(
            name=rest_id,
            args=helpers.empty_args(),
            body=rest_body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        if_body_def = ast.FunctionDef(
            name=if_id,
            args=helpers.empty_args(),
            body=if_body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        else_body_def = ast.FunctionDef(
            name=orelse_id,
            args=helpers.empty_args(),
            body=else_body,
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        control_call = ast.Call(
            func=ast.Name(id="if_else", ctx=ast.Load()),
            args=[
                self.visit(node.test),
                ast.Name(id=if_id, ctx=ast.Load()),
                ast.Name(id=orelse_id, ctx=ast.Load()),
            ],
            keywords=[],
        )

        return [
            *declares,
            rest_body_def,
            if_body_def,
            else_body_def,
            ast.Return(value=control_call),
        ]

    def visit_FunctionDef(self, node):
        for i in range(len(node.body)):
            stmt = node.body[i]
            if isinstance(stmt, ast.If):
                node.body = [
                    *node.body[:i],
                    *self.expand_If(stmt, node.body[i + 1:]),
                ]
                node.body = [self.visit(n) for n in node.body]
                return node

        return self.generic_visit(node)
