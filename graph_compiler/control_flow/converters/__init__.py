from .if_else import TransformIfElse


def replace_if_else(tree, fn):
    transformer = TransformIfElse(glob=fn.__globals__)
    ret = transformer.visit(tree)
    return ret
