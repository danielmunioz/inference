"""
    Placeholders for ivy control functions
"""


def if_else(condition, fn1, fn2):
    if condition:
        return fn1()
    else:
        return fn2()


def while_loop(condition, fn, default):
    while condition():
        breaking, returning, ret = fn()
        if breaking:
            break
        if returning:
            return ret
    return default()


def for_loop(generator, fn, default):
    arg = None

    def iterator():
        nonlocal arg
        try:
            arg = next(generator)
            return True
        except StopIteration:
            return False

    def body():
        fn(arg)

    return while_loop(iterator, body, default)


def nothing():
    pass


def identity(x):
    return x


def try_except(fn1, fn2, errortype=Exception):
    try:
        return fn1()
    except errortype:
        return fn2()
