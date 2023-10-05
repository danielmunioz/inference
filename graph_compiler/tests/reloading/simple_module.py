import torch
from functools import partial

partial_add = partial(torch.add)


def exported_add(*args):
    return partial_add(*args)


class TestCallable:
    def __init__(self):
        pass

    def __call__(self, *args):
        return exported_add(*args)

    def test_method(self, *args):
        return exported_add(*args)
