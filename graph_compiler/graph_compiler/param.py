# global
from typing import Optional, Tuple, Type

# local
import ivy


class Param:
    """A class for parameters in the graph. Parameters can be native arrays
    such as torch tensors or something else tracked in the graph such as
    native shapes, ints etc.

    Attributes
    ----------
    ptype
        the type of the parameter e.g. <class 'torch.Tensor'>.
    is_var
        whether the parameter is a variable.
    shape
        the shape of the parameter, if one exists.
    """

    def __init__(
        self,
        ptype: Type[ivy.NativeArray],
        is_var: bool = False,
        shape: Optional[Tuple[int, ...]] = None,
    ):
        self.ptype = ptype
        self.is_var = is_var
        self.shape = tuple(shape) if ivy.exists(shape) else None

    def __repr__(self):
        return "<Param, type={}>".format(self.ptype)
