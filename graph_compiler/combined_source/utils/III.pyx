import os

Module_key = "106301343e283bd0bbe27081aa23d91e0d3549f773311be6f9a89c6d6be43be5"

current_key = os.environ.get('IVY_SO_KEY')

if current_key == Module_key:
    pass

else:
    import sys
    sys.exit("!!!!!!")

import numpy as np


class NewNDArray(np.ndarray):
    def __new__(cls, data):
        obj = np.asarray(data) if not isinstance(data, np.ndarray) else data
        new_obj = obj.view(cls)
        new_obj._data = obj
        return new_obj

    def __init__(self, data):
        self._data = np.asarray(data) if not isinstance(data, np.ndarray) else data

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._data = getattr(obj, "_data", None)

    @property
    def data(self):
        return self._data


class NewFloat128(np.float128):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewFloat64(np.float64):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewFloat32(np.float32):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewFloat16(np.float16):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewComplex256(np.complex256):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewComplex128(np.complex128):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewComplex64(np.complex64):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewInt64(np.int64):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewInt32(np.int32):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewInt16(np.int16):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewInt8(np.int8):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewUint64(np.uint64):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewUint32(np.uint32):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewUint16(np.uint16):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewUint8(np.uint8):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


class NewBool(np.bool_):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


NUMPY_TO_CUSTOM = {
    np.ndarray: NewNDArray,
    np.float128: NewFloat128,
    np.float64: NewFloat64,
    np.float32: NewFloat32,
    np.float16: NewFloat16,
    np.complex256: NewComplex256,
    np.complex128: NewComplex128,
    np.complex64: NewComplex64,
    np.int64: NewInt64,
    np.int32: NewInt32,
    np.int16: NewInt16,
    np.int8: NewInt8,
    np.uint64: NewUint64,
    np.uint32: NewUint32,
    np.uint16: NewUint16,
    np.uint8: NewUint8,
    np.bool_: NewBool,
}

custom_np_classes = [
    NewNDArray,
    NewBool,
    NewFloat64,
    NewFloat32,
    NewFloat16,
    NewComplex128,
    NewComplex256,
    NewComplex64,
    NewFloat128,
    NewInt16,
    NewInt32,
    NewInt64,
    NewInt8,
    NewUint16,
    NewUint32,
    NewUint64,
    NewUint8,
]