from __future__ import annotations
import typing
from typing import Self, Any
import json
import numpy as np
from ...core import DataDict, DataTable

_builtin_encoder = json.JSONEncoder()


def _builtin_encode(o):
    return _builtin_encoder.encode(o)


def _added_encode(o) -> Any:
    try:
        _Registry.type_map


def _encode(o) -> Any:
    pass


def _decode(o) -> Any:
    pass


class TypeInfo:

    annotation: str | None

    @property
    def _annotation(self):
        a = self.annotation
        if a is None:
            return ''


class _NumpyNdarrayTypeInfo(TypeInfo):

    annotation = 'numpy.ndarray'
    type_obj = np.ndarray

    def endcode(self, o: np.ndarray):
        a = self.annotation
        o_out = o.tolist()
        return {
            '__type__': a,
            'value': o_out
        }

    def decode(self, o_in: dict):
        a = self.annotation
        try:
            assert a == o_in['__type__']
            value = o_in['value']
            value = np.array(value)
        except:
            raise TypeError(f'Failed to decode {o_in} as {a}')
        return value


class _PyhippCoreDataDictTypeInfo(TypeInfo):

    annotation = 'pyhipp.core.DataDict'
    type_obj = DataDict

    def encode(self, o: DataDict):
        a = self.annotation
        return {
            '__type__': a,
            'value': {k: _encode(v) for k, v in o.items()}
        }

    def decode(self, o_in: dict):
        a = self.annotation
        try:
            assert a == o_in['__type__']
            value: dict = o_in['value']
            value = DataDict({k: _decode(v) for k, v in value.items()})
        except:
            raise TypeError(f'Failed to decode {o_in} as {a}')
        return value


class _Registry:
    type_map = {t.annotation: t() for t in
                [
        _NumpyNdarrayTypeInfo,
        _PyhippCoreDataDictTypeInfo,
    ]
    }
