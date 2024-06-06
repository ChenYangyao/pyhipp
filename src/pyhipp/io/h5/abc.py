from __future__ import annotations
import typing
from typing import Self, ClassVar, Type, Any
from .named_objs import Group, File
from pathlib import Path

class GroupLike:
    
    h5_data_typedict: ClassVar[dict[str, Type[GroupLike]]] = {}
    
    def _to_h5_data(self) -> dict:
        return {
            k: to_h5_data(getattr(self, k))
            for k in self.h5_data_typedict.keys()
        }
    
    @classmethod
    def _from_h5_data(cls, data: dict, **init_kw) -> Self:
        init_kw = {
            k: from_h5_data(data.pop(k), t)
            for k, t in cls.h5_data_typedict.items()    
        } | init_kw
        return cls(**init_kw)
    
    def to_h5_group(self, group: Group, flag='x'):
        data = self._to_h5_data()
        group.dump(data, flag=flag)
    
    def to_h5_file(self, path: Path, group='/', f_flag='x', flag='x'):
        with File(path, flag=f_flag) as f:
            self.to_h5_group(f[group], flag=flag)
    
    @classmethod    
    def from_h5_group(cls, group: Group, **init_kw) -> Self:
        data = group.load()
        return cls._from_h5_data(data, **init_kw)
    
    @classmethod
    def from_h5_file(cls, path: Path, group='/', **init_kw) -> Self:
        with File(path) as f:
            out = cls.from_h5_group(f[group], **init_kw)
        return out
    
def to_h5_data(obj: Any):
    try:
        fn = getattr(obj, '_to_h5_data')
    except AttributeError:
        out = obj
    else:
        out = fn()
    return out

def from_h5_data(data: Any, tp: type, init_kw: dict = {}):
    try:
        fn = tp._from_h5_data
    except AttributeError:
        if issubclass(tp, str):
            out = tp(data.decode(), **init_kw)
        else:
            out = data
    else:
        out = fn(data, **init_kw)
    return out