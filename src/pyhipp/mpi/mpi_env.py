from __future__ import annotations
from mpi4py import MPI as _MPI
from ..core import abc
from typing import Any, Optional

class Op(abc.HasDictRepr):
    
    Impl = _MPI.Op
    
    def __init__(self, impl: Impl = None, **kw) -> None:
        
        super().__init__(**kw)
        
        self.impl = impl

class Predefined:
    
    any_source  = _MPI.ANY_SOURCE
    any_tag     = _MPI.ANY_TAG
    
    sum         = Op(_MPI.SUM)
    null_op     = Op(_MPI.OP_NULL)
    max         = Op(_MPI.MAX)
    min         = Op(_MPI.MIN)
    sum         = Op(_MPI.SUM)
    prod        = Op(_MPI.PROD)
    land        = Op(_MPI.LAND)
    lor         = Op(_MPI.LOR)
    bor         = Op(_MPI.BOR)
    lxor        = Op(_MPI.LXOR)
    bxor        = Op(_MPI.BXOR)
    maxloc      = Op(_MPI.MAXLOC)
    minloc      = Op(_MPI.MINLOC)
    replace     = Op(_MPI.REPLACE)
    no_op       = Op(_MPI.NO_OP)

class Info(abc.HasDictRepr):
    
    Impl = _MPI.Info
    
    def __init__(self, impl: Impl = None, **kw) -> None:
        super().__init__(**kw)
        
        self.impl = impl


class Status(abc.HasDictRepr):
    
    Impl = _MPI.Status
    
    def __init__(self, impl: Impl = None, **kw) -> None:
        
        super().__init__(**kw)
        
        self.impl = impl

