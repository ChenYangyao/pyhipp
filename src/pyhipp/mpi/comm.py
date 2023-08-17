from __future__ import annotations
from ..core import abc
from .mpi_env import _MPI, Predefined, Status, Info, Op
from typing import Any, Optional, List, Union

class Comm(abc.HasDictRepr):
    
    Impl = _MPI.Comm
    
    repr_attr_keys = ('rank', 'size')

    @staticmethod
    def world() -> IntraComm:
        return IntraComm(_MPI.COMM_WORLD)
    
    def __init__(self, impl: Impl = None, **kw) -> None:
        
        super().__init__(**kw)
        
        self.impl = impl
        
    def dup(self, info: Info = None) -> Comm:
        _info = None if info is None else info.impl
        return Comm(self.impl.Dup(info=_info))
    
    def connect(self, pair_rank: int, tag: int = 0) -> Conn:
        return Conn(self, pair_rank, tag=tag)
        
    @property
    def size(self):
        return self.impl.Get_size()
    
    @property
    def rank(self):
        return self.impl.Get_rank()
    
    def send(self, obj: Any, dst: int, tag: int = 0) -> None:
        return self.impl.send(obj, dst, tag=tag)
    
    def recv(self, 
             src: int = Predefined.any_source, 
             tag: int = Predefined.any_tag, 
             return_status: bool = False):
        
        _status = Status.Impl() if return_status else None
        out = self.impl.recv(source=src, tag=tag, status=_status)
        
        if return_status:
            return out, Status(_status)
        else:
            return out
        
    def barrier(self) -> None:
        return self.impl.barrier()
    
    def gather(self, send_obj: Any, root: int = 0) -> Union[List[Any],None]:
        return self.impl.gather(send_obj, root=root)
    
    def Reduce(self, send_buf, recv_buf, op: Op, root: int = 0) -> None:
        '''
        @send_buf: BufSpec or InPlace.
        @recv_buf: BufSpec or None.
        '''
        return self.impl.Reduce(sendbuf = send_buf,
                         recvbuf = recv_buf, 
                         op = op.impl, root = root)
        
class IntraComm(Comm):
    pass

class InterComm(Comm):
    pass

class Conn(abc.HasDictRepr):
    
    repr_attr_keys = ('comm', 'pair_rank', 'tag')
    
    def __init__(self, comm: Comm, pair_rank: int, tag = 0, **kw) -> None:
        super().__init__(**kw)
        
        self.comm = comm
        self.pair_rank = pair_rank
        self.tag = tag
        
    def send(self, obj: Any) -> None:
        return self.comm.send(obj, self.pair_rank, tag=self.tag)
    
    def recv(self, return_status: bool = False):
        return self.comm.recv(self.pair_rank, tag=self.tag, 
                              return_status=return_status)
