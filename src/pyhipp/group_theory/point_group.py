from __future__ import annotations
from ..core import abc
import numpy as np
from typing import Any, List, Iterator

class Op(abc.HasName, abc.HasSimpleRepr):
    
    group_size: int = None
    n_dims: int = None
    
    def __init__(self, **kw):
        
        super().__init__(**kw)
        
        self.mat = np.eye(self.n_dims, dtype=float)
        
    @classmethod
    def create(cls, idx: int):
        out = cls()
        out.mat[...] = cls.mats[idx]
        return out
    
    @classmethod
    def group(cls) -> Group:
        ops = [cls.create(idx) for idx in range(cls.group_size)]
        return Group(ops)

    def to_simple_repr(self) -> dict:
        return {
            'mat': self.mat
        }

    def copied(self) -> Op:
        out = type(self)()
        out.mat[...] = self.mat
        return out

    def combined(self, *ops: Op) -> Op:
        out = self.copied()
        for op in ops:
            out.mat = out.mat @ op.mat
        return out
    
    def __call__(self, x: np.ndarray) -> Any:
        return x @ self.mat.T
    
class Group(abc.HasName, abc.HasSimpleRepr):
    def __init__(self, ops: List[Op] = None, **kw) -> None:
        super().__init__(**kw)
        
        if ops is None:
            ops = []
        self.ops = ops
        
    @staticmethod
    def create(name):
        if name in ('cubic', '3d'):
            g2 = ReflectionX.group()
            g1 = RotationXY.group()
            g0 = RotationOther.group()
            return g0.combined(g1, g2)
        elif name in ('square', '2d'):
            g1 = Reflection2DX.group()
            g0 = Rotation2DXY.group()
            return g0.combined(g1)
        else:
            raise KeyError(name)
    
    def copied(self) -> Group:
        return Group([op.copied() for op in self.ops])
    
    def combined(self, *groups: Group) -> Group:
        if len(groups) == 0:
            return self.copied()
        
        g0 = self
        g1, *groups = groups
        ops = []
        for op0 in g0:
            for op1 in g1:
                ops.append(op0.combined(op1))
        g01 = Group(ops)
        
        return g01.combined(*groups)

    def to_simple_repr(self) -> List:
        return [op.to_simple_repr() for op in self.ops]

    def __iter__(self) -> Iterator[Op]:
        return iter(self.ops)
    
    def __len__(self) -> int:
        return len(self.opts)
    
    def __getitem__(self, idx) -> Op:
        return self.ops[idx]
    

class RotationXY(Op):
    
    mats = [
        np.array([
            [ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.],
        ]),
        np.array([
            [ 0., -1.,  0],
            [ 1.,  0.,  0.],
            [ 0.,  0.,  1.],
        ]),
        np.array([
            [-1.,  0.,  0],
            [ 0., -1.,  0.],
            [ 0.,  0.,  1.],
        ]),
        np.array([
            [ 0.,  1.,  0.],
            [-1.,  0.,  0.],
            [ 0.,  0.,  1.],
        ]),
    ]
    group_size = 4
    n_dims = 3
    
class RotationOther(Op):
    
    mats = [
        np.array([
            [ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.],
        ]),
        np.array([
            [ 1.,  0.,  0.],
            [ 0.,  0.,  1.],
            [ 0., -1.,  0.],
        ]),
        np.array([
            [ 1.,  0.,  0.],
            [ 0.,  0., -1.],
            [ 0.,  1.,  0.],
        ]),
        np.array([
            [ 1.,  0.,  0.],
            [ 0., -1.,  0.],
            [ 0.,  0.,  -1.],
        ]),
        np.array([
            [ 0.,  0.,  1.],
            [ 0.,  1.,  0.],
            [-1.,  0.,  0.],
        ]),
        np.array([
            [ 0.,  0., -1.],
            [ 0.,  1.,  0.],
            [ 1.,  0.,  0.],
        ]),
    ]
    
    group_size = 6
    n_dims = 3

class ReflectionX(Op):
    
    mats = [
        np.array([
            [ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.],
        ]),
        np.array([
            [ -1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.],
        ])
    ]
    
    group_size = 2
    n_dims = 3
    
class Rotation2DXY(Op):
    
    mats = [
        np.array([
            [ 1.,  0.],
            [ 0.,  1.],
        ]),
        np.array([
            [ 0., -1.],
            [ 1.,  0.],
        ]),
        np.array([
            [-1.,  0.],
            [ 0., -1.],
        ]),
        np.array([
            [ 0.,  1.],
            [-1.,  0.],
        ]),
    ]
    group_size = 4
    n_dims = 2
    
class Reflection2DX(Op):
    
    mats = [
        np.array([
            [ 1.,  0.],
            [ 0.,  1.],
        ]),
        np.array([
            [ -1.,  0.],
            [ 0.,   1.],
        ])
    ]
    
    group_size = 2
    n_dims = 2