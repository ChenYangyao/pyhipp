from __future__ import annotations
import typing
from typing import Self
from .named_objs import Group, File
import numpy as np
from pathlib import Path


def txt_to_h5(f_in: str, f_out: str | Group, dtype: np.ndarray,
              d_flag='ac', f_flag='w'):
    '''
    @f_out: 
        str (file_name) or Group (h5.Group) under which the datasets are 
        created. If a file_name is given, a new file is created with f_flag.
    @dtype: describing fields (names and datatype).
    '''
    if isinstance(f_out, (str, Path)):
        with File(f_out, f_flag) as _f_out:
            txt_to_h5(f_in, _f_out, dtype)
        return
    assert isinstance(f_out, Group)
    dsets = np.loadtxt(f_in, dtype=dtype)
    for k in dsets.dtype.names:
        f_out.datasets.create(k, data=dsets[k], flag=d_flag)
