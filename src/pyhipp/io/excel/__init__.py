from __future__ import annotations
import typing
from typing import Self
import pandas as pd
from pyhipp.core import DataTable


class Excel:
    @staticmethod
    def dump_file(file: str | pd.ExcelWriter, d: dict | pd.DataFrame,
                  sheet_prefix='', index=False, float_format=None):
        '''
        Save a dict as an Excel file.
        
        @file: filename or pd.ExcelWriter object to which the dict is saved.
        
        @d: dict or pd.DataFrame, data to save. A dict can be nested within 
        another. Keys along the access path are joined by '.' to form the 
        sheet name.
        
        @sheet_prefix: str. Prefix to be added to the sheet name.
        
        @index: bool. Whether to include the index.
        
        @float_format: str. Format for floating point numbers.
        
        Example
        -------
        ```
        d = {
            'panel_a' : {
                'galaxies': pd.DataFrame({
                    'log_Sigma_*': np.array([1.1,2.,3.]),
                    'w_p': np.array([1.,2.,3.]),
                }),
                'halos': pd.DataFrame({
                    'log_M': np.array([1.1111,2.,3.]),
                    'w_p': np.array([1.,2.,3.]),
                }),
            }
        }
        Excel.dump_file('d.xlsx', d, float_format='%.2f')
        d_in = Excel.load_file('d.xlsx'); d_in
        ```
        
        Output:
        ```
        {'panel_a': {'galaxies':    log_Sigma_*  w_p
          0          1.1    1
          1          2.0    2
          2          3.0    3,
          'halos':    log_M  w_p
          0   1.11    1
          1   2.00    2
          2   3.00    3}}
        ```
        
        '''
        if not isinstance(file, pd.ExcelWriter):
            with pd.ExcelWriter(file) as writer:
                Excel.dump_file(writer, d, sheet_prefix=sheet_prefix,
                                index=index, float_format=float_format)
            return
        if isinstance(d, pd.DataFrame):
            d.to_excel(file, sheet_name=sheet_prefix,
                       index=index, float_format=float_format)
            return
        for k, v in d.items():
            _p = sheet_prefix + '.' + k if len(sheet_prefix) > 0 else k
            Excel.dump_file(file, v, _p, index=index, float_format=float_format)

    @staticmethod
    def load_file(file: str | pd.ExcelFile, as_datatables=False,
                  reorder=True):
        d_in = pd.read_excel(file, sheet_name=None)
        if as_datatables:
            d_in = {k: DataTable({k1: v1.to_numpy() for k1, v1 in v.to_dict(
                orient='series', index=False)}, copy=False) for k, v in d_in.items()}
        if reorder:
            d_out = Excel.__reorder_input(d_in)
        else:
            d_out = d_in
        return d_out

    def __reorder_input(
            d_in: dict[str, pd.DataFrame]) -> dict[str, dict | pd.DataFrame | DataTable]:
        d_out = {}
        for key, val in d_in.items():
            if '.' not in key:
                d_out[key] = val
                continue
            keys = key.split('.')
            keys, key_last = keys[:-1], keys[-1]
            _d_out = d_out
            for k in keys:
                _d_out = _d_out.setdefault(k, {})
            _d_out[key_last] = val
        return d_out
