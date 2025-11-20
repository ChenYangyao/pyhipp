from pyhipp.core import DataDict, DataTable
import numpy as np

class TestDataDict:
    
    Self = DataDict
    
    def get_d1(self):
        d = DataDict({
            'a': 1,
            'b': '2',
            'c': DataDict({'d': 3, 'e': 4}),
        })
        return d
        
        
    def test_ctor(self):
        d = self.get_d1()
        assert isinstance(d['a'], int) and d['a'] == 1
        assert isinstance(d['b'], str) and d['b'] == '2'
        assert isinstance(d['c/d'], int) and d['c/d'] == 3
        assert isinstance(d['c/e'], int) and d['c/e'] == 4
        print(d)
        
        
    def get_d2(self):
        d = DataDict({
            "a": 1, 
            "b": 2, 
            "c": DataDict({
                "d": 3, 
                "e": [4,5,6,7,8,9,10,11,12,13,14,15,16]
                }),
            "e": DataDict({
                "d": 3, 
                "e": [4,5,6,7,8,9,10,11,12,13,14,15,16]
                })
            })
        return d
    
    def test_ctor2(self):
        d = self.get_d2()
        assert isinstance(d['c/e'], list)
        assert d['c/e'] == [4,5,6,7,8,9,10,11,12,13,14,15,16]
        

class TestDataTable:
    
    Self = DataTable
    
    def get_data_1(self):
        return DataTable({
            'a': np.arange(5),
            'b': np.linspace(0., 1., 5),
            'c': np.ones((5,3)),
        })
        
    def test_ctor(self):
        data = self.get_data_1()
        for key in ['a', 'b', 'c']:
            assert isinstance(data[key], np.ndarray) 
            assert data[key].shape[0] == 5
        a, b, c = data['a', 'b', 'c']
        for v in [a, b, c]:
            assert isinstance(v, np.ndarray) 
            assert v.shape[0] == 5
        a, b, c = data['a,b,c']
        for v in [a, b, c]:
            assert isinstance(v, np.ndarray) 
            assert v.shape[0] == 5
        