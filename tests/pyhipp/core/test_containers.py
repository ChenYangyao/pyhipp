from pyhipp.core import DataDict

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
        
