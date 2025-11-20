import os

def test_env():
    out = os.environ.get('PYTHONPATH', None)
    print('Python PATH:', out)