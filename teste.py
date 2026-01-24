# test.py
print(__name__)

try:
    # Trying to find module in the parent package
    from . import EDA_utils
    print(EDA_utils.checkNormality())
    del config
except ImportError:
    print('Relative import failed')

try:
    # Trying to find module on sys.path
    import EDA_utils
    print(EDA_utils.checkNormality())
except ModuleNotFoundError:
    print('Absolute import failed')