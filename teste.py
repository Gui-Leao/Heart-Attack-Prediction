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

try:
    import EDA_utils
    print(EDA_utils.__file__)
except ModuleNotFoundError:
    print('Absolute i222mport failed')


import pkgutil

mods = [m.name for m in pkgutil.iter_modules()]
#print(mods)



import src
print(src.__file__)

from src import EDA_utils
print(EDA_utils.__file__)