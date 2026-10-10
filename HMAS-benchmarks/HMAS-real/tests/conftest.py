import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAS_DIR   = os.path.dirname(_TESTS_DIR)
_ROOT      = os.path.dirname(_MAS_DIR)

for p in (_ROOT, _MAS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
