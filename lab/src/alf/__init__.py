"""Batched ALFWorld games and the handcoded expert as a teacher. `lab/src` is the import root, as for `sys1` and `menu`."""
import pathlib
import sys

_SRC = str(pathlib.Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
