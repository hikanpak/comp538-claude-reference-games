"""The loop that decides who acts, runs the command and records the output. `lab/src` is the import root."""
import pathlib
import sys

_SRC = str(pathlib.Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
