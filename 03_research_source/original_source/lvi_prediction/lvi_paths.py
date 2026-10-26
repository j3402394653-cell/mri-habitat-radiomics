"""Independent LVI exploration paths; original studies remain read-only."""
from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
for directory in (PROJECT_ROOT, CODE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from fold_locked_lvi import digest as sha256
