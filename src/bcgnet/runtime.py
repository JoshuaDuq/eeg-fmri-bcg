"""Put the upstream BCGNet tree on sys.path.

jiaangyao/BCGNet is a script package (`from config import get_config`), not a
modern installable module. Workers import it that way after this hook.
"""

from __future__ import annotations

import sys
from pathlib import Path

VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"


def prepare_vendor_imports() -> Path:
    root = str(VENDOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return VENDOR_ROOT
