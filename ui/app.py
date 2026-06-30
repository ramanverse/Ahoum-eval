import os
import sys
from pathlib import Path

# Resolve root path and add to python path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Exec the real app script inside the global namespace
real_app_path = _ROOT / "src" / "ui" / "app.py"
with open(real_app_path, "r", encoding="utf-8") as f:
    code = f.read()

exec(code, globals())
