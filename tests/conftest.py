"""conftest.py — 把项目根目录加入 sys.path，确保 agent / skills 可正常导入。"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
