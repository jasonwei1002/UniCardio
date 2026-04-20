"""UniCardio 测试的共享 pytest fixture 与 path 设置。"""

from __future__ import annotations

import sys
from pathlib import Path

# 将仓库根目录加入 sys.path，方便在未安装成包时直接 ``import src.*``。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
