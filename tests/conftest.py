"""UniCardio 测试的共享 pytest fixture 与 path 设置。"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# 将仓库根目录加入 sys.path，方便在未安装成包时直接 ``import src.*``。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# flex_attention 在 CPU eager（无 torch.compile）路径会逐次 warn；测试只验
# 证功能正确性，warning 屏蔽掉以保持输出干净。
warnings.filterwarnings(
    "ignore",
    message=".*flex_attention called without torch.compile.*",
)
