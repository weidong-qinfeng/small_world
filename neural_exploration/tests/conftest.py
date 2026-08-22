"""pytest 路径配置：把仓库根目录加入 sys.path，使 `import neural_exploration.src...` 可用。"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    """L2 处置：过滤 M0 遗留的弃用警告噪音（清单 §1 L2）。"""
    for w in ("ignore::DeprecationWarning", "ignore::FutureWarning"):
        config.addinivalue_line("filterwarnings", w)
