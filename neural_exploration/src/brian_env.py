"""Brian2 运行环境配置（M1 实测：cython 编译默认不落盘，每进程重编译 ~10–30s）。

在构建/运行前调用 `configure_brian2()` 一次即可（幂等）。

M2 调整（见 docs/m2_env_notes.md）：
  - 缓存目录改为项目内 `.cache/brian2`（/tmp 下的缓存偶发 macOS Spotlight
    索引导致的 80s 级编译卡顿；项目内目录稳定且可 gitignore）；
  - `delete_source_files=False`：保留 .pyx 源，避免缓存校验失败时重编译。
"""

from __future__ import annotations

import os
import tempfile

_configured = False

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def configure_brian2(cache_dir: str = ""):
    """设置持久化编译缓存目录（默认 <项目>/.cache/brian2）。"""
    global _configured
    if _configured:
        return
    from brian2 import prefs

    cache_dir = cache_dir or os.path.join(_PROJECT_ROOT, ".cache", "brian2")
    os.makedirs(cache_dir, exist_ok=True)
    prefs.codegen.runtime.cython.cache_dir = cache_dir
    # 保留编译源（避免源文件缺失触发重编译卡顿）
    prefs.codegen.runtime.cython.delete_source_files = False
    _configured = True
