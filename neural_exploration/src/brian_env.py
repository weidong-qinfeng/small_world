"""Brian2 运行环境配置（M1 实测：cython 编译默认不落盘，每进程重编译 ~10–30s）。

在构建/运行前调用 `configure_brian2()` 一次即可（幂等）。
"""

from __future__ import annotations

import os
import tempfile

_configured = False


def configure_brian2(cache_dir: str = ""):
    """设置持久化编译缓存目录（默认 /tmp/brian2_m1_cache）。"""
    global _configured
    if _configured:
        return
    from brian2 import prefs

    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "brian2_m1_cache")
    os.makedirs(cache_dir, exist_ok=True)
    prefs.codegen.runtime.cython.cache_dir = cache_dir
    # 运行时确定性的额外保障（无随机性来源，本项仅文档化）
    prefs.codegen.runtime.cython.delete_source_files = True
    _configured = True
