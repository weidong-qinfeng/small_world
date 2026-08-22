"""占位：HH Na/K/漏 离子通道（M1 实现）。

M0 阶段参考解与基准实验直接使用 tools/hh_spec.py 中的 HH 参数与速率函数。
M1 将在此实现可组合的离子通道模块（Na/K/漏，多隔室支持）。
"""

from tools.hh_spec import EK, EL, ENA, GK, GL, GNA  # noqa: F401  # 参数已在 M1 起点确认
