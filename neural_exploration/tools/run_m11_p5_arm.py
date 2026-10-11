"""**单数名别名**（`tools/run_m11_p5_arm.py`）——主agent 指令里点名的文件名。

实现与 CLI **完全等同** `tools/run_m11_p5_arms.py`（同一个 `main()`）；
存在的唯一原因是消除"文件名与指令不一致"造成的核查失败（本项目已多次因**名字/口径**而非事实产生误判）。
两种用法都可用：
    ./.venv-m9/bin/python -m neural_exploration.tools.run_m11_p5_arm  --arm <arm_spec.json> --out-dir <dir>
    ./.venv-m9/bin/python -m neural_exploration.tools.run_m11_p5_arms --plan <plan.json>    --out-dir <dir>
"""
from __future__ import annotations

from neural_exploration.tools.run_m11_p5_arms import main

if __name__ == "__main__":
    raise SystemExit(main())
