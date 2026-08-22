# M1 环境与前置遗留处置记录（清单 §1：L1–L4）

> 对应《生物仿真M1实施清单》§1。
> 记录时间：2026-08-22（M1 开工）
> 状态：L1–L4 处置完成，结论已回填 M1 报告

---

## L1 — Python 3.9.6 接近 EOL：评估独立 3.11/3.12

| 项 | 结果 |
|---|---|
| 系统检查 | 本机无 `python3.11`/`python3.12`（`which` 与 `/opt/homebrew`、`/usr/local` 均无），仅有 Xcode 系统 Python 3.9.6 与 `.venv-neuro`（3.9.6） |
| 评估结论 | **无收益则不迁移**：Brian2 2.6.0 / NEURON 9.0.1 / numpy 1.26.4 / scipy 1.13.1 / matplotlib 3.9.4 在 3.9.6 上全部工作正常（M1 全程实测），且 3.11 迁移需重建 venv + lock 文件（~1 天），无功能增益 |
| 决策 | **推迟到 M5**（届时全虫模型若需要更新的引擎/库再评估）；记录为 M1 报告遗留项 |

## L2 — 测试 warnings 噪音（M0 遗留 146 条弃用警告）

- 处置：`tests/conftest.py` 增加 `pytest_configure`，注入 `-W ignore::DeprecationWarning` 与 `ignore::FutureWarning`。
- 效果：`pytest neural_exploration/tests` 输出干净；Brian2 自身弃用提示不再刷屏。

## L3 — 多隔室参考解（M0 遗留：单隔室 RK4 解不适用）

- M0 的 `data/hh1952_trace.csv` 是单隔室 RK4 解；M1 需多隔室参考解。
- 处置：`tools/build_neuron_ref.py` 用 **NEURON 9.0.1 cvode（atol/rtol=1e-8）** 构建与 Brian2
  逐隔室一致的形态学/通道参数参考解 → `data/m1_multicomp_ref.npz`（胞体/树突端/轴突端/各郎飞结/髓鞘中点 V(t)，重采样 dt=0.01ms）。
- **关键可比性要点（实测）**：
  1. **温度必须 6.3°C**：NEURON hh.mod 的速率函数带 Q10 温度缩放（参考温度 6.3°C）；若用 35°C，
     动力学加快 ~23 倍，与经典 HH 1952（Brian2 侧）不可比 → 参考脚本显式 `h.celsius = 6.3`。
  2. **符号约定**：Brian2 2.6.0 SpatialNeuron 的 `Im` 是**内向正**约定（文档式 `Im=gL*(EL-v)`；
     dv/dt=(Im−Iaxial)/Cm）；写成外向正（`gL*(v-EL)` 等）会整体反号 → 静息不稳定、V 发散至 ±10³ mV
     （M1 实测踩坑，已修正为文档约定并复用 `tools/hh_spec.py` 的速率函数）。
  3. **胞体模型差异**：Brian2 `Soma` 为球体（面积 π·d²、内部电阻≈0）；NEURON 侧用 L=diam 的圆柱段 +
     `soma.Ra=0.001` 近似，两侧侧面积均为 π·d²，轴向耦合电阻差异 <1% 量级（P2 实测验证）。
  4. **注入电流单位**：Brian2 point current（nA）与 NEURON IClamp（nA）同单位；
     密度→总量换算用胞体面积 π·d²（10 µA/cm² → 0.1257 nA）。
  5. **门控初值**：NEURON hh INITIAL 与 Brian2 `steady_state(V0)` 一致（v_init=-65）。

## L4 — 单位约定（多隔室扩展）

沿用 `tools/hh_spec.py` 归一约定（µF/cm² · mS/cm² · mV · ms），新增：

| 量 | 单位 | 说明 |
|---|---|---|
| 轴向电阻 Ra / Ri | Ω·cm | 皮层神经元典型值 150（清单 §2.2） |
| 形态学 | µm | 直径/长度；Soma 为球体（面积 π·d²） |
| 注入电流 | nA（point current / IClamp） | 与膜面积换算：`nA = µA/cm² × 面积(cm²) × 1e3` |
| 膜电容 Cm | µF/cm²（逐隔室） | 髓鞘段 0.02（绝缘，见 CSV 注释） |

## 其他实测结论（供复现/报告引用）

1. **Brian2 2.6.0 SpatialNeuron 不支持 `(membrane)` 标志**（新版语法）；非 shared 变量天然逐隔室。
2. **区段访问**：根区段用 `morpho._indices()`；子树用 `morpho['name']`；SpatialNeuron 上按隔室赋值
   用 `neuron[i]`（单隔室 subgroup），避免子树跨段赋值（`neuron.dend1.gNa=x` 会连子区段一起赋）。
3. **显式 Network 必需**：类方法内构建的对象不在 `run()` 调用帧，magic 网络收集不到 →
   `Network(neuron, mon, spmon).run(duration, namespace={'stim': stim})`；TimedArray 非 BrianObject，经 namespace 传入。
4. **数值方法**：rk4 / dt=0.01ms 对 1.5µm 郎飞结稳定（修正符号后）；exponential_euler 亦稳定但慢 ~18×
   （每步全隐式求解）。主线用 rk4（清单 §3 指定），exponential_euler 作精度自检。
5. **编译缓存**：brian2 cython 默认不落盘，每进程重编译 ~10–30s → 设置
   `prefs.codegen.runtime.cython.cache_dir`（测试/脚本统一配置）。
6. **髓鞘 Cm 降低必要性**：gNa=gK=0 但 Cm=1 时，髓鞘电容负载过大（每髓鞘段表面积电容 ≈ 节点 130×），
   AP 跨髓鞘衰减后不足以驱动下一节点再发放 → 改为 cm=0.02 µF/cm²（真实髓鞘的电容绝缘作用），
   AP 得以在郎飞结逐结再发放（P6 前提，详见 m1_report.md）。
7. **变量名坑**：脚本中不要用 `cm` 作循环变量——会遮蔽 Brian2 的厘米单位（M1 实测）。

## 环境快照（与 M0 相同，无新增依赖）

- `.venv-neuro`：Python 3.9.6；Brian2 2.6.0；NEURON 9.0.1；numpy 1.26.4；scipy 1.13.1；matplotlib 3.9.4；pytest 8.4.2
- 版本锁定：`docs/m0_requirements.lock`（M1 未新增包，无需更新）
