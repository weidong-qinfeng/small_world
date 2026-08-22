# M0 引擎选型基准（清单 §3）

> 实验日期：2026-08-22（M0 实施）
> 实验机：Apple M1 Pro（arm64），macOS 14.8.7，venv: `.venv-neuro`（Python 3.9.6）
> 完整原始数据：`reports/neuro/m0_benchmark_results.json`、`m0_benchmark_traces.npz`、`m0_benchmark_compare.png`

---

## 1. 统一基准实验规格（三引擎 + RK4 参考解共用，保证可比）

| 项 | 值 |
|---|---|
| HH 模型 | HH 1952 标准参数：gNa=120 mS/cm², gK=36, gL=0.3, ENa=+50, EK=-77, EL=-54.4 mV, Cm=1 µF/cm² |
| 注入电流 | 阶跃 **10 µA/cm²**（清单"10nA"按面积归一换算，见下方单位说明），持续 50 ms |
| 步长 / 时长 | dt=0.01 ms，输出 100 ms 轨迹 |
| 参考解 | 手工 RK4（`tools/reference_data.py`），落盘 `data/hh1952_trace.csv` |
| 求解器 | Brian2: rk4；NEURON: 隐式欧拉（默认）；NEST: 未装 |

> **单位说明**：标准 HH 参数按膜面积归一（mS/cm²、µA/cm²）。清单 §3.1 的"10nA"在本基准统一为 **10 µA/cm² 单位面积注入**——三引擎与参考解全部按同一面积归一实现，数值严格可比。NEURON 侧按膜面积换算绝对电流：I_abs = 10 µA/cm² × A_cell（A_cell≈π·d·L，本实验 7854 µm² → 0.0785 nA）。

## 2. 基准结果

| 指标 | Brian2 2.6.0 | NEURON 9.0.1 | NEST |
|---|---|---|---|
| 安装 | ✅ pip 秒装（纯 Python + 可选 Cython 加速） | ✅ pip 秒装（官方 macosx arm64 wheel） | ❌ 源码编译失败（本机无 cmake） |
| HH 波形 RMSE（vs RK4） | **2.5e-12 mV**（数值恒等） | **0.60 mV（0.52%）** | — |
| 发放次数（参考=4） | 4 | 4 | — |
| 首个 AP 时间（参考=1.82ms） | 1.82 ms | 1.83 ms | — |
| HH 单神经元耗时（100ms，含预热后稳态） | 0.080 s | 0.023 s | — |
| 20 神经元随机网络耗时（100ms，20% 连接） | 0.135 s | 0.036 s | — |
| 波形峰值 / 谷值 | +40.27 / -75.19 mV | +40.07 / -75.22 mV | — |

> NEURON 与 RK4 参考的 0.6 mV 差异来源：NEURON 默认隐式欧拉求解 vs 参考 rk4（dt 相同），
> 尖峰上升沿处离散化差异累积；0.52% ≪ 5% 验收阈值，判定**波形正确**。
> Brian2 用 rk4 求解器 → 与手工 RK4 数值一致（RMSE≈0），交叉验证了两套独立实现。

## 3. 扩展性探测（为 M5 全虫 302 神经元预留）

- 同一引擎上跑 **20 神经元 HH + 随机化学突触网络**（20% 连接概率，确定性种子 42），100 ms。
- Brian2：0.135 s；NEURON：0.036 s。
- 线性外推到 302 神经元（×15 规模）：Brian2 ≈ 2 s，NEURON ≈ 0.5 s 量级 → **全虫级网络在两引擎上均数量级可行**（M5 实测确认）。

## 4. 基准表评分与主线引擎决策

| 维度（权重） | Brian2 | NEURON | NEST | 权重 |
|---|---|---|---|---|
| 安装难度（40%） | 5（秒装，无编译） | 4（秒装，wheel 依赖平台） | 1（需 cmake 源码编译，本机失败） | 40% |
| HH 波形正确性（30%） | 5（RMSE≈0） | 5（0.52% < 5%） | —（未装，0 分） | 30% |
| 20 神经元网络耗时（20%） | 4（0.135s，快但含脚本解释开销） | 5（0.036s，最稳） | —（0 分） | 20% |
| API 表达力（10%） | 4（声明式、多隔室/突触/STDP 齐备） | 5（多隔室/突触最完整、学术界标准） | —（0 分） | 10% |
| **加权总分** | **4.6** | **4.7** | 0.4 | |

**主线引擎决策：Brian2 为主线**（4.6 vs 4.7，差距 0.1 < 0.3，按清单 §3.3 决策模板取分接近者中开发速度快的）。

**书面理由**：
1. 波形正确性两引擎都达标（Brian2 与 RK4 数值恒等；NEURON 0.52%），科学可靠性均满足 M1 的 <5% 验收。
2. 差距仅 0.1 分，完全在测量噪声内——此时清单决策模板指定"推荐 Brian2 为主线（开发速度快）"。
3. Brian2 声明式语法适合 M1–M4 快速迭代（本 M0 的 smoke_loop 即用 Brian2 一次跑通闭环），
   NEURON 隐式欧拉 + 面积/绝对电流换算的工程摩擦更高。
4. **M5 前若遇性能墙**：两引擎均以 HH 连续方程实现，迁移代价 = 重写 NeuronGroup/膜方程层（约 1 周），
   且 M0 已锁定两引擎版本与参考解，届时用同一 RK4 参考解回归即可。设计文档（附录引擎对比表）
   倾向 NEURON 用于 M1–M5 全虫，因此 M5 前再评估一次：若 302 神经元网络 Brian2 实测 >10×NEURON，
   主线切换为 NEURON，理由补记于 m0_report.md。

**NEST 结论**：安装可行性=否（无 cmake），按清单 §8 记为 1 分"待 M5 评估"；其 HH 参数体系（hh_psc_alpha）
与标准 HH 1952 有差异，即便装上基准可比性也弱，不列为候选主线。

---

## 5. 复现命令

```bash
cd /Users/weidong/ai/small_world
python3 -m venv .venv-neuro
source .venv-neuro/bin/activate
pip install -r neural_exploration/docs/m0_requirements.lock
python neural_exploration/tools/reference_data.py      # 生成参考解 csv
python neural_exploration/tools/run_benchmark.py       # 重跑基准（自动跳过未装引擎）
```
