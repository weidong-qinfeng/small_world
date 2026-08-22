# M0 验证报告

> 里程碑：M0 工具链与环境（清单《生物仿真M0实施清单》）
> 完成日期：2026-08-22
> 状态：**P1–P5 全部通过 ✅**（详见 §6 Pass 对照）

---

## 1. 环境记录

### C1–C5 前置确认（详见 `m0_env_report.md`）

| # | 检查项 | 结果 |
|---|---|---|
| C1 | 芯片架构 | arm64（Apple M1 Pro，macOS 14.8.7） |
| C2 | 网络可用性 | ✅ pypi/github 可达，走在线安装 |
| C3 | 系统 Python | /usr/bin/python3 = 3.9.6（Xcode 系统），必须用 venv |
| C4 | 磁盘空间 | 409 GiB 可用（远超 2 GiB） |
| C5 | git 状态 | neural_exploration/ 未跟踪，M0 产物独立提交 |

### 虚拟环境与依赖

- venv：`.venv-neuro`（Python 3.9.6），版本锁定 `docs/m0_requirements.lock`（29 个包，一条命令重建）
- 关键版本：numpy 1.26.4（brian2 2.6.0 预编译扩展与 numpy 2.x 不兼容，已降级并记录）、scipy 1.13.1、
  matplotlib 3.9.4、pytest 8.4.2、Brian2 2.6.0、NEURON 9.0.1、Cython 0.29.37（brian2 需 <3.x）

## 2. 引擎基准与主线引擎决策

**完整基准表见 `m0_engine_benchmark.md`**。摘要：

| 引擎 | 安装 | HH RMSE vs RK4 | 20 神经元网络 | 加权分 |
|---|---|---|---|---|
| Brian2 2.6.0 | ✅ 秒装 | 2.5e-12 mV（数值恒等） | 0.135 s | 4.6 |
| NEURON 9.0.1 | ✅ 秒装 | 0.60 mV（0.52%） | 0.036 s | 4.7 |
| NEST | ❌ 需 cmake 编译 | — | — | 0.4 |

**主线引擎 = Brian2**（4.6 vs 4.7，差距 0.1 < 0.3 → 按清单决策模板取开发速度快的 Brian2；
理由与 M5 前切换 NEURON 的代价评估见基准文档 §4）。

## 3. 最小闭环结果（刺激 → 感觉神经元 → 突触 → 运动神经元 → 虚拟肌肉）

引擎：**Brian2**（主线引擎，rk4，dt=0.01ms，100ms）

| 场景 | sensory_spikes | motor_spikes | muscle_contraction（终值） | muscle_max |
|---|---|---|---|---|
| 刺激 10 µA/cm²（50ms） | 4 | 3 | 0.088（>0 ✅） | 1.66 |
| 无刺激（对照） | 0 | 0 | 0.000 | 0.00 |

- 刺激/响应叠加图：`reports/neuro/m0_smoke.png`（感觉 V / 运动 V / 肌肉收缩三轨）
- 确定性铁律：同参数重跑逐位一致（测试 `test_response_reproducible` 验证）

### 冒烟测试结果（`pytest neural_exploration/tests -v`）

```
4 passed（3 断言全绿 + 1 附加轨迹有限性测试）
```

| 断言 | 结果 |
|---|---|
| test_stimulus_produces_response（刺激→运动发放→收缩>0） | ✅ |
| test_response_reproducible（同参数重跑一致） | ✅ |
| test_no_stimulus_no_response（无刺激→无发放无收缩） | ✅ |
| test_traces_are_finite（轨迹无 NaN/Inf） | ✅ |

## 4. 阻塞项与遗留问题

| 项 | 状态 | 处置 |
|---|---|---|
| NEST 未装成功（无 cmake） | 遗留 | 不阻塞 M0；按清单打 1 分，M5 全虫前再评估（届时装 cmake 或直接用 NEURON） |
| Python 3.9.6 接近 EOL | 风险 | M0 后评估装独立 3.11/3.12；当前 numpy 1.26.4 是 py3.9 最后一版主线 |
| numpy 2.x 与 brian2 2.6.0 二进制不兼容 | 已解决 | 锁定 numpy 1.26.4 + cython 0.29.37（见 lock 文件） |
| Brian2 无 reset 神经元阈值事件持续触发 | 已解决 | HH 神经元统一加 refractory=2ms（生理合理，不改变轨迹） |
| NEURON 默认 el=-54.3 vs 清单 -54.4 | 已解决 | 基准脚本显式设 `sec.hh.el = -54.4` |

## 5. M1 开工前提确认清单

- [x] **波形误差 <5% 的参考解就绪**：`data/hh1952_trace.csv`（RK4 解，M1 需对齐 HH 1952 原文数据）
- [x] 指标模块可用：`tools/metrics.py`（waveform_rmse / spike_count / first_spike_time）
- [x] 可视化可用：`tools/plot_trace.py`（--demo 出图，验收通过）
- [x] 主线引擎选定：Brian2（理由见基准文档）
- [x] 冒烟闭环可复现：3 断言全绿 + 图落盘
- [x] 环境可复现：`m0_requirements.lock` 一条命令重建
- [ ] M1 数据对齐：HH 1952 原文电压钳数据（由 `tools/param_fit.py` 标定，M1 任务）

## 6. Pass 标准对照（清单 §0）

| Pass | 定义 | 结果 |
|---|---|---|
| P1 | 独立虚拟环境可复现（requirements 锁定，一条命令重建） | ✅ `.venv-neuro` + `m0_requirements.lock` |
| P2 | 三引擎基准表完成，主线选定并有书面理由 | ✅ `m0_engine_benchmark.md`（Brian2，含切换代价评估） |
| P3 | 评测框架骨架可运行：≥1 指标 + ≥1 验证图落盘 | ✅ metrics.py + m0_example.png / m0_benchmark_compare.png / m0_smoke.png |
| P4 | 最小闭环测试通过：刺激产生可重复响应 | ✅ 4 passed（含确定性断言） |
| P5 | pytest tests/neuro 全绿，M0 验证报告写盘 | ✅ 本报告 + 4 passed |

**M0 达标。** 交接给 M1 的起点（清单 §10）：`src/ion_channels.py` 正式实现 HH 通道、
`waveform_rmse` 用于 <5% 验收、参考解对齐原文数据、感觉神经元升级多隔室。
