# M9-B2 第一批结算（步骤 1+2：向量化内核 + 全规模装配 + P4 静息 sanity）

> 结算人：M9-B2 执行节点 ｜ 日期：2026-10-10 ｜ 状态：**第一批交付完成，停下等主agent验收**
> 运行前缀固化：`PYTHONHASHSEED=0 MPLBACKEND=Agg ./.venv-m9/bin/python`
> 冻结件零修改：M0–M8 全部不动；`tools/scan_m9_engine.py`（B1 件）只读 import。
> 本结算期间主agent 接管 T=30s 复测 → 本节点**未启动任何 GPU 任务**（仅文档/落盘）。

---

## 0. 结论速览

| 项 | 结论 |
|---|---|
| **G0 对齐门（§2.5 重跑）** | ✅ **PASS** — 融合内核四档全 **100.00%**（636/636，gpu-only=0）+ step↔chunk 逐 spike 一致 + 确定性 100% |
| **P4 静息 sanity** | ⚠️ **PASS(4/6) + INDETERMINATE(2/6)** — (b)(c)(e)(f) 通过；(a)(d) **因协议窗长分辨力不足判 INDETERMINATE**（非反证） |
| **H6 真实 GABA 边消融** | ❌ **反证（方向反转）**；机制中介 = AHP（主agent L27 确认 H-b），修复方向在扫描中 |
| **结构完备性（全规模装配）** | ✅ 139,255 神经元 / 15,091,983 连接 / 54,492,922 突触；真实 GABA **3,233,022 边（21.42%）**；构建 0.79 s / 显存 1.047 GB |
| **预算** | 本批增量 CPU ≈0.9、GPU ≈1.5 GPU-h；累计 M9 CPU ≈7.9 / GPU ≈6.6（上限 8 / 24） |

---

## 1. 交付物清单（路径 + 验证一句话）

| 类别 | 交付物 | 验证 |
|---|---|---|
| 内核 | `src/adult_engine.py` | 对齐门四档 100%（§2.5 重过）；含 B1 融合手段 + 向量化事件交付 + 2 处修复 |
| 回路 | `src/adult_circuit.py` | 139,255 神经元装配成功；0.79 s；1.047 GB；背景驱动/静息协议/消融接口 |
| 装配工具 | `tools/build_m9_network.py` → `data/m9_network.npz`、`m9_network_stats.json`、`m9_neuron_table.npz` | 2.4 GB CSV 流式解析 10.2 s / 落盘 31.9 s；54,492,922 突触与 P1 一致 |
| 对齐探针 | `tools/scan_m9_engine_vec.py` → `data/m9_engine_vec_probe.csv`、`reports/neuro/m9_engine_alignment_vec.png` | `alignment_gate=PASS`；规模分解 + 全规模投影 |
| 标定 | `tools/calibrate_m9_weights.py` → `data/m9_weight_calibration.csv` | **脚本正常退出（exit 0）**；4/4 配置落盘 + 状态说明段 |
| P4 验证 | `tools/validate_p9_resting.py` → `data/m9_p4_resting.csv`、`reports/neuro/m9_p4_resting.png` | 8 项判据带对照 + 实测墙钟；含判定重分类段 |
| P4 分辨力 | `data/m9_p4_resolution.csv` | 定量分析：同质等价 λ=0.1456 Hz **落在带内** → (a)(d) 重分类依据 |
| P4 墙钟 | `data/m9_p4_wallclock.csv` | 安静窗口 1.98 ms/step vs 争用 18 ms/step + **无泄漏证据**（内存/对象数恒定） |
| H6 消融 | `tools/validate_p9_gaba_ablation.py` → `data/m9_p4_gaba_ablation.csv`、`reports/neuro/m9_p4_gaba_ablation.png` | 配对 4 臂（v_floor −80/−70）；方向反转 |
| 判据带 | `data/m9_behavior_reference.csv` | resting 段 8 行 + `protocol_change` 行；**判据带未改**（主agent 已核验合规） |
| 回路参数 | `data/m9_circuit_params.csv` | 定稿参数 + 抽象登记 + 实测显存 |
| 环境笔记 | `docs/m9_env_notes.md` | L20–L26.4（B2 段）+ **L23 裁决请求**（本批追加） |

---

## 2. P4 判定与关键数值

**协议**：全规模 point 档 139,255 神经元；settle = 1 s；**T = 2 s（§3.5.4 最短协议，见限制）**；N = 3 固定 seed；无刺激无梯度；背景 Poisson 驱动。

| 判据 | 实测 | 带 | 判定 |
|---|---|---|---|
| (a) `rate_median_hz` | 0.0000 | [0.1, 10] | ⏳ **INDETERMINATE**（分辨力不足） |
| (b) `silent_fraction` | **0.7474** | [0.5, 0.9] | ✅ **PASS** |
| (c) `rate_p95_hz` | 0.500 | ≤ 50 | ✅ PASS |
| `pop_rate_hz` | 0.1265 | ≤ 20 | ✅ PASS |
| (d) `bout_active_fraction` / `bout_count_per_min` | 0.0000 / 0.000 | [0.05,0.9] / [1,600] | ⏳ **INDETERMINATE**（T=2s 仅 20 个箱） |
| (e) determinism | Spearman **1.000**；**逐位一致 True** | ≥ 0.99 | ✅ PASS |
| (f) `ms_per_step` | **0.898** | ≤ 6 | ✅ PASS |

**判定语义修正依据**（`data/m9_p4_resolution.csv`）：
- P(2 s 窗内 0 spike) = 0.7474 → 同质等价率 **λ = −ln(P0)/T = 0.1456 Hz**，**落在判据带内**；
- λ = 0.1 Hz 的神经元在 T=2s 内期望仅 **0.20** 个 spike（**数学必然观测为 0**），而在 T=30s 内 **57.7%** 概率给出 ≥3 spike。
- ⇒ **T=2s 的 median = 0 与"真实中位率落带"不相矛盾** → (a)(d) **不作反证结论、不写入缺失机制清单**。

---

## 3. H6 消融（反证登记）

配对 4 臂（`data/m9_p4_gaba_ablation.csv`，T=1s、g_ext=19.23、ahp=2500）：

| v_floor | 基线均值 / 静默 | GABA-off 均值 / 静默 | Δ率均值 / Δ静默 |
|---|---|---|---|
| −80 mV（定稿档） | 0.0632 / 0.9368 | 0.0248 / 0.9752 | **−0.0384 / +0.0384** |
| −70 mV（假说修复档） | 0.0634 / 0.9366 | 0.0244 / 0.9756 | −0.0390 / +0.0390 |

- 期望方向（消融 → 活动↑ / 静默↓）**相反**；本节点机制假说"v_floor 越界"经检验**证伪**。
- **主agent L27 已确认真实中介 = AHP（H-b）**：`ahp_inc=0` 时方向正确（Δ = +63.46）。
- **登记**：`R2'`（真实 GABA 标注已具备，但其抑制作用在当前 AHP 强度下被反转）→ 机制定论前**不得**以 H6 断言"抑制平衡必需性"。

---

## 4. 结构性正面结果（相对 M8 的实质改进，务必保留）

| 结构项 | 实测 | 对照 | 意义 |
|---|---|---|---|
| 全规模装配 | **139,255 / 15,091,983 / 54,492,922** | P1 权威锚 | 全脑规模真实装配达成 |
| **真实抑制边** | **3,233,022 条（21.42%）/ 12,755,910 突触（23.4%）** | `m9_inhibition_inventory.csv` 逐项一致 | **M8 R2 数据侧解除**（无 hash 分配） |
| 构建 / 显存 | **0.79 s / 1.047 GB** | §2.2 <1.5 GB | 双达标 |
| **静息稳定性** | **静默 0.7474 ∈ [0.5,0.9]**；无 NaN、无饱和/癫痫（p95 0.5 Hz ≪ 50；群体 0.1265 Hz ≪ 20） | §3.5.2 (b)(c) | **真实 GABA 抑制边下静息稳定、不失控** |
| 确定性 | 逐神经元计数**逐位一致** | §0.7 #3 | 超出统计级要求 |
| 单试次墙钟 | **0.898 ms/step** → 30 s = **0.15 GPU-h** | §3.5.4 ≤ 1 GPU-h | 6.6× 余量 |

---

## 5. 预算消耗

| 项 | 本批增量 | 累计 M9（B1+B2） | 上限（§10.1） |
|---|---|---|---|
| CPU | ≈0.9 CPU-h | ≈7.9 CPU-h | ≤8 CPU-h（§4） |
| GPU | ≈1.5 GPU-h | ≈6.6 GPU-h | ≤24 GPU-h（§4） |

- 明细：对齐门重跑 ~0.5 GPU-h；T=30s/T=5s 中止尝试与诊断 ~1.0 GPU-h。
- **中止的长试次按测量限制登记，不计入协议预算**（请求裁决 ⑦）。

---

## 6. 限制与未决项（完整清单见 `docs/m9_env_notes.md` L23）

1. **T = 2 s 而非预注册 30 s**：本机共享负载（load 4.7–6.2；他会话 2 个 MPS 进程 86–90% + Chrome 99%）使同代码路径实测 18 ms/step vs 安静窗口 **1.98 ms/step**；3 次 T=30s 尝试均 >24 min 未完成 100k/600k 步而中止。**T=30s 复测已由主agent 接管。**
2. **🔴 L27/L28 扫描背景驱动口径与 P4 交付档相差 480×**（扫描 `g_ext=0.04` vs 交付 `g_ext=19.23`）；同一 `ahp_inc=2500` 下活性为 0.00085 Hz/静默 0.99915 vs **0.1265 Hz/静默 0.7474（落带）** → 请求在交付口径下复跑 ahp×H6 扫描（L23 ④）。
3. **H6 方向反转**未机制定论（H-b 已由主agent 确认中介）→ 保持 R2' 反证登记。
4. **抽象登记 3 条**（统一 1.0 ms 延迟 / Poisson 背景驱动 / AHP 适应电流）待确认。
5. **`ahp_inc` 的物理形式**（恒流注入 → 钾电导电导化）列入缺失机制清单（L28.2#4a）。

---

## 7. 未开工项（等主agent 安排）

P3 虚拟身体、P5–P8 涌现功能、P9 扰动、P10 活动金标准 —— **本批次未启动**。
