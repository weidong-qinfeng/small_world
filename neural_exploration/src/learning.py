"""M6 学习协议运行器：习惯化（P3）+ 联想学习（P4）——`src/learning.py`。

对应《生物仿真M6实施清单》§4（步骤 3：习惯化协议）与 §5（步骤 4：联想学习协议）。
本模块 = 学习**协议运行器**（清单 §0 P3/P4 的实现基础；工具脚本
`tools/validate_p6_habituation.py` / `validate_p6_associative.py` 为后续验证
里程碑的消费入口）。协议参数唯一定稿源 = `data/m6_learning_params.csv`
（habituation / associative 段，本模块追加；mod/stdp 段不动——B1a/B1b 定稿）。

---

## P3 习惯化协议（Rankin et al. 1990 对照）

**母版 = M5 P5 逃避协议**（`WormLoop.run_escape` 同款）：T=150ms、触电流
I0=60µA/cm² 注入 PLM/ALM、注入窗 [t0+τ_trans, t0+τ_trans+dur]（定稿
t0=50ms、τ_trans=23ms → touch@73ms）、反应量 R(n) = D_peak(n) =
max(C_back − C_fwd)（响应窗内，escape 同款度量）。

**机制（清单 §4.2 H1）**：触觉通路突触的 **STP 抑制**（Tsodyks–Markram，
M2 已验证方程：u 利用度 / x 资源，u0≈0.6、τ_rec 400–1000ms 预注册）+ **RIM
酪胺调质**（AVA 后退命令激活累积 → 酪胺↑ → fwd 门控↓，B1a 已落地）。

**实测坑（M6-B1c L23+，本模块设计前置）**：
1. **302 O2 全网上 D_peak 非触诱发**：touch@73ms 与 no-touch 的 D_peak 几乎
   相同（+0.355 vs +0.357）——O2 配置（自发 bout 驱动 + AVA→DD 链）下
   150ms 窗内 max(C_back−C_fwd) 由自发动力学主导，触刺激仅向触觉神经元
   增加 ~1 个尖峰（ALM 本身静息态持续发放 20-40Hz）。→ 网络级触诱发反应
   不可干净测量（夹带干扰，G1 部分通过的结构性限制），如实记录。
2. **302 触觉通路化学突触 STP 不可观测**：ALM→命令（AVDR/PVCL/PVCR）ampa
   边不承载逃避反应（触诱发驱动经缝隙 PLML↔PVCL/PLMR↔PVCR/ALMR↔AVDR +
   网络回振）；且 ALM 静息持续发放使 STP x 在首个刺激内即耗竭（x→0.0066），
   对反应无影响（实测 STP 开/关 R(n) 逐位相同）。
3. **M3 反射子图（干净触诱发底物）上 STP 呈现二值坍缩**：PLM→AVM→DA 链
   是干净触诱发反应（无自发动力学），但命令中间神经元阈值使反应近二值
   （发放/不发放）；STP 耗竭在短 ISI（≤50ms）下使反应整体坍缩（0.42 →
   −0.18），长 ISI（≥100ms）下 x 恢复 → 无衰减。→ 习惯化机制在短 ISI
   可演示（冒烟），Rankin 10s-ISI 主协议受模型时程限制（τ_rec 数百 ms 在
   10s ISI 内完全恢复 → R(n) 常数，记录为测量限制，§0 #4 预注册）。

**协议运行器**：`HabituationLoop` 支持两种底物：
- `substrate="reflex"`（默认，机制底物）：M3 反射子图 + 学习层 STP 组装
  （`ReflexCircuit` 冻结类 + `_m2` 变异启用 M2 STP——未改任何冻结文件）；
- `substrate="network"`（302 O2 全网）：`ModulatedCircuit`，R(n) 可计算
  （确定性），夹带限制如实入档。
重复刺激 → 逐刺激 R(n) 序列 + 指数拟合（R(n)=A·exp(−n/τ_hab)+B，确定性
lstsq）+ 恢复窗（休息后测试反应 R_rest）+ 消融开关（stp_enabled / tyramine）。

---

## P4 联想学习协议（盐+食物关联；可逆）

**范式（清单 §5.2 方案①，ASE 通路）**：CS = 盐浓度梯度（`ChemotaxisEnv`
闭环 + ASE 时间差分编码，M4 已验证接口）；US = 食物/血清素信号 = 调质池
协议注入（训练期 M(t)=+1 / 消退期 M(t)=−1 / 测试期 M(t)=0）。机制 = 三因子
规则（`plasticity.ThreeFactorSynapse` 同款方程，B1b 定稿）在 **ASE→AIY/AIB
子图**（8 条连接组事实边）开启：delig/dt = −elig/τ_e；dw/dt = η·M(t)·elig。
训练（CS-US 配对）→ 权重↑ → 盐趋化指数 CI_salt↑；消退（US 撤除/反号）→
权重↓ → CI_salt 回落；η=0 消融 → 无获得。

**底物**：连接组 20-role M4 趋化子图（grouped point；M5 G0 验证 CI 方向
+0.403@5s）。G1 复核 302 全网趋化未缓解（CĪ=−0.263，夹带）→ 按 §0 预注册
#1c「子图学习」路径（网络级学习行为反证记录）。

**协议运行器**：`AssociativeLearningLoop`——基线 CI（配对试次，确定性种子
起点抖动）→ 训练（连续，固定 US 窗 CS-US 配对）→ 训练后 CI → 消退（US
反号）→ 消退后 CI → η=0 消融对照；权重/CI 全量入档 + 确定性重跑逐位一致。

确定性铁律：p=1/n=1；自发输入/试次起点/转向方向全部固定 seed 伪随机
（`np.random.default_rng(seed)`）；同参数重跑逐位一致。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.worm_circuit import (  # noqa: E402
    PROTOCOL_WINDOW_MS, load_weight_scales, make_worm_circuit,
)
from neural_exploration.src.worm_loop import WormLoop  # noqa: E402

DEFAULT_LEARNING_PARAMS_CSV = os.path.join(
    ROOT, "neural_exploration", "data", "m6_learning_params.csv")

#: 触刺激角色（P5/习惯化；连接组 PLM*/ALM*）
TOUCH_ROLES_DEFAULT = ("PLML", "PLMR", "ALML", "ALMR")
#: 反射子图触角色（M3 spec：PLM）
REFLEX_TOUCH_ROLES = ("PLM",)
#: P4 三因子作用域（ASE→AIY/AIB，连接组事实 g=5nS/delay=0.5ms）
TF_EDGES = (("ASEL", "AIYL"), ("ASEL", "AIYR"), ("ASEL", "AIBL"),
            ("ASEL", "AIBR"), ("ASER", "AIYL"), ("ASER", "AIYR"),
            ("ASER", "AIBL"), ("ASER", "AIBR"))


# --------------------------------------------------------------------- #
# 参数（data/m6_learning_params.csv habituation / associative 段；唯一定稿源）
# --------------------------------------------------------------------- #
@dataclass
class LearningParams:
    """学习协议参数（habituation + associative 段；缺失键 → 默认值）。"""

    # —— 习惯化（P3；母版 = M5 P5 逃避协议）——
    n_stim: int = 20                 # 刺激数（预注册 20–30）
    isi_ms: float = 10000.0          # 刺激间静息窗（Rankin 1990 主协议 10s）
    t_stim_ms: float = 150.0         # 单刺激窗（= M5 escape t_total）
    touch_start_ms: float = 50.0     # 触刺激开始（M5 escape 同款）
    touch_dur_ms: float = 5.0        # 触刺激时长
    touch_i0_uA_cm2: float = 60.0    # 触刺激密度
    touch_tau_trans_ms: float = 23.0  # 转导延迟 τ_trans（G1 定稿）
    touch_roles: Tuple[str, ...] = TOUCH_ROLES_DEFAULT
    rest_ms: float = 0.0             # 恢复窗（自发恢复协议；0 = 不测）
    d_peak_thr: float = 0.3          # 方向 sanity 阈值（escape.direction_peak）
    # —— STP（H1 机制；M2 Tsodyks–Markram 同款）——
    stp_enabled: bool = True         # 习惯化 STP 开关（消融：False → 无衰减）
    stp_u0: float = 0.6              # 基准利用度（预注册 u0≈0.6）
    stp_tau_fac_ms: float = 10.0     # 易化时间常数（depression-dominant）
    stp_tau_rec_ms: float = 1000.0   # 恢复时间常数（预注册 400–1000ms）
    # —— 指数拟合（§0 预注册 #3）——
    fit_tau_band: Tuple[float, float] = (3.0, 15.0)   # τ_hab 预注册带（次）
    fit_r2_min: float = 0.5          # R² 预注册阈值
    recover_frac_min: float = 0.3    # R_rest ≥ 0.3×R(1) 预注册
    fit_seed: int = 0                # 拟合确定性 seed
    # —— 联想学习（P4；ASE→AIY/AIB 三因子）——
    eta: float = 1e-2                # 三因子训练学习率（预注册窗 [1e-4,1e-2] 上界）
    tau_e_ms: float = 200.0          # 资格迹时间常数（stdp 段同值）
    assoc_scale: int = 20            # 联想学习底物规模（M4 趋化子图；
                                     # G1 后 302 趋化未缓解 → 子图学习 §0 #1c）
    tf_edges: Tuple[Tuple[str, str], ...] = TF_EDGES
    n_test: int = 4                  # 测试试次数（基线/训练后/消退后；配对种子）
    t_test_ms: float = 1500.0        # 测试试次时长
    t_train_ms: float = 8000.0       # 训练期时长（连续；CS-US 配对）
    t_ext_ms: float = 8000.0         # 消退期时长（连续；US 撤除/反号）
    us_period_ms: float = 400.0      # US 窗周期（配对训练期重复）
    us_on_ms: float = 200.0          # US 窗时长（每周期 [us_on, period) 注入）
    us_train_signal: float = 1.0     # 训练期 US 信号（C_5ht 协议注入）
    us_ext_signal: float = -1.0      # 消退期 US 信号（US 反号 → 权重回落）
    start_jitter: float = 0.3        # 试次起点抖动（M5 protocol 同款）
    seed_base: int = 0               # 试次种子基（确定性伪随机）

    # —— 消融（P3 机制归属）——
    tyramine_enabled: bool = True    # RIM 酪胺开关（B1a O2 定稿；见 L23 限制）


def _parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return bool(float(s))


def load_learning_params(csv_path: Optional[str] = None) -> LearningParams:
    """读 data/m6_learning_params.csv 的 habituation / associative 段。

    位置解析语义同 m5_worm_params.csv（value 在 fields[9]，L23 惯例）；
    未知键忽略（兼容后续步骤追加）。
    """
    path = csv_path or DEFAULT_LEARNING_PARAMS_CSV
    p = LearningParams()
    if not os.path.exists(path):
        return p
    import csv as _csv

    rows: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if s.startswith('"'):
                s = s.strip('"')
            if not s or s.startswith("#"):
                continue
            fields = next(_csv.reader([s]))
            role = (fields[0] if fields else "").strip().lower()
            key = (fields[1] if len(fields) > 1 else "").strip().lower()
            if role not in ("habituation", "associative") or not key:
                continue
            value = (fields[9] if len(fields) >= 11 else "")
            rows[f"{role}.{key}"] = value
    if not rows:
        return p
    field_map = {f.lower(): f for f in p.__dataclass_fields__}
    for key, val in rows.items():
        attr = field_map.get(key.split(".", 1)[1])
        if attr is None:
            continue
        cur = getattr(p, attr)
        if isinstance(cur, bool):
            setattr(p, attr, _parse_bool(val))
        elif isinstance(cur, tuple):
            if isinstance(val, str) and val.strip():
                if attr == "tf_edges":
                    setattr(p, attr, tuple(
                        tuple(x.split("->")) for x in
                        val.split("|") if x.strip()))
                else:
                    items = [x.strip() for x in val.split("|") if x.strip()]
                    # "3.0..15.0" 带格式 → 数值对
                    if len(items) == 1 and ".." in items[0]:
                        items = [x for x in items[0].split("..") if x.strip()]
                    setattr(p, attr, tuple(
                        float(x) if _is_numeric(x) else x for x in items))
        elif isinstance(cur, int):
            try:
                setattr(p, attr, int(float(val)))
            except (TypeError, ValueError):
                pass
        elif isinstance(cur, float):
            try:
                setattr(p, attr, float(val))
            except (TypeError, ValueError):
                pass
    return p


def _is_numeric(s: str) -> bool:
    try:
        float(str(s).strip())
        return True
    except (TypeError, ValueError):
        return False


def write_learning_params_csv(path: str, p: Optional[LearningParams] = None,
                              append: bool = True) -> None:
    """写 habituation / associative 段到 m6_learning_params.csv（追加，段互不
    冲突；mod/stdp 段保留）。验证脚本生成母版；冒烟直接读 CSV 定稿值。"""
    p = p or LearningParams()
    lines = [
        "# ---- 学习协议参数（M6-B1c 追加；习惯化 P3 + 联想学习 P4；唯一定稿源）----",
        "# 行语义同 mod 行（value 在 fields[9]，位置解析；role=habituation|associative）",
        "# 预注册（清单 §0）：#3 τ_hab∈[3,15] 次/R²≥0.5；#4 恢复用相对判据 R_rest≥0.3×R(1)；",
        "#   §4.1 母版=M5 P5 逃避协议（T=150ms/touch@73ms τ_trans=23/PLM·ALM 60µA/cm²/D_peak）；",
        "#   §4.2 H1 STP u0≈0.6、τ_rec 400–1000ms + RIM 酪胺；§2.2 三因子 η[1e-4,1e-2]/τ_e[100,500]ms",
        "role,neuron_class,synapse_from,synapse_to,synapse_type,g_max_ns,delay_ms,tonic_uA_cm2,value,note",
    ]
    rows = [
        # 习惯化（P3）
        ("habituation", "n_stim", p.n_stim, "刺激数（Rankin 1990 量级；预注册 20–30）"),
        ("habituation", "isi_ms", p.isi_ms, "刺激间静息窗（Rankin 主协议 10s；短 ISI 展示机制）"),
        ("habituation", "t_stim_ms", p.t_stim_ms, "单刺激窗（= M5 escape t_total）"),
        ("habituation", "touch_start_ms", p.touch_start_ms, "触刺激开始（M5 escape 同款）"),
        ("habituation", "touch_dur_ms", p.touch_dur_ms, "触刺激时长"),
        ("habituation", "touch_i0_uA_cm2", p.touch_i0_uA_cm2, "触刺激密度"),
        ("habituation", "touch_tau_trans_ms", p.touch_tau_trans_ms, "转导延迟 τ_trans（G1 定稿 23ms → touch@73ms）"),
        ("habituation", "touch_roles", "|".join(p.touch_roles), "触刺激角色（escape 同款前缀）"),
        ("habituation", "rest_ms", p.rest_ms, "恢复窗（自发恢复协议；0=不测）"),
        ("habituation", "d_peak_thr", p.d_peak_thr, "方向 sanity 阈值（escape.direction_peak）"),
        ("habituation", "stp_enabled", int(p.stp_enabled), "STP 开关（H1 机制；消融：关 → 无衰减）"),
        ("habituation", "stp_u0", p.stp_u0, "STP 基准利用度（M2 Tsodyks–Markram；预注册 u0≈0.6）"),
        ("habituation", "stp_tau_fac_ms", p.stp_tau_fac_ms, "STP 易化时间常数（depression-dominant）"),
        ("habituation", "stp_tau_rec_ms", p.stp_tau_rec_ms, "STP 恢复时间常数（预注册 400–1000ms）"),
        ("habituation", "fit_tau_band", "3.0..15.0", "τ_hab 预注册带（次；§0 #3）"),
        ("habituation", "fit_r2_min", p.fit_r2_min, "指数拟合 R² 预注册阈值"),
        ("habituation", "recover_frac_min", p.recover_frac_min, "R_rest ≥ 0.3×R(1) 预注册"),
        ("habituation", "fit_seed", p.fit_seed, "拟合确定性 seed"),
        # 联想学习（P4）
        ("associative", "eta", p.eta, "三因子训练学习率（预注册窗 [1e-4,1e-2] 上界）"),
        ("associative", "tau_e_ms", p.tau_e_ms, "资格迹时间常数（stdp 段同值）"),
        ("associative", "assoc_scale", p.assoc_scale, "联想学习底物规模（M4 趋化子图；G1 后 302 趋化未缓解 → 子图学习 §0 #1c）"),
        ("associative", "tf_edges", "|".join(f"{a}->{b}" for a, b in p.tf_edges),
         "三因子作用域（ASE→AIY/AIB；连接组事实 g=5nS）"),
        ("associative", "n_test", p.n_test, "测试试次数（基线/训练后/消退后；配对种子）"),
        ("associative", "t_test_ms", p.t_test_ms, "测试试次时长"),
        ("associative", "t_train_ms", p.t_train_ms, "训练期时长（连续；CS-US 配对）"),
        ("associative", "t_ext_ms", p.t_ext_ms, "消退期时长（连续；US 撤除/反号）"),
        ("associative", "us_period_ms", p.us_period_ms, "US 窗周期（配对训练期重复）"),
        ("associative", "us_on_ms", p.us_on_ms, "US 窗时长（每周期 [us_on,period) 注入）"),
        ("associative", "us_train_signal", p.us_train_signal, "训练期 US 信号（C_5ht 协议注入）"),
        ("associative", "us_ext_signal", p.us_ext_signal, "消退期 US 信号（US 反号 → 权重回落）"),
        ("associative", "start_jitter", p.start_jitter, "试次起点抖动（M5 protocol 同款）"),
        ("associative", "seed_base", p.seed_base, "试次种子基（确定性伪随机）"),
    ]
    for role, key, val, note in rows:
        lines.append(f"{role},{key},,,,,,,,{val},{note}")
    existing = []
    if os.path.exists(path) and append:
        with open(path, newline="", encoding="utf-8") as f:
            for line in f:
                l2 = line.strip()
                if l2.startswith(("habituation,", "associative,")):
                    continue
                existing.append(line.rstrip("\n"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        if existing:
            f.write("\n".join(existing) + "\n")


# --------------------------------------------------------------------- #
# 指数拟合（§0 预注册 #3：确定性 lstsq，seed=0）
# --------------------------------------------------------------------- #
def fit_exponential(r_seq: Sequence[float], seed: int = 0,
                    tau_band: Tuple[float, float] = (3.0, 15.0),
                    r2_min: float = 0.5) -> Dict[str, float]:
    """R(n) = A·exp(−n/τ_hab) + B 拟合（n=1..N 索引）。

    确定性：无随机性（numpy lstsq 固定初值网格，最优解确定性）；
    拟合失败（退化/发散）→ A=τ_hab=nan 且 in_band=False（如实记录，不重试）。
    """
    y = np.asarray(r_seq, dtype=float)
    n = np.arange(1.0, y.size + 1.0)
    best = None
    for tau0 in np.linspace(1.0, 30.0, 30):
        X = np.column_stack([np.exp(-n / tau0), np.ones_like(n)])
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        A, B = coef[0], coef[1]
        yhat = X @ coef
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) if y.size > 1 else 0.0
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        if best is None or (r2 == r2 and (best[2] != best[2] or r2 > best[2])):
            best = (A, tau0, r2, B)
    A, tau, r2, B = best if best is not None else (float("nan"), float("nan"),
                                                   float("nan"), float("nan"))
    return dict(
        A=float(A), tau_hab=float(tau), B=float(B), r2=float(r2),
        in_tau_band=bool(tau_band[0] <= tau <= tau_band[1]) if tau == tau else False,
        r2_ok=bool(r2 >= r2_min) if r2 == r2 else False,
        fit_ok=bool(A > 0 and tau == tau and r2 == r2),
    )


# --------------------------------------------------------------------- #
# 习惯化会话（反射子图底物：干净触诱发反应 + 学习层 STP）
# --------------------------------------------------------------------- #
class HabSessionReflex:
    """M3 反射子图 + STP 触觉通路的习惯化会话（组件模式；确定性）。

    组装：`ReflexCircuit`（冻结类）在 build 前经 `_m2` 变异启用 M2 STP
    （`ChemicalSynapse` 内置 Tsodyks–Markram，u0/τ_fac/τ_rec 定稿于 CSV
    learning 段）——未改任何冻结文件（M6 组装层纪律）。
    会话接口：run_stimulus() → D_peak；run_rest(t_ms)；finish()。
    """

    def __init__(self, params: Optional[LearningParams] = None,
                 stp_enabled: Optional[bool] = None, seed: int = 0):
        from neural_exploration.src.worm_circuit import ReflexCircuit

        p = params or load_learning_params()
        self.p = p
        self.seed = int(seed)
        self.circ = ReflexCircuit(fidelity="point", seed=self.seed)
        if (p.stp_enabled if stp_enabled is None else bool(stp_enabled)):
            self.circ._m2["ampa"].u0 = p.stp_u0
            # M2 ChemicalSynapse 的 stp_enabled 判据 = τ_fac>0 且 τ_rec>0；
            # CSV 若被误写为 0（并发写者）→ 回退 10ms（depression-dominant），
            # 避免静默禁用 STP（L23 记录）。
            tau_fac = p.stp_tau_fac_ms if p.stp_tau_fac_ms > 0 else 10.0
            self.circ._m2["ampa"].tau_fac_ms = tau_fac
            self.circ._m2["ampa"].tau_rec_ms = max(p.stp_tau_rec_ms, 1.0)
        self.circ.build()
        self._build_net()
        self._t_ms = 0.0

    # ------------------------------------------------------------------ #
    def _build_net(self):
        from neural_exploration.src.brian_env import configure_brian2
        from brian2 import (Network, SpikeMonitor, TimedArray, amp, ms, nA,
                            seed as bseed)

        configure_brian2()
        circ = self.circ
        spec = circ.spec
        dt = circ.dt_ms
        self._dt = dt
        t_total = PROTOCOL_WINDOW_MS
        n_steps = int(round(max(500.0, t_total) / dt))
        stims = {}
        tonic = dict(circ._tonic_uA_cm2)
        for role in spec.roles:
            n_comp = int(circ.neurons[role].neuron.N)
            arr = np.zeros((n_steps, n_comp)) * amp
            if role == "VB" and role in tonic:
                k = circ.neurons[role].density_to_nA(1.0)
                arr[:, 0] = tonic[role] * k * nA
            stims[role] = TimedArray(arr, dt=dt * ms, name=f"stim_{role.lower()}")
        self._stims = stims
        ns = {f"stim_{r.lower()}": ta for r, ta in stims.items()}
        spmons = {r: SpikeMonitor(n.neuron, "v", name=f"sp_{r.lower()}")
                  for r, n in circ.neurons.items()}
        mons_mus = circ.muscle3.monitor(dt, name="m6_hab_musc")
        net = Network()
        for n in circ.neurons.values():
            net.add(n.neuron)
        for cs in circ.chemicals:
            net.add(cs.synapses)
        for g in circ.muscle3.groups:
            net.add(g)
        for s in circ.muscle3.drivers:
            net.add(s)
        for sp in spmons.values():
            net.add(sp)
        for mm in mons_mus:
            net.add(mm)
        bseed(self.seed)
        self._net = net
        self._ns = ns
        self._spmons = spmons
        self._mons_mus = mons_mus

    # ------------------------------------------------------------------ #
    @property
    def touch_role(self) -> str:
        return self.circ.touch_role

    def touch_window(self, t_offset_ms: float = 0.0) -> Tuple[int, int]:
        """触注入窗（dt 索引）：[t0+τ_trans, t0+τ_trans+dur]（M5 escape 母版）。"""
        p = self.p
        t0 = t_offset_ms + p.touch_start_ms + p.touch_tau_trans_ms
        i0 = int(round(t0 / self._dt))
        i1 = int(round((t0 + p.touch_dur_ms) / self._dt))
        return i0, i1

    def run_stimulus(self, t_stim_ms: Optional[float] = None) -> float:
        """运行一个刺激窗（T=150ms；触注入窗内写 PLM 电流）→ D_peak。"""
        p = self.p
        t_stim = float(t_stim_ms or p.t_stim_ms)
        ta = self._stims[self.touch_role]
        from brian2 import ms, nA

        i_nA = self.circ.neurons[self.touch_role].density_to_nA(p.touch_i0_uA_cm2)
        i0, i1 = self.touch_window(self._t_ms)
        i1 = max(i0, min(i1, ta.values.shape[0]))
        ta.values[i0:i1, :] = i_nA * nA
        t0 = self._t_ms
        self._net.run(t_stim * ms, namespace=self._ns)
        self._t_ms = t0 + t_stim
        ta.values[i0:i1, :] = 0.0
        mus = {}
        for i, ch in enumerate(self.circ.muscle3.channels):
            mus[ch] = np.array(getattr(self._mons_mus[i], f"c_{ch}")[0])
        i0m = int(round(t0 / self._dt))
        i1m = int(round((t0 + t_stim) / self._dt))
        seg = slice(i0m, max(i0m + 1, i1m))
        c_back = mus.get("back", np.zeros(1))[seg]
        c_fwd = mus.get("fwd", np.zeros_like(c_back))[seg]
        return float(np.max(c_back - c_fwd)) if c_back.size else 0.0

    def run_rest(self, t_ms: float):
        from brian2 import ms

        if t_ms > 0:
            self._net.run(t_ms * ms, namespace=self._ns)
            self._t_ms += float(t_ms)


# --------------------------------------------------------------------- #
# 习惯化会话（302 O2 全网底物：ModulatedCircuit）
# --------------------------------------------------------------------- #
class HabSessionNetwork:
    """302 O2 配置习惯化会话（`ModulatedCircuit`；R(n) 确定性可计算）。

    ⚠ 实测限制（L23）：O2 全网上 D_peak 由自发动力学主导（touch≈no-touch，
    +0.355 vs +0.357）——触诱发反应不可干净分离；R(n) 可计算并如实入档
    （夹带干扰记录，不静默）。
    """

    def __init__(self, params: Optional[LearningParams] = None,
                 stp_enabled: Optional[bool] = None, seed: int = 0,
                 mod_dt_ms: float = 5.0):
        from neural_exploration.src.neuromod import (
            ModulatorPool, load_m6_mod_params, make_modulated_circuit,
        )

        p = params or load_learning_params()
        self.p = p
        self.seed = int(seed)
        mp = load_m6_mod_params()
        mp.mod_dt_ms = mod_dt_ms
        if not p.tyramine_enabled:
            mp.tyramine_enabled = False
        self.mc = make_modulated_circuit(scale=302, seed=seed,
                                         mod=ModulatorPool(mp),
                                         **load_weight_scales())
        self.wl = WormLoop(self.mc)
        self.wl.touch["tau_trans_ms"] = p.touch_tau_trans_ms
        self._t_ms = 0.0
        self._sess = None
        self._stp_syn = None

    def _ensure_session(self):
        if self._sess is not None:
            return
        p = self.p
        self._sess = self.mc.make_session(t_total_ms=PROTOCOL_WINDOW_MS)
        self._sess.reset(seed=self.seed)
        self._t_ms = 0.0

    def run_stimulus(self, t_stim_ms: Optional[float] = None) -> float:
        from brian2 import ms

        p = self.p
        t_stim = float(t_stim_ms or p.t_stim_ms)
        self._ensure_session()
        sess = self._sess
        dt_b = self.wl.body.dt_b
        stim = sess.sess.stim
        i_nA = p.touch_i0_uA_cm2 * 1e-6 * 1.257e-5 * 1e9
        i0 = int(round((self._t_ms + p.touch_start_ms + p.touch_tau_trans_ms)
                       / self.mc.circuit.dt_ms))
        i1 = int(round((self._t_ms + p.touch_start_ms + p.touch_tau_trans_ms
                        + p.touch_dur_ms) / self.mc.circuit.dt_ms))
        i0 = max(0, min(i0, stim.values.shape[0]))
        i1 = max(i0, min(i1, stim.values.shape[0]))
        for role in p.touch_roles:
            idx = self.mc.circuit.role_index.get(role)
            if idx is not None:
                stim.values[i0:i1, idx] = i_nA * 1e-9
        cbs, cfs = [], []
        for _e in range(max(1, int(round(t_stim / dt_b)))):
            mus = sess.run_epoch(dt_b, 0.0)
            cbs.append(float(mus.get("back", 0.0)))
            cfs.append(float(mus.get("fwd", 0.0)))
        self._t_ms += t_stim
        stim.values[i0:i1] = 0.0
        return float(np.max(np.asarray(cbs) - np.asarray(cfs)))

    def run_rest(self, t_ms: float):
        from brian2 import ms

        self._ensure_session()
        if t_ms > 0:
            self._sess.sess.net.run(t_ms * ms, namespace=self._sess.ns)
            self._t_ms += float(t_ms)

    def no_touch_d_peak(self, t_stim_ms: Optional[float] = None) -> float:
        """无触对照（L23 限制记录：touch vs no-touch D_peak 对照）。"""
        p = self.p
        t_stim = float(t_stim_ms or p.t_stim_ms)
        self._ensure_session()
        sess = self._sess
        dt_b = self.wl.body.dt_b
        cbs, cfs = [], []
        for _e in range(max(1, int(round(t_stim / dt_b)))):
            mus = sess.run_epoch(dt_b, 0.0)
            cbs.append(float(mus.get("back", 0.0)))
            cfs.append(float(mus.get("fwd", 0.0)))
        self._t_ms += t_stim
        return float(np.max(np.asarray(cbs) - np.asarray(cfs)))


# --------------------------------------------------------------------- #
# HabituationLoop：习惯化协议运行器（P3）
# --------------------------------------------------------------------- #
class HabituationLoop:
    """习惯化协议运行器（母版 = M5 P5 逃避协议；逐刺激 R(n) = D_peak）。

    用法::

        loop = HabituationLoop(substrate="reflex")      # 机制底物（默认）
        res = loop.run(n_stim=6, isi_ms=0.0, stp_enabled=True)
        # res["r_seq"]=R(n)；res["fit"]=指数拟合；res["r_rest"]=恢复反应

    协议（CSV learning 段定稿）：N 刺激 × (T=150ms 刺激窗 + ISI 静息) →
    逐刺激 D_peak 序列 R(n)；可选恢复窗（rest_ms）→ 测试反应 R_rest；
    stp_enabled=False 消融（无衰减）；tyramine_enabled=False 经 ModParams
    消融（302 底物；O2 基线饱和限制见 L23）。
    """

    def __init__(self, params: Optional[LearningParams] = None,
                 params_csv: Optional[str] = None,
                 substrate: str = "reflex"):
        self.p = params or load_learning_params(params_csv)
        self.substrate = substrate
        if substrate not in ("reflex", "network"):
            raise ValueError(f"substrate 需为 reflex/network：{substrate}")

    def _make_session(self, stp_enabled: Optional[bool], seed: int):
        if self.substrate == "reflex":
            return HabSessionReflex(self.p, stp_enabled=stp_enabled, seed=seed)
        return HabSessionNetwork(self.p, stp_enabled=stp_enabled, seed=seed)

    # ------------------------------------------------------------------ #
    def run(self, n_stim: Optional[int] = None, isi_ms: Optional[float] = None,
            t_stim_ms: Optional[float] = None, seed: int = 0,
            stp_enabled: Optional[bool] = None, rest_ms: Optional[float] = None,
            no_touch_control: bool = False) -> Dict[str, object]:
        """运行习惯化协议。

        Returns dict(r_seq, t_ms, fit, r_rest, direction_ok, wall_s,
        substrate, no_touch_d_peak, measured_limitations)。
        """
        import time

        p = self.p
        n = int(n_stim or p.n_stim)
        isi = float(p.isi_ms if isi_ms is None else isi_ms)
        t_stim = float(p.t_stim_ms if t_stim_ms is None else t_stim_ms)
        rest = float(p.rest_ms if rest_ms is None else rest_ms)
        t0 = time.perf_counter()
        sess = self._make_session(stp_enabled=stp_enabled, seed=int(seed))
        r_seq = []
        for k in range(n):
            d = sess.run_stimulus(t_stim)
            r_seq.append(d)
            if isi > 0:
                sess.run_rest(isi)
        r_rest = float("nan")
        if rest > 0:
            sess.run_rest(rest)
            r_rest = sess.run_stimulus(t_stim)
        fit = fit_exponential(r_seq, seed=p.fit_seed,
                              tau_band=p.fit_tau_band, r2_min=p.fit_r2_min)
        no_touch = None
        if no_touch_control and isinstance(sess, HabSessionNetwork):
            no_touch = sess.no_touch_d_peak(t_stim)
        r_arr = np.asarray(r_seq, dtype=float)
        return dict(
            r_seq=[float(x) for x in r_seq],
            t_ms=[float(k * (t_stim + isi)) for k in range(n)],
            fit=fit,
            r_rest=float(r_rest),
            direction_ok=bool(len(r_seq) and r_seq[0] > p.d_peak_thr),
            decay=float(r_arr[0] - r_arr[-1]) if r_arr.size > 1 else 0.0,
            first_half_mean=float(np.mean(r_arr[:max(1, r_arr.size // 2)]))
            if r_arr.size else float("nan"),
            last_half_mean=float(np.mean(r_arr[r_arr.size // 2:]))
            if r_arr.size else float("nan"),
            substrate=self.substrate,
            no_touch_d_peak=no_touch,
            wall_s=time.perf_counter() - t0,
            n_stim=n, isi_ms=isi, t_stim_ms=t_stim, rest_ms=rest,
            seed=int(seed), stp_enabled=(p.stp_enabled if stp_enabled is None
                                         else bool(stp_enabled)),
            measured_limitations=[
                "302 O2 全网 D_peak 由自发动力学主导（touch≈no-touch）→ 网络级"
                "触诱发反应不可干净测量（G1 部分通过结构性限制）",
                "反射子图 STP 呈现二值坍缩（命令阈值）；Rankin 10s-ISI 主协议受"
                "模型时程限制（τ_rec 数百 ms 在 10s ISI 内完全恢复 → R(n) 常数，"
                "§0 #4 预注册；机制在短 ISI 可演示）",
            ],
        )


# --------------------------------------------------------------------- #
# AssociativeLearningLoop：联想学习协议运行器（P4）
# --------------------------------------------------------------------- #
class AssociativeLearningLoop:
    """联想学习协议运行器（CS=盐梯度/ASE 编码；US=调质池协议注入；三因子
    门控在 ASE→AIY/AIB 子图；CI_salt 行为读出）。

    底物 = 连接组 20-role M4 趋化子图（grouped point；M5 G0 CI 方向
    +0.403@5s；G1 后 302 趋化未缓解 → §0 #1c 子图学习路径）。

    协议（CSV associative 段定稿）：
      基线 N_test 试次（配对种子，起点抖动）→ CI_pre；
      训练 T_train 连续（盐梯度 + 固定 US 窗 CS-US 配对）→ w↑；
      训练后 N_test 试次 → CI_post；
      消退 T_ext 连续（US 反号 → w↓）→ 消退后 N_test 试次 → CI_ext；
      η=0 消融（权重不变 → 无获得）。

    确定性：p=1/n=1；试次起点/转向方向固定 seed 伪随机；同参数重跑逐位一致。
    """

    def __init__(self, params: Optional[LearningParams] = None,
                 params_csv: Optional[str] = None, eta: Optional[float] = None,
                 scale: Optional[int] = None, seed: int = 0):
        p = params or load_learning_params(params_csv)
        self.p = p
        self.eta = p.eta if eta is None else float(eta)
        self.scale = int(scale or p.assoc_scale)
        self.seed = int(seed)
        self.circuit = make_worm_circuit(scale=self.scale, seed=self.seed,
                                         **load_weight_scales())
        # ⚠ B1c2 修复：circuit.params.env 是 EnvSpec（无 sample/ci）——须构建
        # ChemotaxisEnv（WormLoop 同款；M4 冻结语义，冻结文件零修改）
        from neural_exploration.src.chemotaxis_env import ChemotaxisEnv
        _pe = self.circuit.params.env
        self.env = ChemotaxisEnv(arena_L=_pe.arena_L, sigma=_pe.sigma,
                                 c_max=_pe.c_max, c_bg=_pe.c_bg,
                                 food_x=_pe.food_x, food_y=_pe.food_y,
                                 boundary=_pe.boundary)
        self._wl = WormLoop(self.circuit)
        self.body = self._wl.body
        self._sess = None
        self._syn = None
        self._mvals = None
        self._n_steps = 0
        self._built = False

    # ------------------------------------------------------------------ #
    def _build(self):
        from brian2 import Synapses, TimedArray, meter, ms, siemens

        if self._built:
            return
        p = self.p
        circ = self.circuit
        sess = circ.make_session(t_total_ms=PROTOCOL_WINDOW_MS)
        sess.reset(seed=self.seed)
        syn_ampa = next(s for s in circ.chem_synapses
                        if getattr(s, "name", "").endswith("_chem_ampa"))
        i_arr = np.asarray(syn_ampa.i)
        j_arr = np.asarray(syn_ampa.j)
        role_index = circ.role_index
        mask = np.zeros(i_arr.size, dtype=bool)
        pre_i, post_i = [], []
        for pre, post in p.tf_edges:
            if pre not in role_index or post not in role_index:
                continue
            m = (i_arr == role_index[pre]) & (j_arr == role_index[post])
            if m.any():
                mask |= m
                pre_i.append(role_index[pre])
                post_i.append(role_index[post])
        if not mask.any():
            raise ValueError(f"三因子目标边缺失（{p.tf_edges} 不在 {self.scale}"
                             f"-role 子图）")
        self._tf_mask = mask
        # ⚠ 实测坑（L23）：`syn.gmax[bool_mask] = 0.0` 对 Quantity VariableView
        # 静默 no-op（逐位不变）——必须整体重建数组再赋值才生效。三因子装配 =
        # 原边 gmax 置 0 + 新建 w 缩放突触（w=1.0 时 gmax=5nS 与连接组事实等价，
        # 回归保护：首跑行为与原始网络一致）。
        _g = np.array(np.asarray(syn_ampa.gmax, dtype=float), dtype=float)
        _g[mask] = 0.0
        syn_ampa.gmax = _g * siemens / meter ** 2
        n_steps = sess.stim.values.shape[0]
        self._n_steps = n_steps
        m_arr = np.zeros(n_steps, dtype=float)
        m_ta = TimedArray(m_arr, dt=circ.dt_ms * ms, name="m6_tf_M")
        g_density = 5.0 * 1e-9 / (1.257e-5 * 1e-4)
        ns = {"TAU_E": p.tau_e_ms * ms, "ETA": self.eta, "WMAX": 2.0,
              "GMAXD": g_density * siemens / meter ** 2, "M_t": m_ta}
        model = (
            "delig/dt = -elig/TAU_E : 1 (clock-driven)\n"
            "dw/dt = ETA*M_t(t)*elig/second : 1 (clock-driven)\n")
        on_pre = ("elig = elig + 1\n"
                  "w = clip(w, 0.0, WMAX)\n"
                  "g_ampa_post = g_ampa_post + GMAXD*w")
        on_post = "elig = elig + 1"
        syn = Synapses(circ.group, circ.group, model=model, on_pre=on_pre,
                       on_post=on_post, namespace=ns, name="m6_assoc_tf_ase")
        syn.connect(i=np.array(pre_i, dtype=np.int32),
                    j=np.array(post_i, dtype=np.int32))
        syn.delay = 0.5 * ms
        syn.w = 1.0
        syn.elig = 0.0
        sess.net.add(syn)
        sess.net.store()
        self._sess = sess
        self._syn = syn
        self._mvals = np.asarray(m_ta.values)
        self._built = True

    # ------------------------------------------------------------------ #
    def weights(self) -> np.ndarray:
        self._build()
        return np.array(self._syn.w)

    def set_weights(self, w: np.ndarray):
        self._build()
        self._syn.w[:] = np.asarray(w, dtype=float)

    def _trial(self, seed: int, t_total_ms: float, us_sign: float = 0.0,
               t0_ms: float = 0.0, us_mode: str = "fixed") -> Tuple[float, np.ndarray]:
        """闭环趋化单试次（M4 `_session_trial` 语义含机制 A 转向；确定性）。

        us_mode：
          "fixed" — 固定 US 窗（每 us_period_ms 周期 [us_on, period) 注入 ±1；
                    CS-US 配对 = 盐梯度在场 + 周期性食物信号（协议简化登记）；
          "s_up"  — 条件 US（CS-US 配对更严格）：仅当 s>0（升盐梯度/接近盐源）
                    注入 ±1——训练期强化接近通路（ASEL/ON），消退期反号。
        """
        p = self.p
        circ = self.circuit
        sess = self._sess
        body = self.body
        env = self.env
        dt_b = body.dt_b
        rng = np.random.default_rng(seed)
        sx = 5.0 + rng.normal(0.0, p.start_jitter)
        sy = 5.0 + rng.normal(0.0, p.start_jitter)
        th0 = rng.uniform(0.0, 2.0 * np.pi)
        self._mvals[:] = 0.0
        sess.reset(seed=seed)
        # ⚠ B1c2 修复（相位时钟漂移）：store/restore 后网络时钟 ≠ 0——
        # 训练/消退试次的 US 协议注入必须写**绝对**网络时间索引，否则 M 全 0、
        # 三因子 dw/dt=η·M·elig 恒 0（实测：基线后 store 快照 t=1000，
        # 训练窗 [1000,4000] 读 M[1000..]=0 → 无 LTP，B1c2 L13）。
        from brian2 import ms
        t_net0 = float(sess.net.t / ms)
        body.reset(sx, sy, th0)
        mech = circ.params.mech_a
        turn_rng = np.random.default_rng(seed)
        from neural_exploration.src.chemotaxis_env import TimeDiffTracker

        tracker = TimeDiffTracker(circ.params.transduction.tau_win_ms,
                                  env.sample(sx, sy))
        n_epochs = max(1, int(round(t_total_ms / dt_b)))
        xs, ys = [], []
        for e in range(n_epochs):
            t_e = e * dt_b
            c_now = env.sample(body.x, body.y)
            s = tracker.s_at(t_e, c_now)
            if mech.enabled and not body.is_turning():
                if s < -mech.theta_pir and sess.any_spikes_in_window(
                        ("SMDDL", "SMDDR"), t_e, t_e + dt_b):
                    direction = 1.0 if turn_rng.random() < 0.5 else -1.0
                    body.trigger_turn(direction, mech.omega_pir,
                                      mech.t_pir_ms)
            if us_sign != 0.0:
                us_on = False
                if us_mode == "fixed" and p.us_period_ms > 0:
                    us_on = (t0_ms + t_e) % p.us_period_ms >= p.us_on_ms
                elif us_mode == "s_up":
                    us_on = s > 0.0
                if us_on:
                    i0 = int(round((t_net0 + t_e) / circ.dt_ms))
                    i1 = int(round((t_net0 + t_e + dt_b) / circ.dt_ms))
                    self._mvals[max(0, i0):min(self._n_steps, i1)] = us_sign
            mus = sess.run_epoch(dt_b, s)
            body.step(float(mus.get("fwd", 0.0)), float(mus.get("back", 0.0)),
                      float(mus.get("left", 0.0)), float(mus.get("right", 0.0)),
                      dt_b, t_e)
            xs.append(body.x)
            ys.append(body.y)
        xa = np.array(xs, dtype=float)
        ya = np.array(ys, dtype=float)
        env.assert_bounded(xa, ya)
        body.assert_trajectory(xa, ya)
        return float(env.ci_per_trial(xa, ya)), np.array(self._syn.w)

    # ------------------------------------------------------------------ #
    def run(self, n_test: Optional[int] = None, t_test_ms: Optional[float] = None,
            t_train_ms: Optional[float] = None, t_ext_ms: Optional[float] = None,
            seed_base: Optional[int] = None, with_extinction: bool = True,
            with_eta0: bool = True, us_mode: str = "fixed") -> Dict[str, object]:
        """运行完整联想学习协议。

        us_mode（CS-US 配对语义）："fixed"=固定 US 窗（盐梯度在场 + 周期性
        食物信号）；"s_up"=条件 US（仅 s>0 升盐梯度时注入——配对更严格，
        训练期强化接近通路）。

        Returns dict(ci_pre/ci_post/ci_ext/ci_eta0, w_pre/w_post/w_ext/w_eta0,
        ci_stats*, acquisition_ok, extinction_ok, eta0_ok, wall_s)。
        """
        import time

        p = self.p
        n_test = int(n_test or p.n_test)
        t_test = float(p.t_test_ms if t_test_ms is None else t_test_ms)
        t_train = float(p.t_train_ms if t_train_ms is None else t_train_ms)
        t_ext = float(p.t_ext_ms if t_ext_ms is None else t_ext_ms)
        sbase = int(p.seed_base if seed_base is None else seed_base)
        t0 = time.perf_counter()
        self._build()

        # 基线（M=0；配对种子）
        ci_pre, w_pre = [], None
        for t in range(n_test):
            ci, w = self._trial(sbase + t, t_test, 0.0)
            ci_pre.append(ci)
            w_pre = w.copy()
        w0 = w_pre.copy()

        # 训练（连续；CS-US 配对 → w↑）
        self.set_weights(w0)
        self._sess.net.store()
        _, w_tr = self._trial(sbase + 99, t_train, p.us_train_signal,
                              us_mode=us_mode)
        dw_train = float(np.max(w_tr - w0))
        self._sess.net.store()
        ci_post = []
        for t in range(n_test):
            ci, w = self._trial(sbase + t, t_test, 0.0)
            ci_post.append(ci)

        # 消退（连续；US 撤除/反号 → w↓）
        ci_ext, w_ext, dw_ext = [], None, 0.0
        if with_extinction:
            _, w_ext = self._trial(sbase + 99, t_ext, p.us_ext_signal,
                                   us_mode=us_mode)
            # 消退量 = 训练后权重均值的回落（负 = 权重下降，可逆性判据）
            dw_ext = float(np.mean(w_ext) - np.mean(w_tr))
            self._sess.net.store()
            for t in range(n_test):
                ci, w = self._trial(sbase + t, t_test, 0.0)
                ci_ext.append(ci)

        # η=0 消融（权重不变 → 无获得）
        ci_eta0 = []
        if with_eta0:
            loop0 = AssociativeLearningLoop(self.p, eta=0.0,
                                            scale=self.scale, seed=self.seed)
            loop0.run(n_test=n_test, t_test_ms=t_test, t_train_ms=t_train,
                      t_ext_ms=t_ext, seed_base=sbase, with_extinction=False,
                      with_eta0=False, us_mode=us_mode)
            ci_eta0 = loop0.last_ci()

        out = dict(
            ci_pre=[float(x) for x in ci_pre],
            ci_post=[float(x) for x in ci_post],
            ci_ext=[float(x) for x in ci_ext] if ci_ext else None,
            ci_eta0=[float(x) for x in ci_eta0] if ci_eta0 else None,
            w_pre=w0.tolist(), w_tr=w_tr.tolist(),
            w_ext=w_ext.tolist() if w_ext is not None else None,
            dw_train=float(dw_train), dw_ext=float(dw_ext),
            mean_ci_pre=float(np.mean(ci_pre)) if ci_pre else float("nan"),
            mean_ci_post=float(np.mean(ci_post)) if ci_post else float("nan"),
            mean_ci_ext=(float(np.mean(ci_ext)) if ci_ext else float("nan")),
            mean_ci_eta0=(float(np.mean(ci_eta0)) if ci_eta0 else float("nan")),
            acquisition_ok=bool(np.mean(ci_post) > np.mean(ci_pre)),
            extinction_ok=bool(ci_ext and np.mean(ci_ext) < np.mean(ci_post)),
            eta0_ok=bool(ci_eta0 and abs(np.mean(ci_eta0) - np.mean(ci_pre))
                         < 0.05),
            n_test=n_test, t_test_ms=t_test, t_train_ms=t_train,
            t_ext_ms=t_ext, seed_base=sbase, eta=self.eta, us_mode=us_mode,
            wall_s=time.perf_counter() - t0,
        )
        self._last = out
        return out

    def last_ci(self) -> List[float]:
        if getattr(self, "_last", None):
            return list(self._last["ci_post"])
        return []
