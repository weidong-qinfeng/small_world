"""M6 神经调质池（`ModulatorPool`）+ M5 反证清单四项落地（组装层，冻结文件零修改）。

对应《生物仿真M6实施清单》§3（步骤 2：神经调质系统 + M5 反证清单落地，P2 验证对象）
与 §0 G1 机制门（M5 P2/P4/P6 复核 + 逃避方向相位复核）。

科学依据（docs/m5_env_notes.md L37–L41 反证笔记 = M6 优先验证清单）：
  - L40 #1：命令回路互抑缺失（AVA/AVD ↔ AVB/PVC 在真实连接组中全部互为兴奋——
    ampa 实测无互抑边；真实 C. elegans 经 RIM 酪胺能介导后退时抑制前进）；
  - L40 #2：AVA→DD/VD GABA 抑制链缺失（实测 AVA/AVD→DD/VD 化学边 0 条；现有
    DD/VD gaba 池 motor→motor 57 条无命令驱动）；
  - L40 #3：单一张力驱动夹带（M4 14µA/cm² AVB 张力是 302 唯一持续驱动，自发/
    调质缺失 → 86% 同步 2.7-13.8Hz → 静默上限 ~44%、fwd/back 共同发放 → v≈0）；
  - L40 #5：网络节律污染逃避方向（touch@50ms → back；touch@73ms（定稿 τ_trans=23）
    → not_back——方向相位敏感）。

本模块 = **组装层**实现（清单 §3.2：`neuromod.py` 在 `GroupedWormCircuit.build` 之后、
`make_session` 之前挂接；连接组是事实不动，m5_connectome.csv 内容零修改）：

  ① **RIM 酪胺**（`tyramine_enabled`）：AVA/AVD（后退命令）激活 → C_tyr↑（τ 数百 ms）
     → 门控抑制 AVB/PVC 前进驱动（C_fwd 通路增益 ↓）——兼作习惯化的生物机制
     （重复刺激 → 后退命令激活累积 → 酪胺抑制前进，Rankin 1990 习惯化一致）；
  ② **命令互抑**（`mutual_inh_enabled`）：AVA/AVD ↔ AVB/PVC 组装层功能互抑边
     （真实连接组无此边——功能补充，登记抽象；后退命令抑制前进命令 + 前进命令
     抑制后退命令 → 互斥方向分离）；
  ③ **AVA→DD/VD GABA 功能链**（`gaba_chain_enabled`）：AVA/AVD 激活 → DD/VD GABA
     池驱动 → 其既有 gaba→fwd 运动池突触抑制 fwd 池（真实连接组 0 条该边——
     功能补充）；
  ④ **自发/调质输入**（`spont_enabled`）：确定性伪随机自发输入（seed 固定，p=1/n=1
     纪律保持可重跑）→ 打破"单一张力驱动 AVB"的夹带；另含调质池基线浓度
     （C_da/C_5ht 网络驱动释放 + 基线）。

另含**多巴胺/血清素浓度模型**（§3.1 规格）：C_da/C_5ht/C_tyr ∈ [0,1] 全局状态变量，
ODE `dC/dt = (R − C)/τ`（exponential_euler，与点档同款；τ 预注册 100–1000ms），
调制各层增益（T2 横切层理念，门控函数单调有界）：
  - C_tyr → 前进命令增益 ↓（①）；
  - C_5ht（血清素，食物信号）→ 前进命令增益 ↑（capped 1.2）；
  - C_da（多巴胺，运动调制）→ 运动层增益 ↓（Hill 型 1/(1+K·C)）。
作用域只落在有真实递质语义的通路（§0 #10，不臆造受体）。

集成方式（组合复用，冻结文件零修改）：`ModulatedCircuit` 包装 M5 `GroupedWormCircuit`
（`make_session` 返回 `ModulatedGroupedSession`，接口与 `GroupedWormSession` 一致——
`WormLoop` 无需改动即可消费：run_trial/run_trials/run_escape/run_spontaneous 全部复用）；
`ModulatedCircuit.run_resting` 覆盖为调质会话路径（`WormLoop.run_resting` 委托点）。
每项机制 enabled 开关（消融 sanity：删项 → 现象消失）。

确定性：p=1/n=1；调质 ODE 无随机性；自发输入表由固定 seed 伪随机生成（会话内恒定，
试次间不重生成）；同参数重跑逐位一致（重跑逐位一致判据沿用）。

参数唯一定稿源：`data/m6_learning_params.csv`（mod 行 + mod_gating 行；
位置解析语义与 m5_worm_params.csv 一致：value 在 fields[9]，L23）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_M6_PARAMS_CSV = os.path.join(ROOT, "neural_exploration", "data",
                                     "m6_learning_params.csv")

# --------------------------------------------------------------------- #
# 角色池（连接组 roster 语义；与 data/m5_connectome.csv 神经元行一致）
# --------------------------------------------------------------------- #
#: 后退命令池（AVA/AVD，谷氨酸能后退命令中间神经元；L40 #1）
BACK_CMD = ("AVAL", "AVAR", "AVDL", "AVDR")
#: 前进命令池（AVB/PVC，胆碱能前进命令中间神经元；L40 #1）
FWD_CMD = ("AVBL", "AVBR", "PVCL", "PVCR")
#: 酪胺能 RIM（真实递质标注=other/酪胺；L9；连接组中无 g>0 输出——功能经调质池落地）
RIM_ROLES = ("RIML", "RIMR")
#: GABA 抑制池（DD/VD，motor 类 gaba；现有 gaba→运动池突触在连接组中，
#: 本模块补 AVA/AVD→DD/VD 的功能驱动链，L40 #2）
GABA_POOL = tuple(f"DD{k}" for k in range(1, 7)) + \
    tuple(f"VD{k}" for k in range(1, 14))
#: 前进运动池（DB/VB → muscle_fwd；AVA→DD/VD 链的抑制目标，L40 #2）
FWD_MOTOR = tuple(f"DB{k}" for k in range(1, 8)) + \
    tuple(f"VB{k}" for k in range(1, 12))
#: 后退运动池（DA/VA/AS → muscle_back）
BACK_MOTOR = tuple(f"DA{k}" for k in range(1, 10)) + \
    tuple(f"VA{k}" for k in range(1, 13)) + \
    tuple(f"AS{k}" for k in range(1, 12))
#: 多巴胺能释放源（8 神经元：ADE/CEP/PDE，L5#5/L9）
DA_SRC = ("ADEL", "ADER", "CEPDL", "CEPDR", "CEPVL", "CEPVR",
          "PDEL", "PDER")
#: 血清素能释放源（7 神经元：ADF/NSM/RIH；HSN 为性特异在 roster 内但不驱动肌肉，
#: 作释放源计入并登记抽象取舍，L5#2）
HT_SRC = ("ADFL", "ADFR", "NSML", "NSMR", "RIH", "HSNL", "HSNR")
#: 自发输入默认作用域（命令池 + RIM + GABA 池 + 头运动：打破夹带的核心子集 + 转向）
DEFAULT_SPONT_ROLES = BACK_CMD + FWD_CMD + RIM_ROLES + GABA_POOL + \
    ("SMDDL", "SMDDR", "SMBVL", "SMBVR", "RMDL", "RMDR")


# --------------------------------------------------------------------- #
# 参数（唯一定稿源 data/m6_learning_params.csv；None → 默认值）
# --------------------------------------------------------------------- #
@dataclass
class ModParams:
    """调质池参数（§3.1 规格 + §0 #10 作用域预注册；每项可消融）。"""

    # —— 总开关与机制开关（消融 sanity：enabled=False → 现象消失）——
    enabled: bool = True            # 总开关（False → 完全等价 M5 冻结行为）
    tyramine_enabled: bool = True   # ① RIM 酪胺门控（AVA→C_tyr→抑 AVB/PVC）
    mutual_inh_enabled: bool = True  # ② 命令互抑（AVA/AVD ↔ AVB/PVC）
    gaba_chain_enabled: bool = True  # ③ AVA→DD/VD GABA 功能链
    spont_enabled: bool = True      # ④ 自发/调质输入（确定性伪随机）
    da_enabled: bool = True         # 多巴胺浓度门控（运动层增益）
    ht_enabled: bool = True         # 血清素浓度门控（前进增益）

    # —— 浓度 ODE（§3.1：τ 预注册 100–1000ms；C ∈ [0,1]）——
    tau_da_ms: float = 500.0
    tau_5ht_ms: float = 500.0
    tau_tyr_ms: float = 500.0
    rate_norm_hz: float = 30.0      # 释放源发放率归一化基准（→[0,1]）
    da_baseline: float = 0.05       # 调质池基线浓度（L2④ 预注册二选一之"基线浓度"）
    ht_baseline: float = 0.05

    # —— ① RIM 酪胺 ——
    tyr_gain: float = 0.60          # fwd_gate = 1 − tyr_gain·C_tyr（单调有界）
    tyr_floor: float = 0.30         # fwd_gate 下限（不彻底关闭前进）
    tyr_baseline: float = 0.55      # 酪胺基线浓度（L2④"调质池基线浓度"选项：把
                                    # AVB 张力压到持续发放分岔点之下 → 打破单张力夹带；
                                    # 低于阈值 AVB 靠自发脉冲瞬态驱动 → 前进 bout）
    # —— ② 命令互抑 ——
    inh_gain_nA: float = 0.15       # 每归一化率的抑制电流（nA；对称互抑基准）
    inh_back_on_fwd_gain_nA: Optional[float] = None  # 后退→前进（RIM/酪胺语义，
                                    # 强；None → 用 inh_gain_nA 对称）
    inh_fwd_on_back_gain_nA: Optional[float] = None  # 前进→后退（弱；None → 对称）
    # —— ③ AVA→DD/VD GABA 链 ——
    gaba_chain_gain_nA: float = 0.15  # AVA/AVD 归一化率 → DD/VD 驱动电流（nA）
    # —— ④ 自发输入 ——
    spont_rate_hz: float = 2.0      # 每角色自发发放率（低率；fwd/rev/头 bout 驱动）
    spont_amp_nA: float = 0.15      # 每脉冲注入电流（nA；精确脉冲窗写入）
    spont_dur_ms: float = 3.0       # 脉冲时长（ms；瞬态 bout 驱动，不持续夹带）
    spont_roles: Tuple[str, ...] = DEFAULT_SPONT_ROLES
    spont_seed: int = 20260826      # 固定 seed（确定性伪随机，可重跑）
    # —— 多巴胺/血清素门控 ——
    da_gain: float = 0.30           # C_da → 运动层增益 1/(1+K·C)（Hill 型）
    ht_gain: float = 0.20           # C_5ht → 前进增益 1+K·C（capped 1.2）
    # —— 会话 ——
    chunk_ms: float = 25.0          # 静息/连续运行时的调质更新块（= 行为 tick）
    mod_dt_ms: Optional[float] = None  # 调质更新子步（None → 与 epoch 同步）；
                                    # 更细子步 → 互抑/酪胺反应延迟更短（逃避方向
                                    # 相位复核：25ms 单步滞后 → 触后首 25ms 内
                                    # fwd/back 共同发放无法被互抑抑制）


def _parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return bool(float(s))


def load_m6_mod_params(csv_path: Optional[str] = None) -> ModParams:
    """读 data/m6_learning_params.csv 的 mod 行（位置解析语义同 m5_worm_params：
    value 在 fields[9]，L23 惯例）；缺失键 → 默认值。"""
    path = csv_path or DEFAULT_M6_PARAMS_CSV
    p = ModParams()
    if not os.path.exists(path):
        return p
    rows: Dict[str, object] = {}
    with open(path, newline="", encoding="utf-8") as f:
        import csv as _csv
        for ln in f:
            s = ln.strip()
            if s.startswith('"'):
                s = s.strip('"')
            if not s or s.startswith("#"):
                continue
            fields = next(_csv.reader([s]))
            role = (fields[0] if fields else "").strip().lower()
            key = (fields[1] if len(fields) > 1 else "").strip().lower()
            if role != "mod" or not key:
                continue
            value = (fields[9] if len(fields) >= 11 else "")
            rows[key] = value
    if not rows:
        return p
    # dataclass 字段名（含 nA/camelCase）；CSV key 转小写后模糊匹配
    field_map = {f.lower(): f for f in p.__dataclass_fields__}
    for key, val in rows.items():
        attr = field_map.get(key)
        if attr is None:
            continue  # 未知键忽略（兼容后续步追加的 mod 参数）
        cur = getattr(p, attr)
        if isinstance(cur, bool):
            setattr(p, attr, _parse_bool(val))
        elif isinstance(cur, tuple):
            if isinstance(val, str) and val.strip():
                setattr(p, attr, tuple(r.strip().upper() for r in
                                       val.split("|") if r.strip()))
        elif cur is None:
            # Optional[float] 默认（如非对称互抑增益）：数值型字符串 → float
            try:
                setattr(p, attr, float(val))
            except (TypeError, ValueError):
                pass
        elif isinstance(cur, (int, float)):
            setattr(p, attr, float(val))
    return p


# --------------------------------------------------------------------- #
# ModulatorPool：浓度 ODE + 门控增益
# --------------------------------------------------------------------- #
class ModulatorPool:
    """调质浓度池（全局状态变量；确定性 ODE，无随机性）。

    C_da/C_5ht/C_tyr ∈ [0,1]；每步 `update(dt_ms, rates)`（rates = 逐角色发放率
    Hz）后计算门控增益（单调有界）供 `ModulatedGroupedSession` 写入门控电流。
    """

    def __init__(self, params: Optional[ModParams] = None):
        self.p = params or ModParams()
        self.C_da = 0.0
        self.C_5ht = 0.0
        self.C_tyr = 0.0
        self.last_rates: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def reset(self):
        """试次开始：浓度清零（基线由释放源/协议在运行中建立）。"""
        self.C_da = 0.0
        self.C_5ht = 0.0
        self.C_tyr = 0.0
        self.last_rates = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _norm(rate: float, norm_hz: float) -> float:
        return float(np.clip(rate / norm_hz, 0.0, 1.0)) if norm_hz > 0 else 0.0

    def _pool_rate(self, rates: Dict[str, float], roles: Sequence[str]
                   ) -> float:
        vals = [rates.get(r, 0.0) for r in roles if r in rates]
        return float(np.mean(vals)) if vals else 0.0

    # ------------------------------------------------------------------ #
    def update(self, dt_ms: float, rates: Dict[str, float]):
        """一步 ODE（exponential_euler）：C += (R − C)·dt/τ，clip 到 [0,1]。

        rates = 上一窗逐角色发放率（Hz）。释放源活性：
          - R_tyr = AVA/AVD 后退命令率（真实 RIM 接受 AVA/AVD 缝隙输入，L40 #1）；
          - R_da  = 多巴胺能神经元（ADE/CEP/PDE）网络发放率 + 基线；
          - R_5ht = 血清素能神经元（ADF/NSM/RIH）网络发放率 + 基线。
        """
        p = self.p
        self.last_rates = dict(rates)
        n_ava = self._norm(self._pool_rate(rates, BACK_CMD), p.rate_norm_hz)
        r_tyr = max(n_ava, p.tyr_baseline) if p.tyramine_enabled else 0.0
        r_da = self._norm(self._pool_rate(rates, DA_SRC), p.rate_norm_hz)
        r_ht = self._norm(self._pool_rate(rates, HT_SRC), p.rate_norm_hz)
        if not p.da_enabled:
            r_da = 0.0
        if not p.ht_enabled:
            r_ht = 0.0
        r_da = max(r_da, p.da_baseline)
        r_ht = max(r_ht, p.ht_baseline)
        dt = max(float(dt_ms), 0.0)

        def _step(C: float, R: float, tau_ms: float) -> float:
            if tau_ms <= 0:
                return float(np.clip(R, 0.0, 1.0))
            C = C + (R - C) * (dt / tau_ms)
            return float(np.clip(C, 0.0, 1.0))

        self.C_da = _step(self.C_da, r_da, p.tau_da_ms)
        self.C_5ht = _step(self.C_5ht, r_ht, p.tau_5ht_ms)
        self.C_tyr = _step(self.C_tyr, r_tyr, p.tau_tyr_ms)

    # ------------------------------------------------------------------ #
    # 门控增益（单调有界，§3.1：G ∈ [0,1] 或 capped 上限）
    # ------------------------------------------------------------------ #
    def fwd_gate(self) -> float:
        """前进命令增益：酪胺抑制 × 血清素促进 × 多巴胺运动层抑制。"""
        p = self.p
        if p.tyramine_enabled:
            g = 1.0 - p.tyr_gain * self.C_tyr
            g = float(np.clip(g, p.tyr_floor, 1.0))
        else:
            g = 1.0  # 消融：删酪胺门控 → 无门控（sanity）
        if p.ht_enabled:
            g *= float(np.clip(1.0 + p.ht_gain * self.C_5ht, 1.0, 1.2))
        if p.da_enabled:
            g *= 1.0 / (1.0 + p.da_gain * self.C_da)
        return float(np.clip(g, p.tyr_floor, 1.2))

    def back_inh_nA(self, fwd_rate: float) -> float:
        """前进命令率 → 后退命令池收到的抑制电流（nA；② 互抑反向边，弱）。"""
        if not self.p.mutual_inh_enabled:
            return 0.0
        g = (self.p.inh_fwd_on_back_gain_nA
             if self.p.inh_fwd_on_back_gain_nA is not None
             else self.p.inh_gain_nA)
        return g * self._norm(fwd_rate, self.p.rate_norm_hz)

    def fwd_inh_nA(self, back_rate: float) -> float:
        """后退命令率 → 前进命令池收到的抑制电流（nA；② 互抑正向边，RIM 语义强）。"""
        if not self.p.mutual_inh_enabled:
            return 0.0
        g = (self.p.inh_back_on_fwd_gain_nA
             if self.p.inh_back_on_fwd_gain_nA is not None
             else self.p.inh_gain_nA)
        return g * self._norm(back_rate, self.p.rate_norm_hz)

    def gaba_chain_nA(self, back_rate: float) -> float:
        """后退命令率 → DD/VD 驱动电流（nA；③ AVA→DD/VD GABA 功能链）。"""
        if not self.p.gaba_chain_enabled:
            return 0.0
        return self.p.gaba_chain_gain_nA * self._norm(
            back_rate, self.p.rate_norm_hz)


# --------------------------------------------------------------------- #
# ModulatedGroupedSession：M5 GroupedWormSession 的调质包装（同接口）
# --------------------------------------------------------------------- #
class ModulatedGroupedSession:
    """包装 `GroupedWormSession`：每 epoch 先算窗口发放率 → 调质 ODE 更新 →
    写入门控/互抑/GABA 链/自发输入电流（stim 列，amp）→ 委托底层 run_epoch。

    接口与 `GroupedWormSession` 一致（reset/run_epoch/run_resting_window/
    role_spike_times/any_spikes_in_window/muscle_read/finish/stim），
    `WormLoop` 不经修改直接消费（组合复用纪律）。
    """

    def __init__(self, mc: "ModulatedCircuit", sess):
        self.mc = mc
        self.sess = sess
        self.circuit = mc.circuit
        self.mod = mc.mod
        self.p = mc.mod.p
        self._t_last_ms = 0.0
        self._n_epochs = 0

    # ------------------------------------------------------------------ #
    def __getattr__(self, name):
        return getattr(self.sess, name)

    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None):
        self.sess.reset(seed)
        self.mod.reset()
        self._t_last_ms = 0.0
        self._n_epochs = 0

    # ------------------------------------------------------------------ #
    def _role_rates(self, t0_ms: float, t1_ms: float) -> Dict[str, float]:
        """[t0, t1) 窗内逐角色发放率（Hz；读底层 SpikeMonitor，确定性）。"""
        from brian2 import ms as bms

        c = self.circuit
        t = np.asarray(c._sp.t / bms)
        i = np.asarray(c._sp.i)
        win_s = max((t1_ms - t0_ms) / 1000.0, 1e-9)
        out: Dict[str, float] = {}
        if t.size:
            mask = (t >= t0_ms - 1e-9) & (t < t1_ms)
            idx = i[mask]
            for role, ri in c.role_index.items():
                out[role] = float(np.sum(idx == ri)) / win_s
        else:
            for role in c.role_index:
                out[role] = 0.0
        return out

    def _spont_nA(self, role: str, t0_ms: float, t1_ms: float) -> float:
        """确定性伪随机自发输入（④）：[t0,t1) 内活动脉冲数 × 幅度（nA）。

        兼容旧 epoch 平均语义（保留，实际写入走精确脉冲窗 `_write_spont_pulses`）。
        """
        if not self.p.spont_enabled:
            return 0.0
        times = self.mc._spont_table.get(role)
        if times is None or times.size == 0:
            return 0.0
        n = int(np.sum((times >= t0_ms - 1e-9) & (times < t1_ms)))
        return n * self.p.spont_amp_nA

    def _write_spont_pulses(self, t0_ms: float, t1_ms: float):
        """精确脉冲窗写入（④）：每脉冲在 stim 列 [tp, tp+dur) 加 amp（+=）。

        瞬态脉冲（dur≈3ms）而非持续注入 → 产生异步 bout 驱动而非持续夹带；
        确定性伪随机表固定 seed（会话内恒定，试次间不重生成）。
        """
        p = self.p
        if not p.spont_enabled:
            return
        dt = self.circuit.dt_ms
        dur = max(p.spont_dur_ms, dt)
        role_index = self.circuit.role_index
        values = self.sess.stim.values
        n_steps = values.shape[0]
        amp_amp = p.spont_amp_nA * 1e-9
        for role, times in self.mc._spont_table.items():
            idx = role_index.get(role)
            if idx is None or times.size == 0:
                continue
            sel = times[(times >= t0_ms - dur) & (times < t1_ms)]
            for tp in sel:
                p0 = int(round(max(t0_ms, tp) / dt))
                p1 = int(round(min(t1_ms, tp + dur) / dt))
                p0 = max(0, min(p0, n_steps))
                p1 = max(p0, min(p1, n_steps))
                if p1 > p0:
                    values[p0:p1, idx] += amp_amp

    # ------------------------------------------------------------------ #
    def _write_mod(self, dt_ms: float, t_now_ms: float):
        """调质门控写入门控/互抑/GABA 链/自发输入电流（stim 列，amp 单位）。

        - AVB/PVC（前进命令）：门控后张力（①③多巴胺/血清素/酪胺增益）− 后退
          互抑（②） + 自发（④）；
        - AVA/AVD（后退命令）：− 前进互抑（②） + 自发（④）；
        - DD/VD（GABA 池）：AVA/AVD 驱动（③）+ 自发（④）。
        """
        from brian2 import ms  # noqa: F401  (仅触发命名空间无关——值在 numpy 侧)

        p = self.p
        c = self.circuit
        if not p.enabled:
            return
        n_steps = self.sess.stim.values.shape[0]
        i0 = int(round(t_now_ms / c.dt_ms))
        i1 = int(round((t_now_ms + dt_ms) / c.dt_ms))
        i0 = max(0, min(i0, n_steps))
        i1 = max(i0, min(i1, n_steps))
        if i1 <= i0:
            return
        rates = self.mod.last_rates
        back_rate = self.mod._pool_rate(rates, BACK_CMD)
        fwd_rate = self.mod._pool_rate(rates, FWD_CMD)
        g_fwd = self.mod.fwd_gate()
        inh_back_on_fwd = self.mod.fwd_inh_nA(back_rate)
        inh_fwd_on_back = self.mod.back_inh_nA(fwd_rate)
        chain = self.mod.gaba_chain_nA(back_rate)
        role_index = c.role_index
        t1 = t_now_ms + dt_ms

        for role in FWD_CMD:
            idx = role_index.get(role)
            if idx is None:
                continue
            tonic = self.circuit._tonic_nA.get(role, 0.0)
            val = tonic * g_fwd - inh_back_on_fwd
            self.sess.stim.values[i0:i1, idx] = val * 1e-9
        for role in BACK_CMD:
            idx = role_index.get(role)
            if idx is None:
                continue
            val = -inh_fwd_on_back
            self.sess.stim.values[i0:i1, idx] = val * 1e-9
        if p.gaba_chain_enabled:
            for role in GABA_POOL:
                idx = role_index.get(role)
                if idx is None:
                    continue
                val = chain
                self.sess.stim.values[i0:i1, idx] = val * 1e-9
        # ④ 自发输入：精确脉冲窗（+=，叠加在门控/互抑基础上）
        self._write_spont_pulses(t_now_ms, t1)

    # ------------------------------------------------------------------ #
    def run_epoch(self, dt_ms: float, s_value: float) -> Dict[str, float]:
        from brian2 import ms

        if not self.p.enabled:
            self._t_last_ms = float(self.sess.net.t / ms)
            self._n_epochs += 1
            return self.sess.run_epoch(dt_ms, s_value)
        sub = self.p.mod_dt_ms
        if sub is None or sub >= dt_ms:
            return self._run_epoch_once(dt_ms, s_value)
        # 细粒度：把 25ms epoch 拆成 mod_dt_ms 子步（互抑/酪胺反应延迟更短）。
        # ASE 注入逻辑复制自 GroupedWormSession.run_epoch（组装层副本，冻结零修改）。
        c = self.circuit
        p = c.params
        tr = p.transduction
        n_sub = max(1, int(round(dt_ms / sub)))
        out = None
        for k in range(n_sub):
            t_now = float(self.sess.net.t / ms)
            rates = self._role_rates(self._t_last_ms, t_now)
            self.mod.update(sub, rates)
            self._write_mod(sub, t_now)
            # ASE 注入（本子步切片）
            i0 = int(round(t_now / p.dt_ms))
            i1 = int(round((t_now + sub) / p.dt_ms))
            on_role, off_role = c.sens_roles
            if on_role:
                i_on = tr.g_on * max(float(s_value), 0.0)
                idx = c.role_index[on_role]
                self.sess.stim.values[i0:i1, idx] = (
                    i_on * 1e-6 * 1.257e-5 * 1e9) * 1e-9
            if off_role:
                i_off = tr.g_off * max(-float(s_value), 0.0)
                idx = c.role_index[off_role]
                self.sess.stim.values[i0:i1, idx] = (
                    i_off * 1e-6 * 1.257e-5 * 1e9) * 1e-9
            self.sess.net.run(sub * ms, namespace=self.sess.ns)
            self._t_last_ms = t_now + sub
            out = self.circuit.muscle3.read()
        self._n_epochs += 1
        return out

    def _run_epoch_once(self, dt_ms: float, s_value: float) -> Dict[str, float]:
        from brian2 import ms

        t_now = float(self.sess.net.t / ms)
        rates = self._role_rates(self._t_last_ms, t_now)
        self.mod.update(dt_ms, rates)
        self._write_mod(dt_ms, t_now)
        out = self.sess.run_epoch(dt_ms, s_value)
        self._t_last_ms = t_now + float(dt_ms)
        self._n_epochs += 1
        return out

    def run_resting_window(self, t_total_ms: float):
        """连续运行（静息协议）：调质开启时按 chunk 分块更新（等价语义，
        调质关闭时直接委托底层单次 run——数值与 M5 冻结一致）。"""
        from brian2 import ms

        if not self.p.enabled:
            self.sess.run_resting_window(t_total_ms)
            self._n_epochs += 1
            return
        chunk = max(self.p.mod_dt_ms or self.p.chunk_ms, self.circuit.dt_ms)
        n = max(1, int(round(t_total_ms / chunk)))
        for k in range(n):
            t_now = float(self.sess.net.t / ms)
            rates = self._role_rates(self._t_last_ms, t_now)
            self.mod.update(chunk, rates)
            self._write_mod(chunk, t_now)
            self.sess.net.run(chunk * ms, namespace=self.sess.ns)
            self._t_last_ms = t_now + chunk
        self._n_epochs += 1


# --------------------------------------------------------------------- #
# ModulatedCircuit：M5 GroupedWormCircuit 的组装层包装（冻结文件零修改）
# --------------------------------------------------------------------- #
class ModulatedCircuit:
    """包装 M5 冻结 `GroupedWormCircuit`：`make_session` 返回调质会话；
    `run_resting` 覆盖为调质路径（`WormLoop.run_resting` 委托点）。
    其余属性/方法全部委托底层 circuit（组合复用纪律，未改任何冻结文件）。
    """

    def __init__(self, circuit, mod: Optional[ModulatorPool] = None,
                 params_csv: Optional[str] = None, spont_seed: Optional[int] = None):
        self.circuit = circuit
        self.mod = mod or ModulatorPool(load_m6_mod_params(params_csv))
        if spont_seed is not None:
            self.mod.p.spont_seed = int(spont_seed)
        self.params = circuit.params  # WormLoop 读取入口（委托语义）
        self._spont_table: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    def __getattr__(self, name):
        return getattr(self.circuit, name)

    # ------------------------------------------------------------------ #
    def make_session(self, t_total_ms: Optional[float] = None,
                     record: Optional[Sequence[str]] = None,
                     stimulated_roles: Optional[Sequence[str]] = None
                     ) -> ModulatedGroupedSession:
        sess = self.circuit.make_session(t_total_ms=t_total_ms,
                                         record=record,
                                         stimulated_roles=stimulated_roles)
        self._ensure_spont_table()
        return ModulatedGroupedSession(self, sess)

    # ------------------------------------------------------------------ #
    def _ensure_spont_table(self):
        """确定性伪随机自发输入表（固定 seed，会话内恒定；④）。

        每角色泊松发放时刻（ms，覆盖固定协议窗口）；幅度统一 spont_amp_nA
        （epoch 平均语义）。表一旦生成不再变 → 同参数重跑逐位一致。
        """
        from neural_exploration.src.worm_circuit import PROTOCOL_WINDOW_MS

        if self._spont_table:
            return
        p = self.mod.p
        rng = np.random.default_rng(int(p.spont_seed))
        roles = [r for r in p.spont_roles if r in self.circuit.role_index]
        n_ms = float(PROTOCOL_WINDOW_MS)
        for role in roles:
            n_pulses = int(rng.poisson(p.spont_rate_hz * (n_ms / 1000.0)))
            times = np.sort(rng.uniform(0.0, n_ms, n_pulses))
            self._spont_table[role] = times

    # ------------------------------------------------------------------ #
    def run_resting(self, t_total_ms: float = 1000.0,
                    seed: Optional[int] = None) -> dict:
        """静息协议（调质路径；语义与 GroupedWormCircuit.run_resting 一致：
        逐角色发放率 + 静默比例 + 稳定性检查）。"""
        from brian2 import ms as bms

        p = self.params
        seed = self.circuit.seed if seed is None else int(seed)
        sess = self.make_session(t_total_ms=t_total_ms)
        sess.reset(seed=seed)
        import time
        t0w = time.perf_counter()
        sess.run_resting_window(t_total_ms)
        wall = time.perf_counter() - t0w
        rates = {}
        t_end = sess.net.t / bms
        for role, times in sess.role_spike_times().items():
            rates[role] = float(len(times)) / (float(t_end) / 1000.0)
        arr = np.array(list(rates.values()), dtype=float)
        silent_frac = float(np.mean(arr < 0.5)) if arr.size else float("nan")
        return dict(
            rates_hz=rates, median_hz=float(np.median(arr)) if arr.size
            else float("nan"),
            max_hz=float(arr.max()) if arr.size else float("nan"),
            silent_frac=silent_frac, n_spiking=float(np.sum(arr >= 0.5)),
            has_nan=bool(np.any(~np.isfinite(arr))), wall_s=wall,
            t_total_ms=t_total_ms,
        )


def make_modulated_circuit(scale: int = 302, mod: Optional[ModulatorPool] = None,
                           params_csv: Optional[str] = None,
                           spont_seed: Optional[int] = None, **kw):
    """M6 组装层工厂：M5 `make_worm_circuit`（冻结）+ 调质包装（组合复用）。

    用法：``mc = make_modulated_circuit(scale=302, **load_weight_scales())`` →
    ``WormLoop(mc)``（run_trial/run_trials/run_escape/run_spontaneous/run_resting
    全部复用，仅 run_resting 走调质路径）。
    """
    from neural_exploration.src.worm_circuit import make_worm_circuit

    base = make_worm_circuit(scale=scale, **kw)
    return ModulatedCircuit(base, mod=mod, params_csv=params_csv,
                            spont_seed=spont_seed)


def apply_modulation(sess, mod: Optional[ModulatorPool] = None,
                     params_csv: Optional[str] = None):
    """挂接接口（清单 §3.2：`apply_modulation(sess)`）：把既有
    `GroupedWormSession` 包装为调质会话（已包装则幂等返回）。"""
    if isinstance(sess, ModulatedGroupedSession):
        return sess
    mc = ModulatedCircuit(sess.circuit, mod=mod, params_csv=params_csv)
    return ModulatedGroupedSession(mc, sess)
