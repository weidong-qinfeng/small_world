"""M8 扰动/通路注入机制（P7 扰动预测 + P5 嗅觉 CS 通路注入共用）。

《生物仿真M8实施清单》§9.2（src/larva_perturb.py）：
- `PerturbLarvaCircuit`：LarvaCircuit 子类，把扰动/注入目标角色加入稀疏 stim 列
  （激活 tonic 注入需要目标角色列；冻结文件零修改——继承覆写 `_assign_stim_cols`）；
- `silence_role`：目标神经元**全部出边 gmax→0**（M6 L15 教训：掩码赋值静默
  no-op → **整体重建数组再赋值**）；
- `activate_role`：tonic 电流注入（写入 `_tonic_nA`，`LarvaSession.reset` 时经
  `_fill_tonic` 生效；不重建网络）；
- `run_spont_protocol`：无刺激自发协议（复刻 LarvaCircuit.run_spontaneous 循环），
  返回状态比例 + 肌肉通道均值 + 状态序列（确定性）；
- `classify_consequence`：行为后果类判定（预注册类集 + 阈值，
  与 data/m8_perturbation_plan.csv 阈值一致）。

确定性：全部固定 seed；同参数重跑逐位一致（p=1/n=1）。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.src.larva_circuit import (  # noqa: E402
    LarvaCircuit,
)
from neural_exploration.src.virtual_body import (  # noqa: E402
    VirtualBody,
    classify_state,
    state_fractions,
)

#: 行为后果类阈值（预注册，§9.1；与 gen_m8_perturbation_plan.py 一致）
CONSEQ_THRESH = dict(fwd=0.05, back=0.05, turn=0.05, pause=0.05, curl=0.05)
#: 后果类集（预注册）
CONSEQ_CLASSES = ("无变化", "前进↑", "前进↓", "转弯↑", "停驻↑", "蜷缩↑", "后退↑")


class PerturbLarvaCircuit(LarvaCircuit):
    """扩展稀疏 stim 列到指定角色（扰动激活 / CS 通路注入；形状定稿后不变）。

    `_assign_stim_cols` 覆写：在冻结基类并集（嗅觉对 + 伤害感受器 + 运动池 +
    张力）之上追加 `perturb_roles` 的角色列；零列索引随列数增长同步更新。
    编译缓存纪律：同一 `perturb_roles` 集合 → 同一形状 → 编译缓存命中。
    """

    def __init__(self, perturb_roles: Sequence[str] = (), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._perturb_roles = list(perturb_roles)

    def _assign_stim_cols(self) -> Tuple[Dict[str, int], int]:
        out, _ = super()._assign_stim_cols()
        for r in self._perturb_roles:
            if r in self.role_index and r not in out:
                out[r] = len(out)
        return out, len(out)


def silence_role(circ: LarvaCircuit, role: str) -> int:
    """沉默：目标神经元全部化学出边 gmax→0（整体重建数组，M6 L15）。

    返回被置零的出边数（含 ampa/gaba 两类；无 STDP——扰动用 plasticity='none'）。
    """
    from brian2 import meter, siemens

    pre_i = circ._idx_of(role, "node3")
    n_zeroed = 0
    for syn in circ.chem_synapses:
        i_arr = np.asarray(syn.i)
        mask = i_arr == pre_i
        if not mask.any():
            continue
        g = np.array(np.asarray(syn.gmax, dtype=float), dtype=float)
        g[mask] = 0.0
        syn.gmax = g * siemens / meter ** 2
        n_zeroed += int(mask.sum())
    return n_zeroed


def activate_role(circ: LarvaCircuit, role: str, nA: float) -> None:
    """激活：tonic 电流注入（reset 时经 _fill_tonic 写入 stim 列）。"""
    circ._tonic_nA[role] = float(nA)


def deactivate_role(circ: LarvaCircuit, role: str) -> None:
    """取消激活（移除 tonic 注入；下次 reset 不写入）。"""
    circ._tonic_nA.pop(role, None)


def run_spont_protocol(circ: LarvaCircuit, t_total_ms: float, seed: int,
                       silence_roles: Sequence[str] = (),
                       activate_roles: Sequence[str] = (),
                       act_nA: float = 0.5) -> dict:
    """无刺激自发协议：状态比例 + 肌肉通道均值 + 状态序列（确定性）。

    - silence_roles：这些角色出边 gmax→0（make_session 后、run 前应用——
      build 会重建突触对象，扰动须在会话内应用）；
    - activate_roles：tonic 注入（reset 后直接写 stim 列——_fill_tonic 已在
      reset 内运行，故不写 _tonic_nA 而直写列，语义同 run_escape）。

    Returns dict(frac, states, ch_mean, n_epochs, n_silenced, wall_s)。
    """
    body = VirtualBody(v_fwd0=circ.v_fwd0, v_rev0=circ.v_rev0,
                       omega_max=circ.omega_max, dt_b=circ.dt_b,
                       arena_L=circ.arena_L, boundary="reflect")
    sess = circ.make_session(t_total_ms=t_total_ms)
    sess.reset(seed=seed)
    n_silenced = 0
    for role in silence_roles:
        n_silenced += silence_role(circ, role)
    for role in activate_roles:
        col = circ._stim_cols.get(role)
        if col is not None:
            sess.stim.values[:, col] = act_nA * 1e-9
    n_epochs = max(1, int(round(t_total_ms / circ.dt_b)))
    ch_acc = dict(fwd=0.0, back=0.0, left=0.0, right=0.0, curl=0.0)
    states, vs, omegas = [], [], []
    for e in range(n_epochs):
        mus = sess.run_epoch(circ.dt_b, 0.0)
        c_fwd = float(mus.get("fwd", 0.0))
        c_back = float(mus.get("back", 0.0))
        c_left = float(mus.get("left", 0.0))
        c_right = float(mus.get("right", 0.0))
        c_curl = float(mus.get("curl", 0.0))
        ch_acc["fwd"] += c_fwd
        ch_acc["back"] += c_back
        ch_acc["left"] += c_left
        ch_acc["right"] += c_right
        ch_acc["curl"] += c_curl
        v = body.speed(c_fwd, c_back)
        omega = body.turn_rate(c_left, c_right, e * circ.dt_b)
        st = classify_state(v, omega, c_fwd, c_back,
                            v_fwd0=circ.v_fwd0, omega_max=circ.omega_max)
        body.step(c_fwd, c_back, c_left, c_right, circ.dt_b, e * circ.dt_b)
        states.append(st)
        vs.append(v)
        omegas.append(omega)
    m5 = state_fractions(states)
    # 幼虫语义（larva_loop 映射）：run=fwd、turn=turn+rev、pause=pause、curl=0
    frac = dict(run=m5.get("fwd", 0.0),
                turn=m5.get("turn", 0.0) + m5.get("rev", 0.0),
                pause=m5.get("pause", 0.0), curl=0.0)
    ch_mean = {k: v / max(1, n_epochs) for k, v in ch_acc.items()}
    return dict(frac=frac, states=states, ch_mean=ch_mean, n_epochs=n_epochs,
                n_silenced=n_silenced, m5_frac=m5)


def classify_consequence(delta: Dict[str, float],
                         thresh: Optional[dict] = None) -> str:
    """行为后果类判定（预注册类集 + 阈值；Δ = 扰动 − sham，同 seed）。

    优先顺序：|Δ| 最大的达标指标（back/curl 防御类优先）；全部不达标 → 无变化。
    阈值为比例差（frac）或通道均值差（ch_mean），由 delta 键前缀区分。
    """
    th = dict(CONSEQ_THRESH if thresh is None else thresh)
    cands = []
    for metric, cls in (("back", "后退↑"), ("curl", "蜷缩↑"),
                        ("fwd", "前进↑"), ("fwd", "前进↓"),
                        ("turn", "转弯↑"), ("pause", "停驻↑")):
        if metric not in delta:
            continue
        d = float(delta[metric])
        thr = th.get(metric, 0.05)
        if cls == "前进↓":
            ok = bool(d < -thr)
            mag = abs(d)
        elif cls == "蜷缩↑":
            # curl 通道在 provisional 肌肉映射不存在 → 结构性不可达（如实记录）
            ok = bool(d > thr)
            mag = abs(d)
        else:
            ok = bool(d > thr)
            mag = abs(d)
        if ok:
            cands.append((mag, cls))
    if not cands:
        return "无变化"
    cands.sort(key=lambda x: -x[0])
    return cands[0][1]
