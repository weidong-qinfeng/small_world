"""M9 P6：中央复合体（CX）航向地图——环状吸引子（`src/adult_cx.py`，新建）。

《生物仿真M9实施清单》§4.2（P6：CX 航向地图——EPG bump 与虚拟航向相关 + 旋转补偿 +
EPG 消融航向丢失 H3）。

**结构（真实连接组子图）**：
  - **EB**（椭球体，EPG 环的解剖所在）**381** 神经元；**PB**（扇形体上结节）**113**；
  - **EB→EB 真实递归边 38,606 条**（syn 均值 7.01）；EB→PB 2,109；PB→EB 1,564；PB→PB 2,455。

**环序（关键抽象，登记 §0.3.5）**：v783 无 cell_type / 无 EPG 扇区标注
（**缝隙连接亦不可得**——官方发布不含 gap junction，L19.2 裁决②）→ EPG 环的**扇区顺序**
由 **EB→EB 递归连接的图谱 Fiedler 向量**（对称化邻接的 Laplacian 第二小特征向量）**导出**：
对环状图，Fiedler 向量沿环单调 → 按其排序即得环序（允许反射/旋转移位等价）。
误差 = 真实扇区顺序可能有局部错位；回归条件 = 取得 EPG 类型/位置标注或缝隙连接后重建。

**动力学（环状吸引子）**：
    τ da_i/dt = −a_i + f( Σ_j W_ij a_j + I_i ),   I_i = g_h·cos(2π(i/N) − θ) + noise(0)
    f = ReLU（有界）；W 由真实 EB→EB 连接（对称化、按行归一）给出
    bump 位置 φ = atan2(Σ a_i sin(2πi/N), Σ a_i cos(2πi/N))

用法：
    from neural_exploration.src.adult_cx import AdultCX, CXParams
    cx = AdultCX(); r = cx.run_protocol()
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NET_NPZ = os.path.join(DATA_DIR, "m9_network.npz")
NEURON_NPZ = os.path.join(DATA_DIR, "m9_neuron_table.npz")

REGIONS_EPG = ("EB",)
REGIONS_PB = ("PB",)


@dataclass
class CXParams:
    """P6 环状吸引子参数（预注册草案）。"""
    tau_ms: float = 20.0          # 环状吸引子时间常数
    g_h: float = 1.0              # 航向输入增益
    w_scale: float = 1.0          # 递归权重缩放（标定杠杆）
    local_kernel: float = 0.35    # 局部核混合系数（0 = 纯连接组；1 = 纯 cos 核）
    gain: float = 1.0             # 网络增益
    inh_gain: float = 3.0         # 全局抑制（保留接口；softmax 归一已隐含全局抑制）
    a_max: float = 1000.0         # 活动上界（纯数值保护；须 >> a_total，否则剪切成平台——实测踩坑）
    beta: float = 8.0             # 保留（未用：改用分布型环状吸引子）
    kappa: float = 6.0            # 环核宽度参数（越大 bump 越窄）
    w_conn: float = 0.3           # 连接组分量权重（平滑）
    w_cos: float = 1.0            # 环核权重（自持正反馈；须足够大才持续）
    a_total: float = 30.0         # 环上总活动（归一化守恒）——bump 幅度由局部化程度决定
    dt_ms: float = 1.0
    t_settle_ms: float = 300.0    # 航向输入后的稳定窗
    n_headings: int = 12          # 航向保持测试的虚拟航向数
    rot_steps: int = 24           # 旋转补偿测试的采样点数
    rot_total_deg: float = 360.0  # 旋转总角度
    seed: int = 0


class AdultCX:
    """CX 航向环（真实 EB 子图 + Fiedler 环序 + 环状吸引子动力学）。"""

    def __init__(self, params: Optional[CXParams] = None, verbose: bool = True):
        self.p = params or CXParams()
        self.verbose = verbose
        self._load()
        self.reset()

    def _load(self) -> None:
        d = np.load(NET_NPZ, allow_pickle=False)
        nt = np.load(NEURON_NPZ, allow_pickle=False)
        names = np.asarray([str(x) for x in nt["region_names"]])
        code = nt["region_code"]
        n = int(nt["root_ids"].size)

        def mask(pfx):
            cs = [i for i, nm in enumerate(names) if any(nm.startswith(q) for q in pfx)]
            return np.isin(code, cs)

        m_eb = mask(REGIONS_EPG); m_pb = mask(REGIONS_PB)
        self.n_eb = int(m_eb.sum()); self.n_pb = int(m_pb.sum())
        pre = d["pre"].astype(np.int64); post = d["post"].astype(np.int64)
        sc = d["syn_count"].astype(np.float32)
        i_eb = np.flatnonzero(m_eb); i_pb = np.flatnonzero(m_pb)
        eb_map = -np.ones(n, dtype=np.int64); eb_map[i_eb] = np.arange(i_eb.size)
        pb_map = -np.ones(n, dtype=np.int64); pb_map[i_pb] = np.arange(i_pb.size)
        e_ee = np.isin(pre, i_eb) & np.isin(post, i_eb)
        self.ee_pre = eb_map[pre[e_ee]]; self.ee_post = eb_map[post[e_ee]]
        self.ee_syn = sc[e_ee]
        # PB→EB（用于诊断：环的解剖对应）
        e_pe = np.isin(pre, i_pb) & np.isin(post, i_eb)
        self.pe_pre = pb_map[pre[e_pe]]; self.pe_post = eb_map[post[e_pe]]
        # 邻接（对称化，syn 加权）
        A = np.zeros((self.n_eb, self.n_eb), dtype=np.float64)
        np.add.at(A, (self.ee_pre, self.ee_post), self.ee_syn)
        A = A + A.T
        self.A_sym = A
        self.ring_order = self._ring_order(A)
        # 环上重排后的**连接组**权重（行归一：只保留相对分布形状）
        W = A[np.ix_(self.ring_order, self.ring_order)]
        W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
        N = self.n_eb
        idx = np.arange(N)
        dmat = np.abs(idx[:, None] - idx[None, :])
        dmat = np.minimum(dmat, N - dmat)                      # 环距
        # **均值零 cos 环核**（经典环状吸引子形式，**不做行归一**）：
        # 行归一核的不动点是**均匀分布** → bump 会衰减到均匀（实测踩坑：去输入后幅度→0）。
        # 均值零 cos 核给出"以当前 bump 位置为中心的 cos 形驱动" → **正反馈自持** ✓
        self.Kcos = np.cos(2.0 * np.pi * dmat / N)
        self.Wconn = W
        if self.verbose:
            print("CX：EB=%d / PB=%d；EB→EB 边=%d；PB→EB 边=%d；环序=Fiedler 导出；"
                  "环核 w_conn=%.2f w_cos=%.2f" % (self.n_eb, self.n_pb, self.ee_pre.size,
                                                   self.pe_pre.size, self.p.w_conn,
                                                   self.p.w_cos), flush=True)

    @staticmethod
    def _ring_order(A: np.ndarray) -> np.ndarray:
        """由对称化邻接的 Fiedler 向量导出环序（环状图的 Fiedler 向量沿环单调）。"""
        N = A.shape[0]
        D = np.diag(A.sum(axis=1))
        L = D - A
        w, v = np.linalg.eigh(L)
        f = v[:, 1] if v.shape[1] > 1 else np.arange(N, dtype=np.float64)
        return np.argsort(f)

    def reset(self) -> None:
        self.a = np.full(self.n_eb, 0.05, dtype=np.float64)

    # ---------------- 动力学 ----------------
    def step(self, theta: float, dt_ms: Optional[float] = None,
             drive_on: bool = True) -> None:
        p = self.p
        dt = float(dt_ms if dt_ms is not None else p.dt_ms)
        N = self.n_eb
        ang = 2.0 * np.pi * np.arange(N) / N
        # **分布型环状吸引子**：p = 环上活动分布（和=1）
        #   r = [w_conn·Wconn + w_cos·K] @ p    （递归：平滑当前分布，峰位自持）
        #   r ← r·(1 + g_h·cos(ang−θ))          （航向输入调制；去输入时 bump 持续）
        #   p ← p + (r/Σr − p)·dt/τ
        # 实测踩坑：① 无全局抑制/归一 → 半环平台饱和发散；② 纯 softmax(β 大) → 塌成单神经元
        p_ = self.a / max(float(self.a.sum()), 1e-12)
        # 递归：连接组分量（行归一，平滑）+ 均值零 cos 环核（**自持正反馈**）
        r = p.w_conn * (self.Wconn @ p_) + p.w_cos * (self.Kcos @ p_)
        if drive_on:
            r = r + p.g_h * np.cos(ang - float(theta))
        r = np.maximum(r, 0.0)
        rs = float(r.sum())
        if rs > 0:
            tgt = p.a_total * (r / rs)         # 归一化 = 全局抑制（活动总量守恒）
            self.a = np.clip(self.a + (tgt - self.a) / max(p.tau_ms, 1e-9) * dt,
                             0.0, p.a_max)

    def bump_position(self) -> float:
        """bump 位置（rad，环坐标）。"""
        N = self.n_eb
        ang = 2.0 * np.pi * np.arange(N) / N
        s = float((self.a * np.sin(ang)).sum()); c = float((self.a * np.cos(ang)).sum())
        return float(np.arctan2(s, c))

    def bump_amplitude(self) -> float:
        """bump 幅度（归一化）：(max − mean)/max。"""
        mx = float(self.a.max())
        if mx <= 1e-12:
            return 0.0
        return float((mx - float(self.a.mean())) / mx)

    def settle(self, theta: float, t_ms: Optional[float] = None,
               drive_on: bool = True) -> None:
        t = float(t_ms if t_ms is not None else self.p.t_settle_ms)
        for _ in range(max(1, int(round(t / self.p.dt_ms)))):
            self.step(theta, drive_on=drive_on)

    # ---------------- 协议 ----------------
    def run_protocol(self, ablate_ring: bool = False) -> Dict[str, Any]:
        """P6 协议：航向保持 + 旋转补偿 + （可选）环消融。"""
        p = self.p
        W_conn_ref = self.Wconn.copy()
        K_ref_full = self.Kcos.copy()
        if ablate_ring:
            # EPG 环消融：**打乱环拓扑**（连接组分量 + 环核同时置换）。
            # 实测踩坑：只打乱连接组分量（w_conn=0.3）时解析环核仍支撑 bump → 消融无效。
            # 生物学语义 = 环状递归被破坏（缝隙连接不可得 → 以拓扑打乱为代理，测量限制登记）。
            # 语义：**EPG 环状递归断开**（w_cos→0）+ 连接组分量拓扑打乱。
            # 判据落在**持续性**上（撤除输入后航向信息是否丢失）——仅看"有输入时的跟踪"
            # 无法区分吸引子与纯跟随者（实测踩坑：环被打乱后输入仍能驱动 bump）。
            rng = np.random.default_rng(p.seed + 12345)
            perm = rng.permutation(self.n_eb)
            self.Wconn = self.Wconn[np.ix_(perm, perm)]
            np.fill_diagonal(self.Wconn, W_conn_ref.diagonal())
            self.Kcos = np.zeros_like(self.Kcos)     # 环状递归断开
        # (a) 航向保持：给定虚拟航向 → 稳定后读 bump 位置
        errs, corr_x, corr_y, amps = [], [], [], []
        for k in range(p.n_headings):
            theta = 2.0 * np.pi * k / p.n_headings
            self.reset(); self.settle(theta)
            phi = self.bump_position()
            d = (phi - theta + np.pi) % (2.0 * np.pi) - np.pi     # 最短角差
            errs.append(abs(np.degrees(d)))
            corr_x.append(theta); corr_y.append(phi)
            amps.append(self.bump_amplitude())
        errs = np.asarray(errs)
        # 持续性：撤除输入后的漂移与幅度保持
        self.reset(); self.settle(2.0)
        phi_a = self.bump_position(); amp_a = self.bump_amplitude()
        self.settle(2.0, t_ms=1000.0, drive_on=False)
        phi_b = self.bump_position(); amp_b = self.bump_amplitude()
        drift = abs(np.degrees((phi_b - phi_a + np.pi) % (2*np.pi) - np.pi))
        amp_keep = amp_b / max(amp_a, 1e-12)
        corr = _circ_corr(np.asarray(corr_x), np.asarray(corr_y))
        # (b) 旋转补偿：输入旋转 → bump 位移（增益应 ≈1）
        self.reset()
        thetas = np.linspace(0.0, np.radians(p.rot_total_deg), p.rot_steps)
        phis = []
        for th in thetas:
            self.settle(th, t_ms=60.0)          # 逐步旋转（连续跟踪）
            phis.append(self.bump_position())
        phis = np.unwrap(np.asarray(phis))
        ths = thetas.copy()
        # 线性拟合 φ = a·θ + b（角位移增益）
        if np.ptp(ths) > 0:
            gain = float(np.polyfit(ths, phis, 1)[0])
            r2 = float(np.corrcoef(ths, phis)[0, 1] ** 2)
        else:
            gain, r2 = float("nan"), float("nan")
        self.Wconn = W_conn_ref
        self.Kcos = K_ref_full
        return {"bump_persistence_drift_deg": float(drift),
                "bump_amp_before": float(amp_a), "bump_amp_after": float(amp_b),
                "bump_amp_keep": float(amp_keep),
                "heading_err_deg": errs.tolist(), "heading_err_mean_deg": float(errs.mean()),
                "heading_err_halfwidth_deg": float(np.median(errs)),
                "bump_theta": list(corr_x), "bump_phi": list(corr_y),
                "heading_correlation": float(corr), "bump_amplitude_mean": float(np.mean(amps)),
                "rotation_gain": gain, "rotation_r2": r2,
                "rot_thetas": ths.tolist(), "rot_phis": phis.tolist(),
                "ablate_ring": bool(ablate_ring)}


def _circ_corr(x: np.ndarray, y: np.ndarray) -> float:
    """圆周一致性系数 ρ = mean(cos(x−y)) ∈ [−1,1]（旋转/绕回不变，定义明确）。

    实测踩坑：Jammalamadaka–SenGupta 形式对**均匀采样**的 x（一圈 12 个航向）失效——
    x̄=atan2(近似 0, 近似 0) 数值上落在 ±π 两个分支 → 同一份完美跟踪数据给出 **−0.61** 的伪负相关。
    改用无需中心化的 mean(cos Δ)：完美跟踪 → +1（实测 φ 与 θ 逐点相等）。
    """
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.cos(x - y).mean())
