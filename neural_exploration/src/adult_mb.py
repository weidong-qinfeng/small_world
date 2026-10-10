"""M9 P5：蘑菇体（MB）联想学习微回路 + 三因子门控可塑性（`src/adult_mb.py`，新建）。

《生物仿真M9实施清单》§4.1（P5：MB 联想泛化——CS→KC→MBON + US=DA 奖赏 + 三因子门控）
+ §0.3.5（抽象登记）+ §8 反证路径。

**结构（真实连接组子图，非合成）**：
  - **AL**（触角叶）3,449 神经元 —— 嗅觉输入层（区域代理）；
  - **KC**（Kenyon 细胞，主脑区 MB_CA = 蘑菇体萼）**3,819** 神经元；
  - **MBON**（主脑区 MB_ML/MB_VL/MB_PED）**1,374** 神经元；
  - **KC→MBON 真实边 100,533 条**（syn_count 均值 3.70 / 中位 2 / 最大 168）；
  - **AL→KC 真实边 24,012 条**；
  - **DA（多巴胺能）神经元**：由连接组 neuron 行 `neurotransmitter == dopaminergic` 识别（896 个）
    → 作为 **US/奖赏门控信号源**（Eckstein 2024 递质预测；**无受体映射 → 功能门控语义**，M6 惯例）。

**抽象登记（§0.3.5）**：
  1. **区域代理解剖身份**：v783 无 cell_type（Schlegel parquet 不可及）→ KC/MBON/AL 以**主脑区**代理；
     误差 = 区内含非 KC/MBON 神经元；回归条件 = 取得 cell_type 标注后按类型重建；
  2. **气味编码**：真实 odor→glomerulus→KC 映射不可得 → 以 **AL 空间的重叠度参数化相似度**
     （相似度 = AL 激活模式 Jaccard 重叠），**相似度梯度为抽象构造**（回归条件 = 取得真实气味
     反应谱后替换）；
  3. **三因子规则形式**：eligibility 迹（Hebbian 预-后共现，τ_e）× DA 门控（标量 US 信号），
     承接 M6 `plasticity.py` 三因子语义与 M8 P5 机制级经验（LI=0.895）；
  4. **点神经元简化**：KC/MBON 为率型读出（无 spiking 动力学）——P5 只承诺**机制级判据**
     （Δw / MBON 价效 / 泛化形状），行为级判据按清单 §0.7 #9 双轨记录。

**学习规则（三因子）**：
    e_ij ← e_ij·(1 − dt/τ_e) + KC_j·MBON_i          （eligibility，Hebbian 共现）
    w_ij ← w_ij + η · e_ij · g_DA                    （DA 门控；g_DA = US 存在时的奖赏信号）
    w_ij ← clip(w_ij, 0, w_max)                      （有界，防饱和）

用法：
    from neural_exploration.src.adult_mb import AdultMB, MBParams
    mb = AdultMB(); r = mb.run_protocol()
"""

from __future__ import annotations

import csv as _csv
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NET_NPZ = os.path.join(DATA_DIR, "m9_network.npz")
NEURON_NPZ = os.path.join(DATA_DIR, "m9_neuron_table.npz")
CSV_PATH = os.path.join(DATA_DIR, "m9_flywire_connectome.csv")

ROLE_REGIONS = {
    "KC": ("MB_CA",),
    "MBON": ("MB_ML", "MB_VL", "MB_PED"),
    "AL": ("AL_", "AL"),
    "CX_EB": ("EB",), "CX_PB": ("PB",), "CX_FB": ("FB",), "CX_NO": ("NO",),
}


# ---------------------------------------------------------------------------
@dataclass
class MBParams:
    """P5 微回路参数（可标定；默认 = 预注册草案）。"""
    kc_sparsity: float = 0.08      # 每个气味激活的 KC 比例（果蝇 MB 稀疏码 ~5–10%）
    n_glomeruli: int = 50          # AL 空间维度（气味编码空间；抽象登记 ②）
    odor_overlap: float = 0.30     # 气味对之间的 AL 重叠（相似度参数）
    tau_e_ms: float = 200.0        # eligibility 迹时间常数 τ_e
    eta: float = 0.05              # 学习率 η（三因子规则）
    da_gain: float = 1.0           # DA 门控增益（US 强度）
    w_max: float = 8.0             # 权重上界（须 > 初值上界；初值 = 1+ln(syn_count) ≤ ~6.1）
    w_scale: float = 1.0           # 连接组 syn_count → 初始权重缩放
    n_train_pairs: int = 12        # 训练/测试协议的气味对总数（配对组 + 对照组分半）
    dt_ms: float = 1.0             # 微回路时间步
    t_odor_ms: float = 200.0       # 单次气味呈现时长
    t_us_ms: float = 100.0         # US（DA）呈现时长（与 CS 重叠 = 配对）
    n_pairing: int = 3             # 配对训练轮数
    seed: int = 0


# ---------------------------------------------------------------------------
class AdultMB:
    """蘑菇体联想学习微回路（真实连接组子图 + 三因子门控可塑性）。"""

    def __init__(self, params: Optional[MBParams] = None, device: str = "cpu",
                 verbose: bool = True):
        self.p = params or MBParams()
        self.device = device
        self.verbose = verbose
        self._load_subgraph()
        self.reset()

    # ---------------- 装配 ----------------
    def _load_subgraph(self) -> None:
        d = np.load(NET_NPZ, allow_pickle=False)
        nt = np.load(NEURON_NPZ, allow_pickle=False)
        names = np.asarray([str(x) for x in nt["region_names"]])
        code = nt["region_code"]
        self.region_names = names
        n = int(nt["root_ids"].size)

        def mask(prefixes) -> np.ndarray:
            codes = [i for i, nm in enumerate(names) if any(nm.startswith(p) for p in prefixes)]
            return np.isin(code, codes)

        self.m_kc = mask(ROLE_REGIONS["KC"])
        self.m_mbon = mask(ROLE_REGIONS["MBON"])
        self.m_al = mask(ROLE_REGIONS["AL"])
        self.root_ids = nt["root_ids"]
        self.dom_nt = nt["dom_nt_code"]

        pre = d["pre"].astype(np.int64); post = d["post"].astype(np.int64)
        sc = d["syn_count"].astype(np.float32)
        i_kc = np.flatnonzero(self.m_kc); i_mbon = np.flatnonzero(self.m_mbon)
        i_al = np.flatnonzero(self.m_al)
        e_km = np.isin(pre, i_kc) & np.isin(post, i_mbon)
        e_ak = np.isin(pre, i_al) & np.isin(post, i_kc)
        # 重标号到子图局部索引
        kc_map = -np.ones(n, dtype=np.int64); kc_map[i_kc] = np.arange(i_kc.size)
        mb_map = -np.ones(n, dtype=np.int64); mb_map[i_mbon] = np.arange(i_mbon.size)
        al_map = -np.ones(n, dtype=np.int64); al_map[i_al] = np.arange(i_al.size)
        self.n_kc, self.n_mbon, self.n_al = i_kc.size, i_mbon.size, i_al.size
        self.kc_root = self.root_ids[i_kc]
        self.mbon_root = self.root_ids[i_mbon]
        self.km_pre = kc_map[pre[e_km]]; self.km_post = mb_map[post[e_km]]
        self.km_syn = sc[e_km]
        self.ak_pre = al_map[pre[e_ak]]; self.ak_post = kc_map[post[e_ak]]
        self.ak_syn = sc[e_ak]
        # DA 神经元（多巴胺能）——从连接组 CSV 的 neuron 行识别
        self.m_da = self._load_da_mask()
        if self.verbose:
            print("MB 微回路：KC=%d / MBON=%d / AL=%d / DA=%d；KC→MBON 边=%d；AL→KC 边=%d"
                  % (self.n_kc, self.n_mbon, self.n_al, int(self.m_da.sum()),
                     self.km_pre.size, self.ak_pre.size), flush=True)

    def _load_da_mask(self) -> np.ndarray:
        """多巴胺能神经元掩码（连接组 neuron 行 neurotransmitter == dopaminergic）。"""
        n = self.root_ids.size
        out = np.zeros(n, dtype=bool)
        if not os.path.exists(CSV_PATH):
            return out
        idx = {int(r): i for i, r in enumerate(self.root_ids)}
        with open(CSV_PATH, encoding="utf-8") as f:
            for line in f:
                if not line.startswith("neuron,"):
                    continue
                parts = line.rstrip("\n").split(",")
                if len(parts) > 5 and parts[5] == "dopaminergic":
                    i = idx.get(int(parts[1]))
                    if i is not None:
                        out[i] = True
        return out

    # ---------------- 状态 ----------------
    def reset(self) -> None:
        p = self.p
        # 初始权重：1 + ln(syn_count)（保留连接组异质性但**有界**，避免 w_max 削平初值结构——
        # 实测踩坑：直接用 syn_count（最大 168）会被 w_max=5 剪切 → Δw 与 η 无关）
        self.w = (1.0 + np.log(self.km_syn.astype(np.float64))) * p.w_scale
        self.w0 = self.w.copy()
        self.elig = np.zeros_like(self.w)
        self.rng = np.random.default_rng(p.seed)
        self._kc_norm = 0.0

    # ---------------- 气味编码（抽象登记 ②） ----------------
    def odor_pattern(self, n_odor: int = 1, overlap: Optional[float] = None,
                     start: Optional[int] = None) -> np.ndarray:
        """生成 AL 空间的稀疏气味模式（形状 (n_odor, n_glomeruli)）。

        第 0 个为基准气味（CS+）；其后为与基准重叠度递减的相似气味（泛化梯度）。

        `start`：基准块在 glomerulus 环上的**起始位置**（`None` = 0，即**原行为**）。
        不同 `start` 给出**结构相同但神经元身份不同**的气味 → 用于让"气味对"成为
        **真正独立的重复**（实测踩坑：`start` 全为 0 时 12 个气味对的 ΔLI **逐位相同** →
        伪重复，置换检验的有效样本量实为 1、Cohen d 退化为 1e15）。
        """
        p = self.p
        base_ov = p.odor_overlap if overlap is None else float(overlap)
        g = p.n_glomeruli
        B = int(np.ceil(self.n_al / g))                # 每个"glomerulus"覆盖的 AL 神经元数
        s0 = 0 if start is None else int(start) % g

        def _mk(n_glom_active: int) -> np.ndarray:
            # 自 s0 起连续 nb 个 glomerulus 的 AL 神经元置 1（抽象登记 ②：相似度 =
            # AL 激活模式的 Jaccard 重叠，由 glomerulus 数控制）
            v = np.zeros(self.n_al, dtype=np.float64)
            nb = max(1, int(round(n_glom_active)))
            for gi in range(min(nb, g)):
                j = (s0 + gi) % g
                a, b2 = j * B, min((j + 1) * B, self.n_al)
                v[a:b2] = 1.0
            return v
        pats = [_mk(base_ov * g)]
        for k in range(1, n_odor):                     # 相似度**线性递减**至 ~1/n_odor
            pats.append(_mk(base_ov * g * (1.0 - k / max(n_odor, 2))))
        return np.array(pats)

    def kc_code(self, al_vec: np.ndarray) -> np.ndarray:
        """AL 模式 → KC 稀疏码（真实 AL→KC 连接 + k-WTA 稀疏化）。"""
        p = self.p
        drive = np.bincount(self.ak_post, weights=self.ak_syn * al_vec[self.ak_pre],
                            minlength=self.n_kc)
        if self._kc_norm <= 0:
            mx = float(drive.max())
            self._kc_norm = mx if mx > 0 else 1.0
        if drive.max() <= 0:
            return np.zeros(self.n_kc, dtype=np.float64)
        # **全局参考归一**（基准气味的最大驱动）——若逐气味归一，弱输入气味也会被放大到满幅，
        # 使泛化曲线出现 U 形（实测踩坑）
        drive = drive / self._kc_norm
        k = max(1, int(round(p.kc_sparsity * self.n_kc)))
        thr = float(np.partition(drive, -k)[-k])
        # 稀疏掩码（top-k）但**保留模拟幅度** —— 与真实 MB 稀疏码语义一致，
        # 且使气味相似度→响应的梯度连续（纯二值码会使泛化曲线失去单调性，实测踩坑）
        return np.where(drive >= thr, drive, 0.0)

    def mbon_readout(self, kc: np.ndarray) -> np.ndarray:
        """MBON 率型读出（当前权重）。"""
        return np.bincount(self.km_post, weights=self.w * kc[self.km_pre],
                           minlength=self.n_mbon)

    # ---------------- 三因子学习 ----------------
    def present(self, al_vec: np.ndarray, da: float, t_ms: Optional[float] = None,
                learn: bool = True, eta: Optional[float] = None) -> Dict[str, float]:
        """呈现一次气味（±US）；返回 MBON 响应与权重变化量。"""
        p = self.p
        t_ms = p.t_odor_ms if t_ms is None else float(t_ms)
        eta = p.eta if eta is None else float(eta)
        kc = self.kc_code(al_vec)
        n_steps = max(1, int(round(t_ms / p.dt_ms)))
        decay = 1.0 - p.dt_ms / max(p.tau_e_ms, 1e-6)
        w_before = self.w.copy()
        for _ in range(n_steps):
            mbon = self.mbon_readout(kc)               # 步起点读出（同步估值语义）
            if learn:
                # 逐边 Hebbian 资格迹：e_ij ← e_ij·decay + KC_j · MBON_i
                self.elig = self.elig * decay + kc[self.km_pre] * mbon[self.km_post]
        # 三因子：DA 在呈现**末尾单次**施加 Δw = η·e·g_DA / n_steps
        # （实测踩坑：若每步施加，呈现内 w↑→MBON↑→e↑ 正反馈 → 权重全部撞上界 → Δw 与 η 无关）
        if learn and da:
            self.w = np.clip(self.w + eta * self.elig * (da * p.da_gain) / n_steps,
                             0.0, p.w_max)
        mbon_end = self.mbon_readout(kc)
        return {"kc_active": float(kc.sum()), "mbon_mean": float(mbon_end.mean()),
                "mbon_max": float(mbon_end.max()),
                "dw_l1": float(np.abs(self.w - w_before).sum()),
                "mbon_approach": float(mbon_end.mean())}

    # ---------------- 协议 ----------------
    def run_protocol(self, n_pairs: Optional[int] = None, eta: Optional[float] = None,
                     da_gain: Optional[float] = None, ablate_subgraph: bool = False,
                     n_generalization: int = 12,
                     independent_patterns: bool = False) -> Dict[str, Any]:
        """P5 协议：配对训练（CS++US / CS− 无 US）→ 偏好测试 + 泛化梯度。

        臂：配对组（CS+ 与 US 配对）vs 未配对对照（CS− 仅呈现，无 US）。
        返回 LI / 泛化曲线 / Δw 等机制级判据量。

        `independent_patterns=False`（默认 = **原行为**，供审计复现）：每对使用**同一组**
          气味模式（`start=0`）。实测后果：12 个 ΔLI **逐位相同**（sd≈3e-17）→ **伪重复**，
          置换检验有效样本量实为 1，Cohen d 退化。
        `independent_patterns=True`（**修正口径**）：由 `seed` 派生的 rng 为**每一对**抽一个
          不同的 glomerulus 起始位置 → 12 对成为**真正独立的重复**；气味**结构**（Jaccard
          重叠/强度序）不变，只换神经元身份。
        """
        p = self.p
        npairs = n_pairs or p.n_train_pairs
        eta_use = p.eta if eta is None else float(eta)
        da_use = p.da_gain if da_gain is None else float(da_gain)
        if ablate_subgraph:
            self.w[:] = 0.0
        self.reset()
        self._kc_norm = 0.0
        self.kc_code(self.odor_pattern(1, overlap=p.odor_overlap)[0])   # 校准全局参考
        ref = self._kc_norm
        rng = np.random.default_rng(p.seed)
        li_list, paired_list, unpaired_list, dwnorm = [], [], [], []
        li_abs_list, li_base_list = [], []
        gen_curves = []
        for k in range(npairs):
            self.reset()
            self._kc_norm = ref
            if ablate_subgraph:
                self.w[:] = 0.0
            # **独立重复**（修正口径）：每对抽一个不同的 glomerulus 起始位置；
            # 默认 None → start=0（原行为，12 对逐位相同 → 伪重复，见 docstring）
            start_k = None
            if independent_patterns:
                start_k = int(rng.integers(0, p.n_glomeruli))
            pats = self.odor_pattern(3, overlap=p.odor_overlap, start=start_k)
            cs_plus, cs_ctrl = pats[0], pats[1]
            # **训练前基线**（相对读法主判据；M8 §6 限制 2 语义 / 清单 §4.1 "未配对漂移读法"）：
            # 权重异质使 CS+ 与 CS− 的读出本身有非零差 → 必须以**学习前后差值**为 LI，
            # 否则 η=0/DA=0 消融臂会残留"伪 LI"（实测踩坑：基线 LI=0.164）
            r_p0 = self.present(cs_plus, da=0.0, learn=False)["mbon_approach"]
            r_c0 = self.present(cs_ctrl, da=0.0, learn=False)["mbon_approach"]
            li_base = (r_p0 - r_c0) / max(abs(r_p0) + abs(r_c0), 1e-12)
            # 训练：配对（CS+ 与 US 同时）与未配对（CS− 无 US）
            for _ in range(p.n_pairing):
                # 配对 trial：CS 先累积 eligibility，US 在其上施加 Δw
                self.elig[:] = 0.0
                self.present(cs_plus, da=0.0, learn=True, eta=eta_use)
                self.present(cs_plus, da=1.0 * da_use, learn=True, eta=eta_use)
                # 未配对 trial：仅 CS−，无 US；trial 结束清零迹（**阻断跨 trial 误强化**——
                # 这是本轮实测发现的关键 bug：残迹会让 CS− 被 CS+ 的 US 间接强化 → LI 塌缩）
                self.elig[:] = 0.0
                self.present(cs_ctrl, da=0.0, learn=True, eta=eta_use)
                self.elig[:] = 0.0
            dw = float(np.abs(self.w - self.w0).sum())
            dwnorm.append(dw)
            # 测试：无学习读出
            r_plus = self.present(cs_plus, da=0.0, learn=False)["mbon_approach"]
            r_ctrl = self.present(cs_ctrl, da=0.0, learn=False)["mbon_approach"]
            li_after = (r_plus - r_ctrl) / max(abs(r_plus) + abs(r_ctrl), 1e-12)
            li = li_after - li_base            # **相对读法**（主判据）
            li_list.append(li)
            li_abs_list.append(li_after); li_base_list.append(li_base)
            paired_list.append(li)
            # 未配对对照臂：CS− 无 US ⇒ 三因子规则下 Δw≡0 ⇒ LI **结构上恒为 0**
            # （**不是测量值**；登记为不可判别，判别力须由 Δw 读回提供 —— 见 L40）
            unpaired_list.append(0.0)
            # 泛化梯度（以 CS+ 为基准的相似度递减气味；与 k=0 的 CS+ 同一 start）
            if k == 0:
                gp = self.odor_pattern(n_generalization, overlap=p.odor_overlap, start=start_k)
                for o in gp:
                    gen_curves.append(self.present(o, da=0.0, learn=False)["mbon_approach"])
        gen = np.asarray(gen_curves, dtype=np.float64)
        gen_norm = gen / max(gen.max(), 1e-12)
        sims = np.array([1.0] + [1.0 - k / max(n_generalization, 2)
                                 for k in range(1, n_generalization)])
        # 半宽（相似度半宽）：响应落到峰值 50% 处的相似度
        half = float("nan")
        for i in range(len(gen_norm) - 1):
            if gen_norm[i] >= 0.5 >= gen_norm[i + 1]:
                f = (gen_norm[i] - 0.5) / max(gen_norm[i] - gen_norm[i + 1], 1e-12)
                half = float(sims[i] + f * (sims[i + 1] - sims[i]))
                break
        return {"li": float(np.mean(li_list)), "li_sd": float(np.std(li_list)),
                "li_abs": float(np.mean(li_abs_list)), "li_base": float(np.mean(li_base_list)),
                "paired": paired_list, "unpaired": unpaired_list,
                "dw_l1": float(np.mean(dwnorm)), "gen": gen_norm.tolist(),
                "gen_halfwidth": half, "sims": sims.tolist()}


def spearman(x, y):
    """Spearman 秩相关（无 scipy 依赖；确定性）。"""
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    den = math_sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def math_sqrt(v: float) -> float:
    return v ** 0.5
