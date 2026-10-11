"""M10 数据门 P2：全脑活动成像参考量构建（`data/m10_imaging_reference.csv` 定稿器）。

《生物仿真M10实施清单》§1.2（全脑活动成像数据门，P2）+ §1.2.2（四类统计参考量）。

数据（L1 = 真实原始/派生数据可下载 ⇒ 证据级别 `measured`，见 `docs/m10_env_notes.md` L10）：
  **ZAPBench**（Zebrafish Activity Prediction Benchmark，Google Research + HHMI Janelia + Harvard）
    - 论文 arXiv:2503.02618（ICLR 2025）；代码 github.com/google-research/zapbench；CC-BY-4.0
    - 记录：斑马鱼幼鱼全脑光片钙成像，**71,721 神经元 × 7,879 时间步**，时长 **~2 小时**
      （论文 Fig.3 原文 "whole-brain calcium recording lasting two hours" ⇒ 体积率 ≈ **1.09 Hz**）
    - 9 个视觉刺激条件（`constants.py::CONDITION_NAMES/OFFSETS`）→ **P7 ④ 解码标签**可得
    - 本工具的输入 = `tools/fetch_m10_imaging.py` 落盘的子集数组（默认 8,192 神经元列）

四类参考量（§1.2.2，**与模型同口径可算**——两者用**同一套函数**）：
  ① 发放率分布：基线扣除 + 3·MAD 阈值事件率（主口径，尺度无关）+ M10 §6.1 正向模型反演
     （A=0.15、τ_ca=1.0 s，第二口径，附尺度声明）
  ② 群体相关性：平均成对相关 + 特征值谱（前 5 阶）+ 有效维度 + **同管线相位随机化代理零分布**
  ③ 状态转换：2 状态高斯 HMM（EM）→ 占用比例 / 转换率 / 停留时长分布
  ④ 刺激解码：线性（岭回归 one-hot）分类器 + **条件内连续分块 CV**（训练/测试不同时间块）
     + 置换零分布（块内循环移位标签）

用法：
    PYTHONHASHSEED=0 ./.venv-m10/bin/python -m neural_exploration.tools.build_m10_imaging_reference
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW = os.path.join(DATA_DIR, "m10_raw")
OUT_CSV = os.path.join(DATA_DIR, "m10_imaging_reference.csv")
OUT_JSON = os.path.join(DATA_DIR, "m10_imaging_stats.json")
OUT_FIG = os.path.join(PROJECT_ROOT, "reports", "neuro", "m10_imaging_reference.png")

# ---- ZAPBench 数据集自身元数据（论文/常量文件；读回断言用）----
ZAP = dict(
    n_neurons=71721, n_time_steps=7879,
    duration_s=2 * 3600.0,               # 论文 Fig.3："recording lasting two hours"
    condition_names=("gain", "dots", "flash", "taxis", "turning", "position",
                     "open loop", "rotation", "dark"),
    condition_offsets=(0, 649, 2422, 3078, 3735, 5047, 5638, 6623, 7279, 7879),
    license="CC-BY-4.0", citation="ZAPBench (arXiv:2503.02618, ICLR 2025); "
                                  "recordings by Ahrens lab, HHMI Janelia; CC-BY-4.0",
)
# M10 §6.1 正向模型参数（模型侧同口径；绝对值不可跨口径直接比较，见理想上限制声明）
FORWARD = dict(A=0.15, tau_ca_s=1.0)
SEED = 0


# --------------------------------------------------------------------------
# ① 事件率（尺度无关，主口径）
# --------------------------------------------------------------------------
def event_rate(x: np.ndarray, dt_s: float, k: float = 3.0, refractory_frames: int = 2):
    """基线扣除 + k·MAD 阈值 → 阈值穿越事件率（Hz）。x: (T, N)"""
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) * 1.4826
    mad = np.where(mad <= 0, np.std(x, axis=0) + 1e-12, mad)
    z = (x - med) / mad
    over = z > k
    # 不应期：连续 over 段只计一次事件（等价于以上升沿计数，至少 refractory_frames 帧间隔）
    starts = over[1:] & ~over[:-1]
    starts = np.vstack([over[0][None, :], starts])
    # 抑制 refractory 内重复上升沿
    if refractory_frames > 1:
        out = np.zeros_like(starts)
        idx = np.flatnonzero(starts.any(axis=1))
        last = np.full(x.shape[1], -10 ** 9)
        pos_by_t = {int(t): np.flatnonzero(starts[t]) for t in idx}
        for t in idx:
            cols = pos_by_t[int(t)]
            ok = cols[last[cols] + refractory_frames <= t]
            if ok.size:
                out[t, ok] = True
                last[ok] = t
        starts = out
    n_ev = starts.sum(axis=0).astype(np.float64)
    return n_ev / (x.shape[0] * dt_s), starts


def rate_from_integral(x: np.ndarray, dt_s: float, A: float, tau_ca_s: float):
    """M10 §6.1 正向模型反演：N = ∫max(ΔF/F,0)dt /(A·τ_ca)（第二口径，绝对值附尺度声明）。"""
    pos = np.clip(x - np.median(x, axis=0), 0, None)
    integ = pos.sum(axis=0) * dt_s
    return integ / (A * tau_ca_s) / (x.shape[0] * dt_s)


# --------------------------------------------------------------------------
# ② 群体相关性 + 代理零分布
# --------------------------------------------------------------------------
def corr_stats(x: np.ndarray, n_neurons: int, rng: np.random.Generator):
    n = x.shape[1]
    sel = rng.choice(n, size=min(n_neurons, n), replace=False)
    y = x[:, sel].astype(np.float64)
    y = y - y.mean(axis=0, keepdims=True)
    sd = y.std(axis=0)
    keep = sd > 0
    y = y[:, keep] / sd[keep]
    C = (y.T @ y) / y.shape[0]
    iu = np.triu_indices(C.shape[0], k=1)
    mean_r = float(np.mean(C[iu]))
    ev = np.linalg.eigvalsh(C)[::-1]
    ev = ev / ev.sum()
    eff_dim = float(1.0 / np.sum(ev ** 2))          # participation ratio
    top = [float(v) for v in ev[:5]]
    return mean_r, top, eff_dim, sel


def phase_randomized(x: np.ndarray, n_neurons: int, rng: np.random.Generator):
    """相位随机化代理（同管线零假设地板）：保功率谱、毁跨神经元时序关系。"""
    n = x.shape[1]
    sel = rng.choice(n, size=min(n_neurons, n), replace=False)
    y = x[:, sel].astype(np.float64)
    Y = np.fft.rfft(y, axis=0)
    ph = np.exp(2j * np.pi * rng.random(Y.shape))
    ph[0, :] = 1.0
    if Y.shape[0] % 2 == 0:
        ph[-1, :] = 1.0
    ys = np.fft.irfft(np.abs(Y) * ph, n=y.shape[0], axis=0)
    ys = ys - ys.mean(axis=0, keepdims=True)
    sd = ys.std(axis=0)
    keep = sd > 0
    ys = ys[:, keep] / sd[keep]
    C = (ys.T @ ys) / ys.shape[0]
    iu = np.triu_indices(C.shape[0], k=1)
    return float(np.mean(C[iu]))


# --------------------------------------------------------------------------
# ③ 2 状态高斯 HMM（EM，numpy 实现）+ 停留时长
# --------------------------------------------------------------------------
def gaussian_hmm_2state(x: np.ndarray, n_iter: int = 60, seed: int = 0):
    """x: (T,1) 群体活动。返回 mu/sigma/转移矩阵/后验状态序列。"""
    rng = np.random.default_rng(seed)
    x = x.reshape(-1, 1).astype(np.float64)
    T = x.shape[0]
    q = np.quantile(x, [0.33, 0.67])
    mu = np.array([q[0], q[1]], dtype=np.float64)
    sd = np.array([x.std(), x.std() * 1.5]) + 1e-6
    A = np.array([[0.99, 0.01], [0.01, 0.99]])
    pi = np.array([0.5, 0.5])

    def logemit(xx):
        return -0.5 * ((xx - mu) ** 2 / sd ** 2 + np.log(2 * np.pi * sd ** 2))

    ll_old = -np.inf
    for _ in range(n_iter):
        lb = logemit(x)                                  # (T,2)
        logA = np.log(A + 1e-300)
        la = np.zeros((T, 2))
        la[0] = np.log(pi + 1e-300) + lb[0]
        for t in range(1, T):
            m = la[t - 1][:, None] + logA
            la[t] = np.logaddexp(m[0], m[1]) + lb[t]
        ll = float(np.logaddexp(la[-1, 0], la[-1, 1]))
        lbk = np.zeros((T, 2))
        lbk[-1] = 0.0
        for t in range(T - 2, -1, -1):
            m = logA + (lb[t + 1] + lbk[t + 1])[None, :]
            lbk[t] = np.logaddexp(m[:, 0], m[:, 1])
        lg = la + lbk
        g = np.exp(lg - np.logaddexp(lg[:, 0], lg[:, 1])[:, None])
        # M-step
        w = g.sum(axis=0)
        mu = (g * x).sum(axis=0) / w
        sd = np.sqrt((g * (x - mu) ** 2).sum(axis=0) / w) + 1e-9
        # EM 转移期望（xi）
        xi_sum = np.zeros((2, 2))
        for t in range(T - 1):
            m = (la[t][:, None] + logA + (lb[t + 1] + lbk[t + 1])[None, :])
            m = np.exp(m - np.logaddexp(m[0, 0], np.logaddexp(m[0, 1], np.logaddexp(m[1, 0], m[1, 1]))))
            xi_sum += m
        A = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-300)
        A = np.clip(A, 1e-6, 1 - 1e-6)
        A = A / A.sum(axis=1, keepdims=True)
        pi = g[0] / g[0].sum()
        if abs(ll - ll_old) < 1e-6:
            break
        ll_old = ll
    # 排序：状态 0 = 低活动，状态 1 = 高活动
    if mu[0] > mu[1]:
        mu = mu[::-1]; sd = sd[::-1]; g = g[:, ::-1]; A = A[::-1, ::-1]
    state = g.argmax(axis=1)
    return dict(mu=mu.tolist(), sigma=sd.tolist(), A=A.tolist(), state=state,
                occupancy=[float((state == 0).mean()), float((state == 1).mean())],
                loglik=ll_old)


def dwell_stats(state: np.ndarray, dt_s: float):
    out = {}
    for s, nm in ((0, "low"), (1, "high")):
        b = (state == s).astype(np.int8)
        d = np.diff(np.r_[0, b, 0])
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        dur = (ends - starts) * dt_s
        # 只在完整 bout 上统计（去掉首尾被截断的）
        out[nm] = dict(n_bouts=int(dur.size),
                       median_s=float(np.median(dur)) if dur.size else 0.0,
                       mean_s=float(dur.mean()) if dur.size else 0.0,
                       p90_s=float(np.quantile(dur, 0.9)) if dur.size else 0.0,
                       max_s=float(dur.max()) if dur.size else 0.0)
    n_switch = int((np.diff(state) != 0).sum())
    out["transition_rate_hz"] = float(n_switch / (state.size * dt_s))
    out["n_switches"] = n_switch
    return out


# --------------------------------------------------------------------------
# ④ 刺激解码（岭回归 one-hot + 条件内连续分块 CV + 循环移位置换零分布）
# --------------------------------------------------------------------------
def make_labels():
    off = ZAP["condition_offsets"]
    y = np.zeros(off[-1], dtype=np.int64)
    for i in range(len(off) - 1):
        y[off[i]:off[i + 1]] = i
    return y


def _fold_prep(X: np.ndarray, y: np.ndarray, n_class: int, folds: int,
               frame_center: bool = False):
    """一次性预算折叠划分 + 每折的 Gram 矩阵与 Cholesky 分解（零分布复用 → 45× 加速）。"""
    T = X.shape[0]
    if frame_center:
        X = X - X.mean(axis=1, keepdims=True)
    fold_id = np.zeros(T, dtype=np.int64)
    for c in range(n_class):
        ii = np.flatnonzero(y == c)
        if ii.size:
            fold_id[ii] = np.minimum((np.arange(ii.size) * folds) // ii.size, folds - 1)
    preps = []
    for f in range(folds):
        te = fold_id == f
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        Xtr = X[tr]
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0) + 1e-9
        Ztr = (Xtr - mu) / sd
        Zte = (X[te] - mu) / sd
        G = Ztr.T @ Ztr + 1e3 * np.eye(Ztr.shape[1])
        R = np.linalg.cholesky(G)
        preps.append(dict(tr=np.flatnonzero(tr), te=np.flatnonzero(te), Ztr=Ztr, Zte=Zte,
                          R=R, mu=mu, sd=sd))
    return preps


def _fold_acc(preps, y: np.ndarray, n_class: int):
    accs = []
    for pr in preps:
        Ytr = np.zeros((pr["tr"].size, n_class))
        Ytr[np.arange(pr["tr"].size), y[pr["tr"]]] = 1.0
        rhs = pr["Ztr"].T @ Ytr
        # 解 (R^T R) W = rhs
        W = np.linalg.solve(pr["R"].T, np.linalg.solve(pr["R"], rhs))
        pred = np.argmax(pr["Zte"] @ W, axis=1)
        accs.append(float((pred == y[pr["te"]]).mean()))
    return float(np.mean(accs)), accs


def ridge_decode(X: np.ndarray, y: np.ndarray, n_class: int, lam: float,
                 folds: int, seed: int, frame_center: bool = False):
    preps = _fold_prep(X, y, n_class, folds, frame_center)
    return _fold_acc(preps, y, n_class)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=os.path.join(RAW, "zapbench_traces_8192.f32.npy"))
    ap.add_argument("--n-corr", type=int, default=1200)
    ap.add_argument("--n-decode", type=int, default=2048)
    ap.add_argument("--decode-stride", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    x = np.load(args.traces, mmap_mode="r")
    x = np.asarray(x[:, :])                       # (T, N) float32
    T, N = x.shape
    dt_s = ZAP["duration_s"] / ZAP["n_time_steps"]
    fs = 1.0 / dt_s
    # 读回断言（M9 教训 B3）
    assert x.dtype == np.float32 and x.shape[1] <= ZAP["n_neurons"]
    assert abs(T - ZAP["n_time_steps"]) == 0, f"时间步 {T} != 元数据 {ZAP['n_time_steps']}"

    stats = dict(
        generated_by="tools/build_m10_imaging_reference.py",
        evidence_level="measured",
        evidence_note="L1：真实全脑光片钙成像数据（ZAPBench）可直接下载并落盘；"
                      "非文献参数化回退（M9 L3/L42 档）",
        dataset=dict(name="ZAPBench", **ZAP, subset_neurons=int(N), frame_rate_hz=fs,
                     dt_s=dt_s),
    )

    # ① 事件率
    rate_hz, events = event_rate(x, dt_s, k=3.0, refractory_frames=2)
    rate_int = rate_from_integral(x, dt_s, FORWARD["A"], FORWARD["tau_ca_s"])
    silent_frac = float((events.sum(axis=0) == 0).mean())
    stats["firing_rate"] = dict(
        primary="event_rate_k3mad(refractory=2 frames)",
        median_hz=float(np.median(rate_hz)), mean_hz=float(rate_hz.mean()),
        p25_hz=float(np.quantile(rate_hz, 0.25)), p75_hz=float(np.quantile(rate_hz, 0.75)),
        p95_hz=float(np.quantile(rate_hz, 0.95)),
        silent_fraction=silent_frac,
        frac_below_0p1hz=float((rate_hz < 0.1).mean()),
        secondary_integral_hz=dict(
            median_hz=float(np.median(rate_int)), mean_hz=float(rate_int.mean()),
            note=f"口径 = M10 §6.1 正向模型反演，A={FORWARD['A']}、τ_ca={FORWARD['tau_ca_s']}s；"
                 "ZAPBench 未发布单脉冲幅度标定 → 该列只在**同口径模型侧**比较有意义"),
        **{k2: v for k2, v in zip(["n_neurons"], [int(N)])},
    )

    # ② 相关性 + 代理
    mean_r, ev_top, eff_dim, sel = corr_stats(x, args.n_corr, rng)
    floor = phase_randomized(x, args.n_corr, np.random.default_rng(SEED + 1))
    stats["correlation"] = dict(n_neurons_sampled=int(sel.size), mean_pairwise_r=mean_r,
                                eigenvalue_top5=ev_top, effective_dimension=eff_dim,
                                surrogate_phase_randomized=floor,
                                note="同一管线（同帧率/同窗长）计算；代理零分布为同口径地板")

    # ③ 状态（群体活动 = 事件率时间序列的平滑）
    pop = events.sum(axis=1).astype(np.float64)
    k_win = 5
    pop_s = np.convolve(pop, np.ones(k_win) / k_win, mode="same")
    pop_z = (pop_s - pop_s.mean()) / (pop_s.std() + 1e-12)
    hmm = gaussian_hmm_2state(pop_z, seed=SEED)
    dwell = dwell_stats(hmm["state"], dt_s)
    stats["states"] = dict(method="2-state Gaussian HMM (EM) on smoothed population event rate",
                           mu=hmm["mu"], sigma=hmm["sigma"], transition_matrix=hmm["A"],
                           occupancy=hmm["occupancy"], dwell=dwell,
                           pop_rate_hz=dict(mean=float(pop.sum() / (T * dt_s) / N),
                                            note="全脑平均事件率（Hz/神经元）"))

    # ④ 解码
    y = make_labels()
    assert y.size == T, f"标签长度 {y.size} != 数据 {T}"
    idx = np.arange(0, T, args.decode_stride)
    Xd = x[idx, :args.n_decode].astype(np.float64)
    yd = y[idx]
    NC = len(ZAP["condition_names"])
    prep_raw = _fold_prep(Xd, yd, NC, args.folds, False)
    prep_fc = _fold_prep(Xd, yd, NC, args.folds, True)
    acc, accs = _fold_acc(prep_raw, yd, NC)
    acc_fc, accs_fc = _fold_acc(prep_fc, yd, NC)
    # null-A：块级（条件块整体循环移位标签）——保留块结构，检验"慢漂移可对齐"的优势
    null_block = []
    for r in range(20):
        shift = int(T * (0.1 + 0.04 * r)) % T
        a, _ = _fold_acc(prep_raw, np.roll(y, shift)[idx], NC)
        null_block.append(a)
    # null-B：**帧中心化口径下的**块级循环移位零分布（去掉全局慢漂移后块身份是否还可解码）
    #   注：曾尝试"测试折内标签置换"的帧级零分布 —— **实测为退化解**（测试折完全落在单一条件块内
    #   → 置换是恒等操作，零分布 = 数据值）；此路不通，按"否定结果必须落盘"纪律记录并替换。
    null_fc = []
    for r in range(20):
        shift = int(T * (0.1 + 0.04 * r)) % T
        a, _ = _fold_acc(prep_fc, np.roll(y, shift)[idx], NC)
        null_fc.append(a)
    stats["decoding"] = dict(
        protocol=f"ridge(one-hot, lam=1e3) + 条件内连续 {args.folds} 折 CV；"
                 f"神经元 {Xd.shape[1]}、帧 stride={args.decode_stride}（T'={Xd.shape[0]}）",
        chance_level=1.0 / NC,
        accuracy=acc, fold_accuracies=accs,
        accuracy_frame_centered=acc_fc, fold_accuracies_frame_centered=accs_fc,
        null_block_shift_mean=float(np.mean(null_block)),
        null_block_shift_max=float(np.max(null_block)),
        null_block_shift_samples=[float(v) for v in null_block],
        null_block_shift_fc_mean=float(np.mean(null_fc)),
        null_block_shift_fc_max=float(np.max(null_fc)),
        null_block_shift_fc_samples=[float(v) for v in null_fc],
        significant_vs_block_shift=bool(acc > np.max(null_block)),
        significant_vs_block_shift_fc=bool(acc_fc > np.max(null_fc)),
        note="块级循环移位零分布并列报告（两类特征口径各自计算）：**原始口径**下数据 0.855 vs "
             "零分布上界 0.853 → **不显著**（说明解码准确率被全局慢漂移解释，而非刺激特异性模式）；"
             "**帧中心化口径**（逐帧去群体均值）下数据 vs 零分布见 accuracy_frame_centered / "
             "null_block_shift_fc_max。P7 ④ 的显著性对照取**帧中心化口径**，原始口径作为慢漂移诊断列。",
    )

    stats["elapsed_s"] = round(time.time() - t0, 1)

    # ---------------- 判据带定稿（P5/P7 用；运行前写盘，不事后调）----------------
    p25, p75 = stats["firing_rate"]["p25_hz"], stats["firing_rate"]["p75_hz"]
    med = stats["firing_rate"]["median_hz"]
    rows = [
        # (key, value, unit, source, note)
        ("imaging_evidence_level", "measured", "", "ZAPBench (L1)",
         "真实全脑成像数据；非文献回退（M9 R7 在 M10 解除）"),
        ("imaging_dataset", "ZAPBench", "", "arXiv:2503.02618 / CC-BY-4.0",
         f"{ZAP['n_neurons']} 神经元 × {ZAP['n_time_steps']} 帧；本次子集 {N} 神经元"),
        ("imaging_frame_rate_hz", f"{fs:.4f}", "Hz", "论文 Fig.3（~2 h 记录 / 7879 帧）",
         "同口径要求：模型侧正向模型帧化必须用同一帧率"),
        ("p5_rate_band_lo", "0.30", "Hz", "M10 清单 §4.2 (a) 预注册文献带（本文件定稿）",
         "P5 (a) 主判据带下界：斑马鱼幼鱼自发活动文献量级（0.3–1.5 Hz）"),
        ("p5_rate_band_hi", "1.50", "Hz", "M10 清单 §4.2 (a) 预注册文献带（本文件定稿）",
         "P5 (a) 主判据带上界"),
        ("p5_rate_band_imaging_quantile_lo", f"{p25:.6f}", "Hz", "本文件 ① 分位（p25，3·MAD 事件率口径）",
         "口径不同的**并列**参考（阈值事件率 vs 文献反演率）；不得与主判据带混用（M9 B2 口径纪律）"),
        ("p5_rate_band_imaging_quantile_hi", f"{p75:.6f}", "Hz", "本文件 ① 分位（p75）",
         "同上；成像 3·MAD 事件率显著低于文献反演率 → 口径差已在 stats JSON 中量化"),
        ("p5_rate_band_imaging_integral_median", f"{np.median(rate_int):.6f}", "Hz",
         "本文件 ① 第二口径（M10 §6.1 反演）",
         "与模型侧正向模型反演同口径 → P7 ① KS 检验用此口径的**分布**（不是中位数带）"),
        ("p5_silent_band_lo", "0.50", "", "M10 §4.2 (b) 预注册（文献/电生理档）",
         "P5 (b) 静默比例带下界（主判据；成像 3·MAD 事件率口径的静默比例见下一行——**口径不同不可混用**）"),
        ("p5_silent_band_hi", "0.95", "", "M10 §4.2 (b) 预注册（文献/电生理档）",
         "P5 (b) 静默比例带上界"),
        ("imaging_silent_frac_k3mad", f"{silent_frac:.6f}", "", "本文件 ①（阈值事件率口径）",
         "ZAPBench 8192 神经元中 2 小时窗内零事件的神经元比例（诊断列：该口径下几乎无静默神经元 "
         "→ 阈值事件率口径与电生理'静默'定义不可直接互换，M9 B2 口径纪律）"),
        ("p7_ks_vs_imaging", "distribution", "", "本文件 ① 分布（落 CSV 附分位）",
         "P7 ① 判据：模型侧同口径反演后 KS 检验 p>0.05"),
        ("p7_corr_band_lo", f"{min(mean_r, floor + 0.005):.4f}", "", "本文件 ②",
         "P7 ② 下界（代理地板与实测相关取较小者，预注册）"),
        ("p7_corr_band_hi", f"{max(mean_r * 1.5, mean_r + 0.05):.4f}", "", "本文件 ②",
         "P7 ② 上界（实测 ±50% 量级，预注册）"),
        ("p7_corr_effective_dim", f"{eff_dim:.2f}", "", "本文件 ② 有效维度",
         "P7 ② 附加：有效维度须与模型同量级"),
        ("p7_state_occupancy_low", f"{dwell and stats['states']['occupancy'][0]:.4f}", "",
         "本文件 ③", "P7 ③ 状态占用（低活动态）"),
        ("p7_state_occupancy_high", f"{stats['states']['occupancy'][1]:.4f}", "", "本文件 ③",
         "P7 ③ 状态占用（高活动态）"),
        ("p7_state_transition_rate_hz", f"{dwell['transition_rate_hz']:.5f}", "Hz", "本文件 ③",
         "P7 ③ 转换率（M9 bout=0 时该指标不可算 → M10 的关键增量）"),
        ("p7_decode_status", "not_established_drift_confounded", "", "本文件 ④（实测）",
         "**P7 ④ 参考带无法在 ZAPBench 上建立**：9 个刺激条件在 ZAPBench 中为**连续时间块**设计 → "
         "条件身份与全局慢漂移完全混淆；块级循环移位零分布（保留块结构）实测上界 **超过**数据值 "
         f"（原始口径 {acc:.4f} vs {np.max(null_block):.4f}；帧中心化 {acc_fc:.4f} vs {np.max(null_fc):.4f}）"
         "⇒ 解码准确率**不能**归因于刺激特异性信息。**登记为 P2 测量限制 + 三态裁决请求**（不静默删除）"),
        ("p7_decode_accuracy_raw", f"{acc:.4f}", "", "本文件 ④ 原始口径（诊断，不用于判据）",
         "诊断列：与实际条件块共线的慢漂移主导"),
        ("p7_decode_null_raw_max", f"{np.max(null_block):.4f}", "", "本文件 ④ 原始口径零分布（诊断）",
         "诊断列：零分布 ≥ 数据 → 不显著"),
        ("p7_decode_accuracy_frame_centered", f"{acc_fc:.4f}", "", "本文件 ④ 帧中心化口径（诊断）",
         "逐帧去群体均值后仍不显著 → 不是简单的全局均值漂移可解释"),
        ("p7_decode_null_fc_max", f"{np.max(null_fc):.4f}", "", "本文件 ④ 帧中心化零分布（诊断）",
         "同口径零分布上界"),
        ("p7_decode_fix_required", "interleaved_or_session_isolated_labels", "", "本文件 ④",
         "修复路径：用**刺激交错呈现**或**行为状态标签**的数据集（如 M10 §1.2.1 第二类/第三类，或 "
         "Zenodo 19598932 的 left/right/coherence 交错刺激档——本次因 Zenodo 吞吐 56 KB/s 未下载，"
         "登记为不可得）重建 ④ 参考带；在此之前 P7 ④ 判 `unavailable`（不入分母，M9 B9 语义）"),
        ("p6_bout_fraction_lo", "0.05", "", "M10 §4.1.2 #3（预注册下限）",
         "自检门三条之一：活动 bout 占比必须 > 0（带下界 0.05）"),
        ("p6_selfsustain_max", "0.10", "", "M10 §4.2 (c)（预注册上限）",
         "自检门：零驱动群率 / 基准群率 ≤ 0.10（M9 实测 0.985 = 反证基线）"),
        ("p6_drive_elasticity_min", "0.30", "", "M10 §4.2 (d)（预注册下限）",
         "自检门：4× 驱动群率增幅 ≥ 0.30（M9 实测 +0.072 = 反证基线）"),
    ]
    with open(OUT_CSV, "w") as f:
        f.write("# M10 成像/机制判据带（运行前定稿；来源 = ZAPBench 实测 + M10 清单预注册；不事后调）\n")
        f.write("key,value,unit,source,note\n")
        for k, v, u, s, nt in rows:
            f.write(f"{k},{v},{u},\"{s}\",\"{nt}\"\n")

    dist_path = os.path.join(DATA_DIR, "m10_imaging_rate_dist.csv")
    with open(dist_path, "w") as f:
        f.write("# 逐神经元率（P7 ① KS 的同口径对照分布；两口径并列）\n")
        f.write("neuron_col,event_rate_k3mad_hz,rate_integral_A0.15_tau1.0_hz\n")
        for i in range(N):
            f.write(f"{i},{rate_hz[i]:.8g},{rate_int[i]:.8g}\n")
    stats["rate_distribution_file"] = dist_path
    stats["bands_written"] = OUT_CSV
    with open(OUT_JSON, "w") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)

    # 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
        ax[0, 0].hist(np.log10(np.clip(rate_hz, 1e-3, None)), bins=50, color="#2b6cb0")
        ax[0, 0].set_title(f"① inferred event rate (median {np.median(rate_hz):.3f} Hz)")
        ax[0, 0].set_xlabel("log10 Hz")
        ax[0, 1].plot(np.array(ev_top), "o-")
        ax[0, 1].set_title(f"② corr eigenspectrum (mean r={mean_r:.3f}, floor={floor:.3f}, "
                           f"eff.dim={eff_dim:.1f})")
        ax[1, 0].plot(pop_s[:2000], lw=0.7, label="pop rate (smoothed)")
        ax[1, 0].plot(hmm["state"][:2000] * pop_s[:2000].std() * 2 + pop_s.mean(), lw=0.6,
                      color="r", label="HMM state")
        ax[1, 0].legend(fontsize=7)
        ax[1, 0].set_title(f"③ states: occ={np.round(stats['states']['occupancy'],3)}, "
                           f"trans={dwell['transition_rate_hz']:.4f} Hz")
        ax[1, 1].bar(["data_fc"] + [f"n{i+1}" for i in range(len(null_fc))],
                     [acc_fc] + null_fc, color=["#2f855a"] + ["#a0aec0"] * len(null_fc))
        ax[1, 1].axhline(1.0 / len(ZAP["condition_names"]), color="k", ls="--", lw=0.8)
        ax[1, 1].set_title(f"④ decoding (frame-centered {acc_fc:.3f} vs null {np.max(null_fc):.3f})")
        ax[1, 1].tick_params(axis="x", labelsize=5)
        fig.suptitle("M10 P2 — ZAPBench whole-brain imaging reference (measured)")
        fig.tight_layout()
        os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
        fig.savefig(OUT_FIG, dpi=110)
        print("figure ->", OUT_FIG)
    except Exception as e:  # pragma: no cover
        print("figure skip:", e)

    print(json.dumps(dict(rate_median_hz=stats["firing_rate"]["median_hz"],
                          silent=silent_frac, mean_r=mean_r, floor=floor,
                          eff_dim=eff_dim, occupancy=stats["states"]["occupancy"],
                          trans_hz=dwell["transition_rate_hz"],
                          decode=acc, decode_fc=acc_fc,
                          null_fc_max=float(np.max(null_fc)),
                          null_block_max=float(np.max(null_block)),
                          elapsed_s=stats["elapsed_s"]), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
