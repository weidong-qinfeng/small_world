"""M5 参考解：NEURON 咽部子图（P3）+ 行为参考模型扩展（P5 逃避 / P6 自发）。

清单《生物仿真M5实施清单》§4（步骤 3：参考解）两级参考：

1) 神经级（P3 咽部节律，NEURON 9.0.1 cvode 高精度 + scipy solve_ivp 缝隙独立解）：
   - **Stage A（NEURON 化学子图）**：按 `data/m5_pharynx_subgraph.csv` 构建咽部 20 神经元
     （build_neuron + ExpSyn + NetCon，模式照 tools/build_reflex_ref.py / build_chemotaxis_ref.py），
     cvode atol/rtol=1e-8、celsius=6.3、v_init=V0；点过程（IClamp/ExpSyn/NetCon）列表持有防 GC
     （M2 L8）。两种协议：无食物基线（无外部输入）/ 有食物驱动（tonic IClamp 到全部 20 神经元，
     "食物 → 广义咽部兴奋"简化，NSM 血清素行 g=0 占位不动）→ 各级 node3 发放序列。
   - **Stage B（scipy solve_ivp 缝隙网络）**：20 点 HH 神经元（方程/参数与 Brian2 点神经元同源，
     hh_spec + point_neuron.py 语义）经 33 条缝隙连接（g=0.5nS，M2 值）欧姆耦合；
     泵马达池 {MCL,MCR,M4} 以**慢适应（slow-AHP）突发放电**为泵节律（Avery & Horvitz 1989：
     MC 定泵速；M4 参与节律；慢 K 适应是咽部 CPG 标准机制——功能参考，参数校准落带）；
     Stage-A 发放经 ExpSyn 卷积（ampa τ=3ms/E=0mV）作为化学输入（w_chem 校准）。
     缝隙因 NEURON gap.mod 不可用（M2 L2）→ 本阶段独立高精度解（solve_ivp LSODA rtol=1e-10,
     atol=1e-12，M2 同款）。输出泵信号（MCL/MCR/M4 发放池）功率谱/自相关 → 主频。
     输出键：pharynx_spike_times_{no_food,food}[_{NAME}]、pharynx_v_{NAME}_{proto}、
     pharynx_psd_{proto}、pharynx_peak_freq_{proto}、pharynx_peak_freq_acf_{proto}、
     pharynx_burst_rate_{proto}（簇率）、pharynx_drift_{proto}、pharynx_gap_v_{NAME}_{proto}。

2) 行为级（纯 numpy，引擎无关）：
   - **逃避参考（P5）**：触刺激 → 后退 numpy 运动学。转导延迟 τ_trans（触→感觉电流）+
     神经链潜伏期（M3 已验证窗 [5,20]ms，实测值抽样 data/m3_reflex_ref.npz）+
     肌肉收缩上升（τ_mus=20ms，M3 值）→ 行为潜伏期 ∈ Chalfie 1985 窗 [30,50]ms（容差 [25,60]）。
     方向：C_back 峰值 0.6 > C_fwd 基线 0.197，D_peak=0.403 > 0.3（M3 判据一致）。
     输出键：escape_ref_*。
   - **自发行为参考（P6）**：bout 状态马尔可夫模型（前进/后退/转弯/暂停转移矩阵 + 指数 bout
     时长），参数以 `data/m5_behavior_reference.csv` 比例带校准（前进 [60,80]/后退 [10,25]/
     转弯 [5,20]%，Srivastava 2013）；状态分类用 `classify_state`（numpy 参考版，共用约定见
     PART C 注释——B1d 的 src/virtual_body.py 须同签名同语义，P6 判定统一从单一实现调用，
     不得复制粘贴，M5 清单 §9 风险表）。输出键：spontaneous_ref_*。

落盘：`data/m5_ref.npz`（含 meta：引擎/参数/校准/带检查/实测坑 L23+ 候选）。
趋化参考（P4）直接复用 `data/m4_ref.npz`（pirouette，本文件不重复，meta 中引用）。

用法：
  .venv-neuro/bin/python -m neural_exploration.tools.build_m5_ref
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.tools.load_morphology import (  # noqa: E402
    V0, load_morphology,
)
from neural_exploration.tools.build_neuron_ref import (  # noqa: E402
    SOMA_AREA_CM2, build_neuron,
)
from neural_exploration.tools.hh_spec import (  # noqa: E402
    CM, EK, EL, ENA, GK, GL, GNA, steady_state,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REF_NPZ = os.path.join(DATA_DIR, "m5_ref.npz")
PHARYNX_CSV = os.path.join(DATA_DIR, "m5_pharynx_subgraph.csv")
BEHAVIOR_CSV = os.path.join(DATA_DIR, "m5_behavior_reference.csv")
WORM_PARAMS_CSV = os.path.join(DATA_DIR, "m5_worm_params.csv")
M3_REF_NPZ = os.path.join(DATA_DIR, "m3_reflex_ref.npz")

# --------------------------------------------------------------------- #
# 常量（协议/机制参数；行为带唯一对照源 = data/m5_behavior_reference.csv）
# --------------------------------------------------------------------- #
DT_OUT = 0.1            # ms：NEURON 重采样 / Stage-B 输出网格（峰定位精度）
SPIKE_THRESH_MV = -15.0   # node3 发放检测阈值（上冲；M3/M4 同款）
SPIKE_REF_MS = 1.5        # 发放去重窗口
NC_THRESH_MV = -20.0      # NetCon 触发阈值（M3/M4 同款）

# Stage-A（NEURON 化学子图）协议
STAGE_A_T_MS = 30000.0        # Stage-A 时长（≥P3 的 T≥10s；与 Stage-B 对齐，化学输入全程覆盖）
FOOD_DRIVE_UA_CM2 = 12.0      # 有食物：tonic 驱动密度（全部 20 神经元；探针实测 12µA/cm² 稀疏发放
                              # ~2Hz/神经元、cvode 快（5s 模拟 ~1-3s），无近阈值挂起）
FOOD_DRIVE_SET = None         # None = 全部 20 神经元（"食物 → 广义咽部兴奋"简化）

# Stage-B（scipy 缝隙网络）协议
STAGE_B_T_MS = 30000.0        # Stage-B 时长（P3 节律 T≥10s；30s 使 0.1–2Hz 带 PSD 分辨率 0.033Hz）
GAP_G_NS = 0.5                # 缝隙电导（M2 定稿 0.5nS；m2_synapse_params gap 行）
AMPA_TAU_MS = 3.0             # ExpSyn τ（m2_synapse_params ampa 行）
AMPA_E_MV = 0.0               # ExpSyn E（m2_synapse_params ampa 行）
AREA_CM2 = SOMA_AREA_CM2      # 点面积（π·d²，d=20µm；point_neuron.py 同值）
PACEMAKER = ("MCL", "MCR", "M4")   # 泵马达池（slow-AHP 起搏；Avery & Horvitz 1989）

# slow-AHP 起搏机制：I_sahp = −g_sahp·w·(V−EK) [µA/cm²]，
#   dw/dt = (w_inf(V)−w)/τ_sahp, w_inf(V) = 1/(1+exp(−(V−θ_w)/k_w))
SAHP_THETA_MV = -30.0
SAHP_K_MV = 5.0
# 校准扫描（协议参数 → 落带；以簇率 burst_rate 为校准目标，稳健）
# 无食物：起搏马达池 I(µA/cm²) × g_sahp(mS/cm²) × τ_sahp(ms)（探针实测 0.2-0.6Hz 区间）
NOFOOD_CAL = dict(I=(12.0, 15.0), g_sahp=(2.0, 4.0), tau=(800.0, 1500.0))
# 有食物：I × w_chem（化学输入权重；g_sahp=8/τ=200 起搏参数——探针实测 2.3-4.5Hz）
FOOD_CAL = dict(I=(18.0, 20.0), g_sahp=(8.0,), tau=(200.0,), w_chem=(0.0, 0.2))
NO_FOOD_TARGET_HZ = 0.5        # 无食物带 [0.1,2] 内目标簇率（带宽中心偏下，稳健）
FOOD_TARGET_HZ = 3.5           # 有食物带 [2,5] 内目标簇率
# Stage-B 求解容差（最终运行）：LSODA rtol=1e-9/atol=1e-11——20 神经元网络高精度档
# （M2 的 1e-10 为 2 神经元先例；20 神经元网络下 1e-9 与 1e-10 发放计数一致、簇率同带，
#  具体主频值对容差敏感——实测记录为测量限制，见 meta note）
STAGE_B_RTOL = 1e-9
STAGE_B_ATOL = 1e-11
CAL_NOFOOD_T_MS = 15000.0      # 校准窗（无食物，≥10s 判据）
CAL_FOOD_T_MS = 10000.0        # 校准窗（有食物）
PUMP_SMOOTH_SIGMA_MS = 100.0   # 泵信号高斯平滑 σ（包络提取）
PSD_BAND_HZ = (0.05, 8.0)      # PSD 峰值搜索带（排除尖峰内率成分）
BURST_GAP_MS = 100.0           # 簇划分 ISI 阈值（泵事件定义）

# 行为参考
RNG_SEED = 0
N_ESCAPE_TRIALS = 20           # P5 协议 N=20
N_SPONT_TRIALS = 10            # P6 协议 N=10
SPONT_T_MS = 30000.0           # P6 协议 T≥30s
DT_B_MS = 25.0                 # 行为 tick（m5_worm_params body.dt_b）
TAU_TRANS_MS = 23.0            # 转导延迟锚（行为40 − 神经9.5 − 肌肉7.1 ≈ 23；P5 协议节点建议写入 CSV）
TAU_TRANS_JIT_MS = 2.0         # 试次间转导延迟抖动（确定性 seed）
MUSCLE_TAU_MS = 20.0           # 肌肉收缩时间常数（M3 定稿 muscle_tau_ms）
MUSCLE_W_BACK = 0.60           # 后退肌肉 w（M3 定稿 muscle_w_back）
C_FWD_BASELINE = 0.197         # 前进命令基线（M3 m3_reflex_ref.npz c_fwd_baseline 实测）
C_SUPP_TAU_MS = 10.0           # 逃避时前进命令抑制时间常数（命令互斥，M3 语义）
B_OUT_THR_FRAC = 0.3           # 行为潜伏期定义：C_back ≥ 0.3·C_back_peak（M3 同款）
# 自发 bout 时长均值（Srivastava 2013 量级；informational；fwd 8s + rev/turn 交替
# 结构使 60-80% 前进时间比例带可达：每前进 bout 后 ~1 后退 + ~1 转弯）
BOUT_MEAN_MS = dict(fwd=8000.0, rev=3000.0, turn=2500.0, pause=1500.0)
# 自发转移矩阵校准网格（嵌入链；行列序 = fwd, rev, turn, pause）
# p_fr+p_ft 需足够大（P_FF 低）→ 前进 bout 不连续自锁（否则 fwd 时间比例 > 带）
SPONT_CAL = dict(p_fr=(0.20, 0.35, 0.50), p_ft=(0.15, 0.25, 0.35),
                 p_rf=(0.4, 0.6, 0.8), p_tf=(0.5, 0.7, 0.9))

# --------------------------------------------------------------------- #
# CSV 读取
# --------------------------------------------------------------------- #
def _csv_rows(path: str):
    """跳过 # 注释行（含引号包裹的注释）→ DictReader 行。"""
    with open(path, newline="", encoding="utf-8") as f:
        for row in f:
            s = row.strip()
            if not s or s.startswith("#") or s.startswith('"#'):
                continue
            yield row


_PHARYNX_COLS = ("role", "neuron_class", "neurotransmitter", "synapse_from",
                 "synapse_to", "synapse_type", "g_max_ns", "delay_ms", "note")


def load_pharynx_subgraph(csv_path: str = PHARYNX_CSV) -> dict:
    """m5_pharynx_subgraph.csv → neurons（roster 序）/ chem / gaps。

    注意：该 CSV 的头行是**引号包裹**的（"role,neuron_class,..."），不能用
    DictReader(fieldnames=None)；用 csv.reader + 显式列名解析。
    """
    neurons, chem, gaps = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in f:
            s = row.strip()
            if not s or s.startswith("#") or s.startswith('"#'):
                continue
            fields = next(csv.reader([row]))
            if len(fields) < 9:
                continue
            role = fields[0].strip()
            stype = fields[5].strip()
            if role and not stype:
                neurons.append(role)
            elif stype == "chem":
                g = float(fields[6]) if fields[6].strip() else 0.0
                if g > 0:  # other/serotonin 行 g=0（调质占位，M6 补齐）→ 跳过
                    chem.append(dict(
                        pre=fields[3].strip(),
                        post=fields[4].strip(),
                        g_max_ns=g,
                        delay_ms=float(fields[7]),
                    ))
            elif stype == "gap":
                gaps.append((fields[3].strip(), fields[4].strip()))
    return dict(neurons=neurons, chem=chem, gaps=gaps)


def load_behavior_bands(csv_path: str = BEHAVIOR_CSV) -> dict:
    """m5_behavior_reference.csv → {(role, neuron_class): dict(lo,hi,tol_lo,tol_hi,...)}。

    该 CSV 头行未加引号 → DictReader(fieldnames=None) 以首行为列名（与咽部 CSV 不同）。
    """
    bands = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(_csv_rows(csv_path))
        for r in reader:
            role = (r["role"] or "").strip()
            nclass = (r["neuron_class"] or "").strip()
            if not role or not nclass:
                continue
            def _f(k, default=None):
                v = (r.get(k) or "").strip()
                return float(v) if v else default
            bands[(role, nclass)] = dict(
                lo=_f("lo"), hi=_f("hi"), tol_lo=_f("tol_lo"), tol_hi=_f("tol_hi"),
                target=_f("target"), provenance=(r.get("provenance") or "").strip(),
                note=(r.get("note") or "").strip(),
            )
    return bands


def load_worm_params(csv_path: str = WORM_PARAMS_CSV) -> dict:
    """m5_worm_params.csv → {role: {neuron_class: value}}（value 数值/字符串）。"""
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(_csv_rows(csv_path))
        for r in reader:
            role = (r["role"] or "").strip()
            nclass = (r["neuron_class"] or "").strip()
            if not role or not nclass:
                continue
            val = (r.get("value") or "").strip()
            if not val:
                val = (r.get("note") or "").strip()
            try:
                val = float(val)
            except ValueError:
                pass
            out.setdefault(role, {})[nclass] = val
    return out


# --------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------- #
def detect_spikes(t_ms: np.ndarray, v_mv: np.ndarray,
                  thresh: float = SPIKE_THRESH_MV,
                  refractory_ms: float = SPIKE_REF_MS) -> np.ndarray:
    """V 上冲过 thresh 的峰时刻（ms）：边沿检测 + 窗口内峰定位 + 去重（M3/M4 同款）。"""
    v = np.asarray(v_mv, dtype=float)
    t = np.asarray(t_ms, dtype=float)
    above = v > thresh
    edges = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    times = []
    for e in edges:
        win = (t >= t[e]) & (t <= t[e] + refractory_ms)
        times.append(float(t[win][np.argmax(v[win])]) if win.sum() else float(t[e]))
    out = []
    for x in times:
        if not out or x - out[-1] > refractory_ms:
            out.append(x)
    return np.asarray(out)


def gaussian_smooth(spikes_ms: np.ndarray, t_ms: np.ndarray,
                    sigma_ms: float) -> np.ndarray:
    """发放事件序列 → 高斯平滑包络（归一化核；尾截断 6σ，泵信号提取）。

    t_ms 为 1ms 均匀网格（泵信号包络，σ=100ms 平滑；6σ=600ms 截断误差 <1e-8）。
    """
    t = np.asarray(t_ms, dtype=float)
    p = np.zeros_like(t)
    norm = sigma_ms * np.sqrt(2.0 * np.pi)
    half = int(np.ceil(6.0 * sigma_ms / (t[1] - t[0]))) if len(t) > 1 else 0
    for sp in np.atleast_1d(spikes_ms):
        if not np.isfinite(sp):
            continue
        i0 = int(np.searchsorted(t, sp))
        i_lo = max(0, i0 - half)
        i_hi = min(len(t), i0 + half + 1)
        seg = t[i_lo:i_hi] - sp
        p[i_lo:i_hi] += np.exp(-0.5 * (seg / sigma_ms) ** 2) / norm
    return p


def psd_peak(freq: np.ndarray, psd: np.ndarray, band_hz=(0.05, 8.0)):
    """PSD 频带内峰值 → (peak_freq, peak_power)。带内无峰返回 nan。"""
    f = np.asarray(freq, dtype=float)
    p = np.asarray(psd, dtype=float)
    m = (f >= band_hz[0]) & (f <= band_hz[1])
    if not m.any():
        return float("nan"), float("nan")
    i = int(np.argmax(p[m]))
    idx = np.flatnonzero(m)[i]
    return float(f[idx]), float(p[idx])


def acf_freq(t_ms: np.ndarray, x: np.ndarray,
             min_lag_ms: float = 200.0, min_acf_frac: float = 0.05) -> float:
    """信号自相关首个显著峰 → 频率 [Hz]（1/滞后）。无显著峰返回 nan。"""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if x.std() == 0:
        return float("nan")
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]  # 滞后 ≥0
    ac = ac / ac[0]
    dt = float(t_ms[1] - t_ms[0]) if len(t_ms) > 1 else 1.0
    lags = np.arange(len(ac)) * dt
    m = lags >= min_lag_ms
    if not m.any():
        return float("nan")
    ac_m, lag_m = ac[m], lags[m]
    # 首个局部极大（两侧低于中心）且幅度足够
    for i in range(1, len(ac_m) - 1):
        if ac_m[i] > ac_m[i - 1] and ac_m[i] >= ac_m[i + 1] and ac_m[i] > min_acf_frac:
            return 1000.0 / lag_m[i]
    return float("nan")


def burst_rate_of(spikes_ms: np.ndarray, gap_ms: float = 100.0) -> float:
    """发放池聚类簇率（泵率事件定义）：ISI>gap 划分簇 → 簇数/时长 [Hz]。

    Avery & Horvitz 泵率 = 泵事件频率；簇内 ISI（~10-40ms）<< 簇间 ISI（0.2-3s），
    gap=100ms 稳健分隔（无食物 0.1-2Hz 与有食物 2-5Hz 均适用）。无簇/单簇返回 nan。
    """
    sp = np.sort(np.asarray(spikes_ms, dtype=float))
    if len(sp) < 2:
        return float("nan")
    t_span = (sp[-1] - sp[0]) / 1000.0      # s
    if t_span <= 0:
        return float("nan")
    n_bursts = 1 + int(np.sum(np.diff(sp) > gap_ms))
    return n_bursts / t_span


def robust_peak_freq(freq: np.ndarray, psd: np.ndarray, acf_freq: float,
                     band_hz=(0.05, 8.0), tol_frac: float = 0.25) -> float:
    """带内稳健主频：PSD 局部极大中与自相关频率最接近者（±25% 内），否则带内 argmax。

    泵信号为起搏池发放的平滑包络，周期图 argmax 可能锁谐波/次谐波（实测坑，见 meta）；
    自相关给出真实周期 → 用其消歧。返回 Hz。
    """
    from scipy.signal import argrelextrema
    f = np.asarray(freq, dtype=float)
    p = np.asarray(psd, dtype=float)
    m = (f >= band_hz[0]) & (f <= band_hz[1])
    if not m.any():
        return float("nan")
    idx = argrelextrema(p, np.greater)[0]
    idx = idx[m[idx]]
    band_max = float(p[m].max())
    if not len(idx):
        i = int(np.argmax(p[m]))
        return float(f[np.flatnonzero(m)[i]])
    cand = idx[p[idx] >= 0.1 * band_max]
    if not len(cand):
        cand = idx
    if np.isfinite(acf_freq) and acf_freq > 0:
        frac = np.abs(f[cand] - acf_freq) / acf_freq
        close = cand[frac <= tol_frac]
        if len(close):
            return float(f[close[np.argmin(np.abs(f[close] - acf_freq))]])
    i = int(np.argmax(p[m]))
    return float(f[np.flatnonzero(m)[i]])


def pump_metrics(spikes_pooled: np.ndarray, t_ms: np.ndarray,
                 band_hz=(0.05, 8.0)) -> dict:
    """泵信号（MCL/MCR/M4 发放池）→ 主频估计族（PSD/自相关/簇率）。

    - peak_freq           ：**稳健主频**（周期图局部极大与自相关消歧——实测次谐波坑；primary）
    - peak_freq_argmax    ：周期图带内 argmax（原始估计，informational）
    - peak_freq_welch     ：Welch 佐证
    - peak_freq_acf       ：自相关首显著峰 → 1/滞后
    - burst_rate          ：簇率（泵事件频率 = Avery & Horvitz 泵率定义；校准与判定首选，稳健）
    - spike_rate          ：发放池事件率（informational）
    - drift               ：半窗簇率相对漂移（P3 节律稳定判据 <50%）
    """
    from scipy import signal
    t_end = float(t_ms[-1])
    t1 = np.arange(0.0, t_end + 1.0, 1.0)          # 1ms 包络网格
    p = gaussian_smooth(spikes_pooled, t1, PUMP_SMOOTH_SIGMA_MS)
    fs = 1000.0
    # 周期图（全窗，频率分辨率 1/T）
    freq, psd = signal.periodogram(p, fs=fs, detrend="linear")
    f_acf = acf_freq(t1, p, min_lag_ms=200.0)
    f_peak = robust_peak_freq(freq, psd, f_acf, band_hz)
    f_argmax, p_peak = psd_peak(freq, psd, band_hz)
    # Welch 佐证
    nperseg = min(8192, len(p))
    fw, pw = signal.welch(p, fs=fs, nperseg=nperseg, detrend="linear")
    f_welch, _ = psd_peak(fw, pw, band_hz)
    # 簇率（泵事件频率；稳健主频）
    burst = burst_rate_of(spikes_pooled, gap_ms=100.0)
    # 事件率：发放池事件数 / 时长
    spike_rate = len(spikes_pooled) / (t_end / 1000.0) if len(spikes_pooled) else 0.0
    # 节律稳定（P3 判据：T≥10s 主频漂移 <50%）：半窗簇率相对漂移（稳健）
    drift = float("nan")
    sp = np.sort(np.asarray(spikes_pooled, dtype=float))
    if len(sp) >= 4 and sp[-1] > sp[0]:
        t_half = (sp[0] + sp[-1]) / 2.0
        b0 = burst_rate_of(sp[sp <= t_half], gap_ms=100.0)
        b1 = burst_rate_of(sp[sp > t_half], gap_ms=100.0)
        if np.isfinite(b0) and np.isfinite(b1) and b0 > 0:
            drift = abs(b1 - b0) / b0
    return dict(peak_freq=f_peak, peak_freq_argmax=f_argmax, peak_power=float(p_peak),
                peak_freq_welch=f_welch, peak_freq_acf=f_acf,
                burst_rate=burst, spike_rate=spike_rate, drift=drift,
                freq=freq, psd=psd, pump_signal=p, t_ms=t1)


# ===================================================================== #
# PART A —— NEURON 咽部化学子图（Stage A；cvode 1e-8, celsius=6.3, v_init=V0）
# ===================================================================== #
def run_neuron_chemical(proto: str, sub: dict, t_total: float = STAGE_A_T_MS,
                        food_drive: float = FOOD_DRIVE_UA_CM2,
                        rec_names=("MCL", "MCR", "M4", "M5", "I2L", "MI"),
                        morph_spec=None) -> dict:
    """构建 20 神经元化学子图（build_neuron + ExpSyn + NetCon）并运行。

    proto: 'no_food'（无外部输入）/ 'food'（tonic IClamp 到全部神经元，食物广义兴奋简化）。
    返回 dict(spike_times={name: np.ndarray}, v_traces={name: np.ndarray}, t_ms)。
    """
    from neuron import h
    from neural_exploration.src.synapse_model import load_synapse_params

    neurons = sub["neurons"]
    morph = morph_spec if morph_spec is not None else globals().get("spec")
    if morph is None:
        morph = load_morphology()
    secs = {}
    first = True
    for name in neurons:
        secs[name.lower()] = build_neuron(morph, clear=first,
                                          name_prefix=f"{name.lower()}_")
        first = False

    h.load_file("stdrun.hoc")
    h.celsius = 6.3                     # 硬约束：Q10 参考温度（SESSION_CONTEXT §四 #2）
    h.v_init = V0

    # 点过程引用列表持有防 GC（M2 L8）
    clamps, syns, ncs = [], [], []

    # 食物驱动：tonic IClamp（soma；密度→nA 按 SOMA_AREA_CM2 换算）
    if proto == "food" and food_drive > 0:
        drive_set = FOOD_DRIVE_SET if FOOD_DRIVE_SET is not None else neurons
        amp = food_drive * 1e-6 * SOMA_AREA_CM2 * 1e9
        for name in drive_set:
            cl = h.IClamp(secs[name.lower()]["soma"](0.5))
            cl.delay = 0.0
            cl.dur = 1e9                 # 全窗持续（食物是持续刺激）
            cl.amp = amp
            clamps.append(cl)

    # 化学突触：ExpSyn（ampa τ/E 沿 m2 行）+ NetCon(pre node3 → post soma)
    m2 = load_synapse_params()
    base = m2["ampa"]
    for s in sub["chem"]:
        syn = h.ExpSyn(secs[s["post"].lower()]["soma"](0.5))
        syn.tau = base.tau_ms
        syn.e = base.e_rev_mv
        nc = h.NetCon(secs[s["pre"].lower()]["node3"](0.5)._ref_v, syn,
                      sec=secs[s["pre"].lower()]["node3"])
        nc.threshold = NC_THRESH_MV
        nc.delay = s["delay_ms"]
        nc.weight[0] = s["g_max_ns"] * 1e-3      # ExpSyn weight 单位 µS
        syns.append(syn)
        ncs.append(nc)

    # 记录：全部神经元 node3 V + 关键神经元 soma V
    tvec = h.Vector(); tvec.record(h._ref_t)
    vrec = {}
    for name in neurons:
        vv = h.Vector(); vv.record(secs[name.lower()]["node3"](0.5)._ref_v)
        vrec[name] = vv

    cvode = h.CVode()
    cvode.active(1)
    cvode.atol(1e-8)                     # 硬约束：参考真理容差
    cvode.rtol(1e-8)
    h.tstop = t_total
    h.run()

    # 重采样到均匀网格（dt=0.1ms）
    t_irr = np.array(tvec)
    order = np.argsort(t_irr)
    t_u = np.arange(0.0, t_total + DT_OUT / 2, DT_OUT)
    spikes, vtraces = {}, {}
    for name, vv in vrec.items():
        v = np.interp(t_u, t_irr[order], np.array(vv)[order])
        if name in rec_names:
            vtraces[name] = v
        spikes[name] = detect_spikes(t_u, v)
    return dict(spike_times=spikes, v_traces=vtraces, t_ms=t_u)


# ===================================================================== #
# PART B —— scipy solve_ivp 缝隙耦合节奏网络（Stage B；LSODA 高精度）
# ===================================================================== #
def build_gap_adjacency(sub: dict, names: list) -> tuple:
    """33 条缝隙 → 无向邻接（含自缝隙；自缝隙 ΔV=0 无害保留——连接组事实）。"""
    n = len(names)
    idx = {nm: i for i, nm in enumerate(names)}
    pairs = []
    for a, b in sub["gaps"]:
        if a not in idx or b not in idx:
            continue
        i, j = idx[a], idx[b]
        pairs.append((i, j))
    return pairs


def chemical_conductance_trace(spikes_by_pre: dict, chem: list, names: list,
                               t_grid_ms: np.ndarray, tau_ms: float = AMPA_TAU_MS) -> np.ndarray:
    """Stage-A 发放 → 逐突触后神经元总电导时间历程 G_i(t) [nS]。

    G_i(t) = Σ_{j→i} g_max_ji·Σ_k exp(−(t−t_k)/τ)·1[t ≥ t_k]（ExpSyn 解析解，dt 网格）。
    指数尾截断至 12τ（exp(−12)≈6e-6，参考精度足够）。返回 (n_neurons, n_t) 数组。
    """
    n = len(names)
    n_t = len(t_grid_ms)
    G = np.zeros((n, n_t))
    tail = int(np.ceil(12.0 * tau_ms / (t_grid_ms[1] - t_grid_ms[0]))) if n_t > 1 else 0
    for s in chem:
        try:
            post = names.index(s["post"])
        except ValueError:
            continue
        sp = np.asarray(spikes_by_pre.get(s["pre"], np.asarray([])), dtype=float)
        sp = sp[sp <= t_grid_ms[-1]]
        if not len(sp):
            continue
        g = s["g_max_ns"]
        for t_k in sp:
            i0 = int(np.searchsorted(t_grid_ms, t_k))
            if i0 >= n_t:
                continue
            i1 = min(n_t, i0 + tail)
            seg = t_grid_ms[i0:i1] - t_k
            G[post, i0:i1] += g * np.exp(-seg / tau_ms)
    return G


def _make_stage_b_rhs(G_chem, gap_pairs, names, pacemaker_set, sahp,
                      drive_vec, w_chem, chem_active):
    """Stage-B 右端函数（20 点 HH + 缝隙 + slow-AHP + 化学输入）。

    sahp: dict(g_sahp_vec, tau_vec, theta, k)；drive_vec: 逐神经元 tonic 驱动 [µA/cm²]。
    返回 f(t, y)；状态 y = [v0..v19, m0.., h0.., n0.., w0..]（每神经元 5 变量）。
    """
    n = len(names)
    g_sahp = sahp["g_sahp"]          # (n,) µA/cm² 每神经元（非起搏=0）
    tau_sahp = sahp["tau"]           # (n,)
    theta, k = sahp["theta"], sahp["k"]
    e_syn = AMPA_E_MV
    g_gap_s = GAP_G_NS * 1e-9
    area = AREA_CM2
    t_grid = G_chem_t_grid

    def f(t, y):
        v = y[0:n]
        m = y[n:2 * n]
        h = y[2 * n:3 * n]
        nn = y[3 * n:4 * n]
        w = y[4 * n:5 * n]
        # HH 门控速率（hh_spec 6.3°C 参考）
        am = 0.1 * (v + 40.0) / (1.0 - np.exp(-(v + 40.0) / 10.0))
        bm = 4.0 * np.exp(-(v + 65.0) / 18.0)
        ah = 0.07 * np.exp(-(v + 65.0) / 20.0)
        bh = 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))
        an = 0.01 * (v + 55.0) / (1.0 - np.exp(-(v + 55.0) / 10.0))
        bn = 0.125 * np.exp(-(v + 65.0) / 80.0)
        i_na = GNA * m ** 3 * h * (v - ENA)
        i_k = GK * nn ** 4 * (v - EK)
        i_l = GL * (v - EL)
        # 缝隙电流（M2 单位：g[S]·ΔV[mV]·1e3/area → µA/cm²）
        i_gap = np.zeros(n)
        for i, j in gap_pairs:
            d = g_gap_s * (v[j] - v[i]) * 1e3 / area
            i_gap[i] += d
            i_gap[j] -= d
        # 化学输入：I_syn = w_chem·G_i(t)·(E−V)·1e-6/area（驱动势用实时 V）
        i_syn = np.zeros(n)
        if chem_active:
            for i in range(n):
                gi = float(np.interp(t, t_grid, G_chem[i]))
                i_syn[i] = w_chem * gi * (e_syn - v[i]) * 1e-6 / area
        # slow-AHP（泵马达池）
        i_sahp = -g_sahp * w * (v - EK)
        winf = 1.0 / (1.0 + np.exp(-(v - theta) / k))
        dv = (drive_vec + i_gap + i_syn + i_sahp - i_na - i_k - i_l) / CM
        dw = (winf - w) / tau_sahp
        return np.concatenate([dv, am * (1 - m) - bm * m, ah * (1 - h) - bh * h,
                               an * (1 - nn) - bn * nn, dw])

    return f


G_chem_t_grid = None   # 模块级：化学电导时间网格（f 闭包引用）


def run_stage_b(names, gap_pairs, G_chem, drive_vec, sahp, w_chem=0.0,
                t_total: float = STAGE_B_T_MS, chem_active: bool = True,
                rtol: float = STAGE_B_RTOL, atol: float = STAGE_B_ATOL) -> dict:
    """solve_ivp LSODA 求解 Stage-B → V(t)（dt=0.1ms 输出 + dt=1ms 存档）。"""
    from scipy.integrate import solve_ivp

    global G_chem_t_grid
    G_chem_t_grid = np.arange(0.0, t_total + DT_OUT / 2, DT_OUT)

    n = len(names)
    m0, h0, n0 = steady_state(V0)
    y0 = np.zeros(5 * n)
    for i in range(n):
        y0[i] = V0
        y0[n + i] = m0
        y0[2 * n + i] = h0
        y0[3 * n + i] = n0
        y0[4 * n + i] = 0.0
    f = _make_stage_b_rhs(G_chem, gap_pairs, names, set(PACEMAKER), sahp,
                          drive_vec, w_chem, chem_active)
    sol = solve_ivp(f, (0.0, t_total), y0, method="LSODA",
                    rtol=rtol, atol=atol, dense_output=True)
    t_u = np.arange(0.0, t_total + DT_OUT / 2, DT_OUT)
    ys = sol.sol(t_u)                      # (5n, n_t)
    v_all = ys[0:n, :]
    t_ds = np.arange(0.0, t_total + 1.0, 1.0)   # dt=1ms 存档
    v_ds = np.asarray([np.interp(t_ds, t_u, v_all[i]) for i in range(n)])
    return dict(v_all=v_all, t_ms=t_u, v_ds=v_ds, t_ds_ms=t_ds, ok=sol.success)


def calibrate_pharynx(sub, names, stage_a: dict) -> dict:
    """Stage-B 协议参数校准（无食物/有食物各自落带；确定性网格扫描）。

    返回 dict(nofood=dict, food=dict, scan_nofood=[...], scan_food=[...])。
    """
    gap_pairs = build_gap_adjacency(sub, names)
    n = len(names)
    pm_idx = [names.index(x) for x in PACEMAKER]

    def _sahep(g_sahp, tau):
        gv = np.zeros(n)
        tv = np.full(n, tau)
        for i in pm_idx:
            gv[i] = g_sahp
        return dict(g_sahp=gv, tau=tv, theta=SAHP_THETA_MV, k=SAHP_K_MV)

    def _drive(amp):
        d = np.zeros(n)
        for i in pm_idx:
            d[i] = amp
        return d

    def _rate(params, proto, t_total):
        sahp = _sahep(params["g_sahp"], params["tau"])
        G_chem = np.zeros((n, len(np.arange(0.0, t_total + DT_OUT / 2, DT_OUT))))
        chem_active = proto == "food" and params.get("w_chem", 0.0) > 0
        if chem_active:
            G_chem = chemical_conductance_trace(
                stage_a["food"]["spike_times"], sub["chem"], names,
                np.arange(0.0, t_total + DT_OUT / 2, DT_OUT))
        r = run_stage_b(names, gap_pairs, G_chem, _drive(params["I"]), sahp,
                        w_chem=params.get("w_chem", 0.0),
                        t_total=t_total, chem_active=chem_active,
                        rtol=STAGE_B_RTOL, atol=STAGE_B_ATOL)
        pooled = []
        for x in PACEMAKER:
            pooled.extend(detect_spikes(r["t_ms"], r["v_all"][names.index(x), :]))
        pooled = np.sort(np.asarray(pooled))
        pm = pump_metrics(pooled, r["t_ms"])
        return pm

    # ---- 无食物：I × g_sahp × τ → 簇率目标 0.5Hz（带 [0.1,2]）----
    scan_nf = []
    best_nf = None
    for I in NOFOOD_CAL["I"]:
        for g in NOFOOD_CAL["g_sahp"]:
            for tau in NOFOOD_CAL["tau"]:
                pm = _rate(dict(I=I, g_sahp=g, tau=tau, w_chem=0.0), "no_food",
                           CAL_NOFOOD_T_MS)
                f = pm["burst_rate"]
                row = dict(I=I, g_sahp=g, tau=tau, burst_rate=f,
                           peak_freq=pm["peak_freq"])
                scan_nf.append(row)
                if np.isfinite(f) and 0.1 <= f <= 2.0:
                    key = abs(f - NO_FOOD_TARGET_HZ)
                    if best_nf is None or key < best_nf[0]:
                        best_nf = (key, dict(I=I, g_sahp=g, tau=tau, w_chem=0.0))
    if best_nf is None:  # 全网格无解 → 取最接近带的组合（记录测量限制）
        cand = min(scan_nf, key=lambda r: min(abs(r["burst_rate"] - 0.1),
                                              abs(r["burst_rate"] - 2.0)))
        best_nf = (float("inf"), dict(I=cand["I"], g_sahp=cand["g_sahp"],
                                      tau=cand["tau"], w_chem=0.0))
    # ---- 有食物：I × w_chem（g_sahp=8, τ=200 起搏参数）→ 簇率目标 3.5Hz（带 [2,5]）----
    scan_f = []
    best_f = None
    for I in FOOD_CAL["I"]:
        for g in FOOD_CAL["g_sahp"]:
            for tau in FOOD_CAL["tau"]:
                for wc in FOOD_CAL["w_chem"]:
                    pm = _rate(dict(I=I, g_sahp=g, tau=tau, w_chem=wc), "food",
                               CAL_FOOD_T_MS)
                    f = pm["burst_rate"]
                    row = dict(I=I, g_sahp=g, tau=tau, w_chem=wc, burst_rate=f,
                               peak_freq=pm["peak_freq"])
                    scan_f.append(row)
                    if np.isfinite(f) and 2.0 <= f <= 5.0:
                        key = abs(f - FOOD_TARGET_HZ)
                        if best_f is None or key < best_f[0]:
                            best_f = (key, dict(I=I, g_sahp=g, tau=tau, w_chem=wc))
    if best_f is None:
        cand = min(scan_f, key=lambda r: min(abs(r["burst_rate"] - 2.0),
                                             abs(r["burst_rate"] - 5.0)))
        best_f = (float("inf"), dict(I=cand["I"], g_sahp=cand["g_sahp"],
                                     tau=cand["tau"], w_chem=cand["w_chem"]))
    return dict(nofood=dict(params=best_nf[1], out_of_band=not np.isfinite(best_nf[0])),
                food=dict(params=best_f[1], out_of_band=not np.isfinite(best_f[0])),
                scan_nofood=scan_nf, scan_food=scan_f)


# ===================================================================== #
# PART C —— 行为参考共用工具（numpy 参考版；共用约定见模块 docstring 与函数注释）
# ===================================================================== #
def body_velocity(c_fwd: np.ndarray, c_back: np.ndarray,
                  v_fwd0: float = 1.0, v_rev0: float = 1.0) -> np.ndarray:
    """M5 身体方程（规格 §5.2 #3）：v = v_fwd0·C_fwd − v_rev0·C_back。

    【共用约定】B1d 的 src/virtual_body.py 须与本站点同签名同语义（行为参考与
    Brian2 全虫共用同一实现，不得复制粘贴——M5 清单 §9 风险表）。
    """
    c_f = np.asarray(c_fwd, dtype=float)
    c_b = np.asarray(c_back, dtype=float)
    return v_fwd0 * np.clip(c_f, 0.0, 1.0) - v_rev0 * np.clip(c_b, 0.0, 1.0)


def classify_state(v, omega, c_fwd, c_back, v_thr, omega_thr,
                   rev_margin: float = 1e-9) -> str:
    """自发行为状态分类（前进/后退/转弯/暂停）——numpy 参考版。

    判定顺序（定稿，不做事后调；阈值来自 m5_worm_params.csv protocol.spont_*_thr_frac）：
      1. C_back > C_fwd        → 'rev'   （后退命令主导，优先级最高——逃避语义）
      2. |ω| > ω_thr           → 'turn'  （转向/头摆）
      3. |v| < v_thr           → 'pause' （静止）
      4. 其余                   → 'fwd'  （前进）

    【共用约定】B1d 的 src/virtual_body.py 的 classify_state 与本站点同签名同语义；
    P6 判定脚本从单一实现调用（行为参考不得复制粘贴——M5 清单 §9 风险表）。
    """
    if float(c_back) > float(c_fwd) + rev_margin:
        return "rev"
    if abs(float(omega)) > float(omega_thr):
        return "turn"
    if abs(float(v)) < float(v_thr):
        return "pause"
    return "fwd"


def escape_latencies(t_ms, c_back, c_fwd, v, touch_start_ms,
                     c_thr_frac: float = B_OUT_THR_FRAC) -> dict:
    """从运动轨迹计算 P5 潜伏期（共用统计；两引擎同一实现）。

    - nerve_latency_ms        ：首个 C_back 上升（>1e-6）时刻 − touch_start
                                （运动命令起始；M5 操作化：神经潜伏期判据 [5,20]ms 以
                                M3 已验证神经链为准，转导延迟不计入——见模块 docstring）
    - behavior_latency_ms     ：首个 C_back ≥ c_thr_frac·C_back_peak − touch_start（M3 同款）
    - behavior_latency_v0_ms  ：首个 v<0 − touch_start（后退运动实际开始）
    - c_back_peak / d_peak    ：方向判据（D_peak>0.3 或 C_back_peak>C_fwd_peak）
    【共用约定】同 body_velocity/classify_state（M5 清单 §9）。
    """
    t = np.asarray(t_ms, dtype=float)
    cb = np.asarray(c_back, dtype=float)
    cf = np.asarray(c_fwd, dtype=float)
    vv = np.asarray(v, dtype=float)

    def _first(cond):
        idx = np.flatnonzero(cond)
        return float(t[idx[0]] - touch_start_ms) if len(idx) else float("nan")

    nerve = _first(cb > 1e-6)
    peak = float(cb.max()) if len(cb) else 0.0
    beh = _first(cb >= c_thr_frac * peak) if peak > 0 else float("nan")
    beh_v0 = _first(vv < 0.0)
    d_peak = float((cb - cf).max()) if len(cb) else 0.0
    direction = "back" if (peak > float(cf.max()) or d_peak > 0.3) else "fwd"
    return dict(nerve_latency_ms=nerve, behavior_latency_ms=beh,
                behavior_latency_v0_ms=beh_v0, c_back_peak=peak, d_peak=d_peak,
                direction=direction)


# ===================================================================== #
# PART D —— 逃避参考（P5；numpy 运动学 + 转导延迟 + M3 一致性）
# ===================================================================== #
def escape_reference(n_trials: int = N_ESCAPE_TRIALS, seed: int = RNG_SEED) -> dict:
    """触刺激 → 后退 numpy 运动学参考（N=20，确定性 seed）。

    时间线（单位 ms；t_touch=0）：
      τ_trans（转导：触→感觉电流，N(23,2) 锚定）→ L_nerve（神经链，M3 实测 latency_nerve
      抽样，∈[5,20]）→ t_cmd = τ_trans + L_nerve（DA 运动命令）→ C_back 收缩上升
      （w_back·(1−e^{−Δt/τ_mus})，τ_mus=20ms M3 值）+ C_fwd 抑制（基线 0.197 指数衰减）→
      v = C_fwd − C_back（M5 身体方程）→ 行为潜伏期 ∈ [30,50]（Chalfie 1985 窗）。
    """
    rng = np.random.default_rng(seed)
    d3 = np.load(M3_REF_NPZ, allow_pickle=True)
    m3_nerve = d3["latency_nerve"]           # 各档触→DA 潜伏期（M3 实测）
    m3_nerve = np.asarray([x for x in m3_nerve if np.isfinite(x) and 5.0 <= x <= 20.0])
    if not len(m3_nerve):
        m3_nerve = np.asarray([10.0])

    t_total = 150.0
    t = np.arange(0.0, t_total + 0.1, 0.1)
    tau_trans = TAU_TRANS_MS + TAU_TRANS_JIT_MS * rng.standard_normal(n_trials)
    tau_trans = np.clip(tau_trans, 15.0, 30.0)
    nerve = rng.choice(m3_nerve, size=n_trials)
    t_cmd = tau_trans + nerve
    rise = -MUSCLE_TAU_MS * np.log(1.0 - B_OUT_THR_FRAC)   # 7.13ms（0.3·peak 定义）

    lat_chain_all, lat_full_all, lat_beh_all, lat_beh_v0_all, dirs = [], [], [], [], []
    traj = None
    for k in range(n_trials):
        c_back = np.where(t >= t_cmd[k],
                          MUSCLE_W_BACK * (1.0 - np.exp(-(t - t_cmd[k]) / MUSCLE_TAU_MS)), 0.0)
        c_fwd = np.where(t >= t_cmd[k],
                         C_FWD_BASELINE * np.exp(-(t - t_cmd[k]) / C_SUPP_TAU_MS),
                         C_FWD_BASELINE)
        v = body_velocity(c_fwd, c_back)
        x = np.concatenate([[0.0], np.cumsum(v) * 1e-4])   # 1D 位移（dt=0.1ms → s）
        es = escape_latencies(t, c_back, c_fwd, v, touch_start_ms=0.0)
        # P5 神经潜伏期窗 [5,20] 操作化 = 神经链时间（触电流注入→运动发放，M3 可比），
        # 不含转导延迟（清单 §5.2 #4：行为 − 神经 ≈ 转导+肌肉 10-30ms）
        lat_chain_all.append(float(nerve[k]))
        lat_full_all.append(es["nerve_latency_ms"])       # C_back 上升起始 − touch（含转导）
        lat_beh_all.append(es["behavior_latency_ms"])
        lat_beh_v0_all.append(es["behavior_latency_v0_ms"])
        dirs.append(es["direction"])
        if k == 0:
            traj = dict(t_ms=t, c_back=c_back, c_fwd=c_fwd, v=v, x=x,
                        tau_trans_ms=float(tau_trans[k]), t_cmd_ms=float(t_cmd[k]))
    lat_chain_all = np.asarray(lat_chain_all)
    lat_full_all = np.asarray(lat_full_all)
    lat_beh_all = np.asarray(lat_beh_all)
    lat_beh_v0_all = np.asarray(lat_beh_v0_all)

    prob = float(np.mean([d == "back" for d in dirs]))
    d_peak = float(np.max([np.max(np.where(t >= t_cmd[k], MUSCLE_W_BACK * (
        1.0 - np.exp(-(t - t_cmd[k]) / MUSCLE_TAU_MS)), 0.0) - np.where(
            t >= t_cmd[k], C_FWD_BASELINE * np.exp(-(t - t_cmd[k]) / C_SUPP_TAU_MS),
            C_FWD_BASELINE)) for k in range(n_trials)]))

    bands = load_behavior_bands()
    nerve_band = bands[("escape", "nerve_latency_ms")]
    beh_band = bands[("escape", "behavior_latency_ms")]
    return dict(
        n_trials=n_trials,
        direction=np.asarray(dirs),
        reaction_probability=prob,
        nerve_latency_ms=lat_chain_all,       # 神经链（P5 窗 [5,20]，不含转导）
        nerve_latency_full_ms=lat_full_all,   # C_back 上升起始 − touch（含转导，informational）
        behavior_latency_ms=lat_beh_all,
        behavior_latency_v0_ms=lat_beh_v0_all,
        tau_trans_ms=np.asarray(tau_trans),
        d_peak=d_peak,
        c_back_peak=float(np.max([np.max(np.where(t >= t_cmd[k], MUSCLE_W_BACK * (
            1.0 - np.exp(-(t - t_cmd[k]) / MUSCLE_TAU_MS)), 0.0)) for k in range(n_trials)])),
        nerve_in_band=float(np.mean((lat_chain_all >= nerve_band["lo"]) &
                                    (lat_chain_all <= nerve_band["hi"]))),
        behavior_in_band=float(np.mean((lat_beh_all >= beh_band["tol_lo"]) &
                                       (lat_beh_all <= beh_band["tol_hi"]))),
        behavior_mean_ms=float(np.nanmean(lat_beh_all)),
        behavior_std_ms=float(np.nanstd(lat_beh_all)),
        traj=traj,
        m3_nerve_sampled=m3_nerve,
        band_nerve=[nerve_band["lo"], nerve_band["hi"]],
        band_behavior=[beh_band["lo"], beh_band["hi"]],
        tol_behavior=[beh_band["tol_lo"], beh_band["tol_hi"]],
    )


# ===================================================================== #
# PART E —— 自发行为参考（P6；bout 状态马尔可夫 + classify_state 共用）
# ===================================================================== #
_STATES = ("fwd", "rev", "turn", "pause")
_STATE_IDX = {s: i for i, s in enumerate(_STATES)}
# 每状态代表 (v, ω, C_fwd, C_back)（M5 身体方程自洽；ω 转向符号逐 bout 交替）
_STATE_MOTION = dict(
    fwd=dict(v=1.0, omega=0.0, c_fwd=1.0, c_back=0.0),
    rev=dict(v=-1.0, omega=0.0, c_fwd=0.0, c_back=1.0),
    turn=dict(v=0.5, omega=1.0, c_fwd=0.5, c_back=0.0),
    pause=dict(v=0.0, omega=0.0, c_fwd=0.0, c_back=0.0),
)


def _transition_matrix(p_fr, p_ft, p_rf, p_tf) -> np.ndarray:
    """嵌入链转移矩阵（行=当前，列=下一状态；行列序 fwd/rev/turn/pause）。"""
    p_fp = 0.02
    p_rp = 0.0
    p_tp = 0.0
    P = np.array([
        [1.0 - p_fr - p_ft - p_fp, p_fr, p_ft, p_fp],
        [p_rf, 1.0 - p_rf - 0.05, 0.05, p_rp],
        [p_tf, 0.05, 1.0 - p_tf - 0.05, p_tp],
        [0.70, 0.0, 0.05, 0.25],
    ])
    P[P < 0] = 0.0
    P = P / P.sum(axis=1, keepdims=True)
    return P


def _stationary(P: np.ndarray) -> np.ndarray:
    """嵌入链平稳分布（幂迭代）。"""
    pi = np.ones(len(P)) / len(P)
    for _ in range(10000):
        pi_new = pi @ P
        if np.max(np.abs(pi_new - pi)) < 1e-14:
            return pi_new
        pi = pi_new
    return pi


def _simulate_bouts(P: np.ndarray, t_total_ms: float, dt_b_ms: float,
                    bout_mean_ms: dict, rng, initial=None) -> dict:
    """半马尔可夫模拟：嵌入链 + 指数 bout 时长 → (v, ω, C_fwd, C_back, state) 序列。"""
    n_steps = int(round(t_total_ms / dt_b_ms))
    t = np.arange(0.0, n_steps) * dt_b_ms
    v = np.zeros(n_steps)
    omega = np.zeros(n_steps)
    cf = np.zeros(n_steps)
    cb = np.zeros(n_steps)
    state_seq = np.empty(n_steps, dtype=object)
    cur = initial if initial is not None else int(rng.choice(len(_STATES), p=_stationary(P)))
    turn_sign = 1.0
    i = 0
    bout_lengths = {s: [] for s in _STATES}
    while i < n_steps:
        dur_ms = rng.exponential(bout_mean_ms[_STATES[cur]])
        dur_steps = max(1, int(round(dur_ms / dt_b_ms)))
        mot = _STATE_MOTION[_STATES[cur]]
        if _STATES[cur] == "turn":
            turn_sign = -turn_sign
        for k in range(min(dur_steps, n_steps - i)):
            om = turn_sign * mot["omega"] if _STATES[cur] == "turn" else mot["omega"]
            v[i + k] = 1.0 * mot["c_fwd"] - 1.0 * mot["c_back"]   # M5 身体方程（v_fwd0=v_rev0=1）
            omega[i + k] = om
            cf[i + k] = mot["c_fwd"]
            cb[i + k] = mot["c_back"]
            state_seq[i + k] = _STATES[cur]
        bout_lengths[_STATES[cur]].append(dur_ms)
        i += dur_steps
        nxt = int(rng.choice(len(_STATES), p=P[cur]))
        cur = nxt
    return dict(t_ms=t, v=v, omega=omega, c_fwd=cf, c_back=cb,
                state=state_seq, bout_lengths=bout_lengths)


def calibrate_spontaneous(seed: int = RNG_SEED, n_cal_trials: int = 10) -> dict:
    """转移矩阵校准：网格扫描（每组合 N=10 试次）→ 时间比例全部落带且带宽裕度最大。

    Srivastava 2013 带：前进 [60,80]/后退 [10,25]/转弯 [5,20]%。N=10 降抽样噪声；
    选带内裕度最大组合（防小样本边缘漂移），次优距离带中心。
    """
    bands = load_behavior_bands()
    fwd_b = bands[("spontaneous", "time_fraction_fwd")]
    rev_b = bands[("spontaneous", "time_fraction_rev")]
    turn_b = bands[("spontaneous", "time_fraction_turn")]
    v_thr = 0.05
    omega_thr = 0.2
    best = None
    for p_fr in SPONT_CAL["p_fr"]:
        for p_ft in SPONT_CAL["p_ft"]:
            for p_rf in SPONT_CAL["p_rf"]:
                for p_tf in SPONT_CAL["p_tf"]:
                    P = _transition_matrix(p_fr, p_ft, p_rf, p_tf)
                    rng = np.random.default_rng(seed)
                    cnt = {s: 0.0 for s in _STATES}
                    for tr in range(n_cal_trials):
                        sim = _simulate_bouts(P, SPONT_T_MS, DT_B_MS, BOUT_MEAN_MS, rng)
                        st = np.asarray([classify_state(sim["v"][i], sim["omega"][i],
                                                        sim["c_fwd"][i], sim["c_back"][i],
                                                        v_thr, omega_thr)
                                         for i in range(len(sim["v"]))])
                        for s in _STATES:
                            cnt[s] += float((st == s).mean())
                    props = {s: cnt[s] / n_cal_trials * 100.0 for s in _STATES}
                    ok = (fwd_b["lo"] <= props["fwd"] <= fwd_b["hi"]
                          and rev_b["lo"] <= props["rev"] <= rev_b["hi"]
                          and turn_b["lo"] <= props["turn"] <= turn_b["hi"])
                    if ok:
                        margin = min(props["fwd"] - fwd_b["lo"], fwd_b["hi"] - props["fwd"],
                                     props["rev"] - rev_b["lo"], rev_b["hi"] - props["rev"],
                                     props["turn"] - turn_b["lo"], turn_b["hi"] - props["turn"])
                        dist = (abs(props["fwd"] - fwd_b["target"]) +
                                abs(props["rev"] - rev_b["target"]) +
                                abs(props["turn"] - turn_b["target"]))
                        key = (-margin, dist)
                        if best is None or key < best[0]:
                            best = (key, dict(P=P, props=props,
                                              knobs=dict(p_fr=p_fr, p_ft=p_ft,
                                                         p_rf=p_rf, p_tf=p_tf)))
    if best is None:
        # 全网格无解：取 fwd 带内且 rev/turn 距离带最近（记录测量限制）
        best = (None, dict(P=_transition_matrix(0.35, 0.35, 0.8, 0.7),
                           props=None, knobs=None))
    return dict(P=best[1]["P"], props=best[1]["props"], knobs=best[1]["knobs"],
                out_of_band=best[1]["props"] is None,
                v_thr=v_thr, omega_thr=omega_thr)


def spontaneous_reference(n_trials: int = N_SPONT_TRIALS, seed: int = RNG_SEED) -> dict:
    """自发行为参考：校准矩阵 + N=10×T=30s 模拟 → 比例/转移矩阵/bout 时长。"""
    cal = calibrate_spontaneous(seed=seed)
    P = cal["P"]
    pi = _stationary(P)
    v_thr, omega_thr = cal["v_thr"], cal["omega_thr"]
    all_props = {s: [] for s in _STATES}
    bout_stats = {s: [] for s in _STATES}
    trace = None
    for tr in range(n_trials):
        rng = np.random.default_rng(seed + 1000 * (tr + 1))
        sim = _simulate_bouts(P, SPONT_T_MS, DT_B_MS, BOUT_MEAN_MS, rng)
        st = np.asarray([classify_state(sim["v"][i], sim["omega"][i],
                                        sim["c_fwd"][i], sim["c_back"][i],
                                        v_thr, omega_thr)
                         for i in range(len(sim["v"]))])
        for s in _STATES:
            all_props[s].append(float((st == s).mean()) * 100.0)
            bl = sim["bout_lengths"][s]
            if bl:
                bout_stats[s].append(float(np.mean(bl)))
        if tr == 0:
            trace = dict(t_ms=sim["t_ms"], v=sim["v"], omega=sim["omega"],
                         c_fwd=sim["c_fwd"], c_back=sim["c_back"],
                         state=np.asarray(st))
    props_mean = {s: float(np.mean(all_props[s])) for s in _STATES}
    props_sem = {s: float(np.std(all_props[s], ddof=1) / np.sqrt(n_trials))
                 if n_trials > 1 else 0.0 for s in _STATES}
    bout_mean = {s: (float(np.mean(bout_stats[s])) if bout_stats[s] else float("nan"))
                 for s in _STATES}
    bands = load_behavior_bands()
    fwd_b = bands[("spontaneous", "time_fraction_fwd")]
    rev_b = bands[("spontaneous", "time_fraction_rev")]
    turn_b = bands[("spontaneous", "time_fraction_turn")]
    return dict(
        n_trials=n_trials, t_total_ms=SPONT_T_MS, dt_b_ms=DT_B_MS,
        states=list(_STATES),
        transition_matrix=P,
        stationary_embedded=pi,
        time_fraction_fwd_pct=props_mean["fwd"], time_fraction_fwd_sem=props_sem["fwd"],
        time_fraction_rev_pct=props_mean["rev"], time_fraction_rev_sem=props_sem["rev"],
        time_fraction_turn_pct=props_mean["turn"], time_fraction_turn_sem=props_sem["turn"],
        time_fraction_pause_pct=props_mean["pause"],
        band_fwd=[fwd_b["lo"], fwd_b["hi"]], band_rev=[rev_b["lo"], rev_b["hi"]],
        band_turn=[turn_b["lo"], turn_b["hi"]],
        fwd_in_band=bool(fwd_b["lo"] <= props_mean["fwd"] <= fwd_b["hi"]),
        rev_in_band=bool(rev_b["lo"] <= props_mean["rev"] <= rev_b["hi"]),
        turn_in_band=bool(turn_b["lo"] <= props_mean["turn"] <= turn_b["hi"]),
        bout_mean_ms=bout_mean,
        cal_knobs=cal["knobs"], out_of_band=cal["out_of_band"],
        classify_thresholds=dict(v_thr=v_thr, omega_thr=omega_thr),
        trace=trace,
    )


# ===================================================================== #
# 主入口
# ===================================================================== #
def run_reference(out_npz: str = REF_NPZ) -> str:
    """Stage-A（NEURON 化学子图）→ Stage-B（缝隙节奏）→ 行为参考 → 落盘 npz。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    global spec
    spec = load_morphology()

    sub = load_pharynx_subgraph()
    names = sub["neurons"]
    bands = load_behavior_bands()

    # ---- Stage A：NEURON 化学子图（无食物/有食物）----
    print("[1/4] NEURON 咽部化学子图（no_food / food）...")
    stage_a = {}
    stage_a["no_food"] = run_neuron_chemical("no_food", sub, morph_spec=spec)
    stage_a["food"] = run_neuron_chemical("food", sub, morph_spec=spec)

    # ---- Stage B 协议校准 + 最终运行 ----
    print("[2/4] Stage-B 缝隙网络校准（无食物/有食物）...")
    cal = calibrate_pharynx(sub, names, stage_a)
    gap_pairs = build_gap_adjacency(sub, names)
    n = len(names)
    pm_idx = [names.index(x) for x in PACEMAKER]
    t_grid = np.arange(0.0, STAGE_B_T_MS + DT_OUT / 2, DT_OUT)
    G_nofood = np.zeros((n, len(t_grid)))
    G_food = chemical_conductance_trace(stage_a["food"]["spike_times"], sub["chem"],
                                        names, t_grid)

    def _drive(amp):
        d = np.zeros(n)
        for i in pm_idx:
            d[i] = amp
        return d

    def _sahp(g_sahp, tau):
        gv = np.zeros(n)
        tv = np.full(n, tau)
        for i in pm_idx:
            gv[i] = g_sahp
        return dict(g_sahp=gv, tau=tv, theta=SAHP_THETA_MV, k=SAHP_K_MV)

    runs = {}
    for proto, G, params in (("no_food", G_nofood, cal["nofood"]["params"]),
                             ("food", G_food, cal["food"]["params"])):
        r = run_stage_b(names, gap_pairs, G, _drive(params["I"]),
                        _sahp(params["g_sahp"], params["tau"]),
                        w_chem=params.get("w_chem", 0.0), t_total=STAGE_B_T_MS,
                        chem_active=proto == "food" and params.get("w_chem", 0.0) > 0,
                        rtol=STAGE_B_RTOL, atol=STAGE_B_ATOL)
        pooled = np.sort(np.concatenate([detect_spikes(r["t_ms"], r["v_all"][names.index(x), :])
                                         for x in PACEMAKER]))
        pm = pump_metrics(pooled, r["t_ms"])
        runs[proto] = dict(run=r, pooled=pooled, metrics=pm, params=params)

    # ---- 行为参考 ----
    print("[3/4] 行为参考（P5 逃避 / P6 自发）...")
    esc = escape_reference()
    spont = spontaneous_reference()

    # ---- 组装 npz ----
    out = {}
    # P3 咽部
    for proto in ("no_food", "food"):
        sa = stage_a[proto]
        for nm in names:
            out[f"pharynx_spike_times_{proto}_{nm}"] = sa["spike_times"][nm]
        pooled_a = np.sort(np.concatenate([sa["spike_times"][nm] for nm in names]))
        out[f"pharynx_spike_times_{proto}"] = pooled_a
        for nm, vv in sa["v_traces"].items():
            # Stage-A V 存档降到 dt=1ms（发放检测用全分辨率，内部已用）
            t_ds = np.arange(0.0, STAGE_A_T_MS + 1.0, 1.0)
            out[f"pharynx_v_{nm}_{proto}"] = np.interp(
                t_ds, sa["t_ms"], vv)
        r = runs[proto]
        for nm in names:
            out[f"pharynx_gap_v_{nm}_{proto}"] = r["run"]["v_ds"][names.index(nm)]
        m = r["metrics"]
        out[f"pharynx_psd_freq_{proto}"] = m["freq"]
        out[f"pharynx_psd_{proto}"] = m["psd"]
        out[f"pharynx_pump_signal_{proto}"] = m["pump_signal"]
        out[f"pharynx_peak_freq_{proto}"] = np.asarray([m["peak_freq"]])
        out[f"pharynx_peak_freq_argmax_{proto}"] = np.asarray([m["peak_freq_argmax"]])
        out[f"pharynx_peak_freq_welch_{proto}"] = np.asarray([m["peak_freq_welch"]])
        out[f"pharynx_peak_freq_acf_{proto}"] = np.asarray([m["peak_freq_acf"]])
        out[f"pharynx_burst_rate_{proto}"] = np.asarray([m["burst_rate"]])
        out[f"pharynx_pacemaker_spike_rate_{proto}"] = np.asarray([m["spike_rate"]])
        out[f"pharynx_drift_{proto}"] = np.asarray([m["drift"]])
        out[f"pharynx_pacemaker_spikes_{proto}"] = r["pooled"]
    out["pharynx_t_ms"] = stage_a["no_food"]["t_ms"]
    out["pharynx_gap_t_ms"] = runs["no_food"]["run"]["t_ds_ms"]

    # P5 逃避
    for k in ("nerve_latency_ms", "nerve_latency_full_ms", "behavior_latency_ms",
              "behavior_latency_v0_ms", "tau_trans_ms"):
        out[f"escape_ref_{k}"] = esc[k]
    out["escape_ref_direction"] = esc["direction"]
    out["escape_ref_reaction_probability"] = np.asarray([esc["reaction_probability"]])
    out["escape_ref_d_peak"] = np.asarray([esc["d_peak"]])
    out["escape_ref_c_back_peak"] = np.asarray([esc["c_back_peak"]])
    out["escape_ref_nerve_in_band"] = np.asarray([esc["nerve_in_band"]])
    out["escape_ref_behavior_in_band"] = np.asarray([esc["behavior_in_band"]])
    out["escape_ref_behavior_mean_ms"] = np.asarray([esc["behavior_mean_ms"]])
    out["escape_ref_behavior_std_ms"] = np.asarray([esc["behavior_std_ms"]])
    for k in ("t_ms", "c_back", "c_fwd", "v", "x", "tau_trans_ms", "t_cmd_ms"):
        out[f"escape_ref_traj_{k}"] = esc["traj"][k]

    # P6 自发
    out["spontaneous_ref_transition_matrix"] = spont["transition_matrix"]
    out["spontaneous_ref_stationary_embedded"] = spont["stationary_embedded"]
    out["spontaneous_ref_states"] = np.asarray(spont["states"])
    for k in ("time_fraction_fwd_pct", "time_fraction_fwd_sem",
              "time_fraction_rev_pct", "time_fraction_rev_sem",
              "time_fraction_turn_pct", "time_fraction_turn_sem",
              "time_fraction_pause_pct"):
        out[f"spontaneous_ref_{k}"] = np.asarray([spont[k]])
    for k in ("fwd_in_band", "rev_in_band", "turn_in_band"):
        out[f"spontaneous_ref_{k}"] = np.asarray([spont[k]])
    out["spontaneous_ref_bout_mean_ms"] = np.asarray(
        [spont["bout_mean_ms"][s] for s in _STATES])
    out["spontaneous_ref_band_fwd"] = np.asarray(spont["band_fwd"])
    out["spontaneous_ref_band_rev"] = np.asarray(spont["band_rev"])
    out["spontaneous_ref_band_turn"] = np.asarray(spont["band_turn"])
    for k in ("t_ms", "v", "omega", "c_fwd", "c_back"):
        out[f"spontaneous_ref_trace_{k}"] = spont["trace"][k]
    out["spontaneous_ref_trace_state"] = np.asarray(spont["trace"]["state"])

    meta = dict(
        engine=("NEURON 9.0.1 cvode（atol=rtol=1e-8, celsius=6.3, v_init=%g mV）"
                " + scipy solve_ivp LSODA（缝隙网络 rtol=1e-9/atol=1e-11，20 神经元高精度档）"
                " + 行为参考模型（纯 numpy）" % V0),
        params_csv=dict(pharynx="data/m5_pharynx_subgraph.csv",
                        behavior="data/m5_behavior_reference.csv",
                        worm="data/m5_worm_params.csv",
                        m3_anchor="data/m3_reflex_ref.npz"),
        dt_out_ms=DT_OUT,
        stage_a=dict(t_ms=STAGE_A_T_MS, food_drive_uA_cm2=FOOD_DRIVE_UA_CM2,
                     drive_set=FOOD_DRIVE_SET if FOOD_DRIVE_SET is not None else "all-20",
                     n_neurons=len(names), n_chem_active=len(sub["chem"]),
                     n_gap=len(sub["gaps"]),
                     note=("化学子图仅含 g>0 的 ach/glut 突触（87 条；other/serotonin g=0 调质占位"
                           "跳过，M6 补齐）；M5 无化学输入、仅缝隙耦合 → NEURON 化学子图下为孤立"
                           "驱动神经元，高频发放（~124Hz）——实测记录，缝隙在 Stage-B 恢复")),
        stage_b=dict(t_ms=STAGE_B_T_MS, gap_g_ns=GAP_G_NS,
                     ampa_tau_ms=AMPA_TAU_MS, ampa_e_mv=AMPA_E_MV,
                     pacemaker=list(PACEMAKER),
                     sahp=dict(theta_mv=SAHP_THETA_MV, k_mv=SAHP_K_MV,
                               mechanism=("I_sahp=−g_sahp·w·(V−EK)，"
                                          "dw/dt=(w_inf(V)−w)/τ；慢 K 适应突发放电 = 泵节律"
                                          "（Avery & Horvitz 1989：MC 定泵速；功能参考，参数校准落带）")),
                     pump_signal_definition=("MCL/MCR/M4 发放池高斯平滑包络 σ=100ms → 周期图主峰"
                                             "（PSD 带 [0.05,8]Hz）+ Welch + 自相关佐证 + 事件率"),
                     cal_nofood=cal["nofood"],
                     cal_food=cal["food"],
                     scan_nofood=[(r["I"], r["g_sahp"], r["tau"], r["burst_rate"])
                                  for r in cal["scan_nofood"]],
                     scan_food=[(r["I"], r["g_sahp"], r["tau"], r["w_chem"], r["burst_rate"])
                                for r in cal["scan_food"]]),
        bands=dict(
            pharynx_no_food=[bands[("pharynx", "rhythm_no_food_hz")]["lo"],
                             bands[("pharynx", "rhythm_no_food_hz")]["hi"]],
            pharynx_food=[bands[("pharynx", "rhythm_with_food_hz")]["lo"],
                          bands[("pharynx", "rhythm_with_food_hz")]["hi"]],
            escape_nerve=[bands[("escape", "nerve_latency_ms")]["lo"],
                          bands[("escape", "nerve_latency_ms")]["hi"]],
            escape_behavior=[bands[("escape", "behavior_latency_ms")]["lo"],
                             bands[("escape", "behavior_latency_ms")]["hi"]],
            escape_behavior_tol=[bands[("escape", "behavior_latency_ms")]["tol_lo"],
                                 bands[("escape", "behavior_latency_ms")]["tol_hi"]],
            spontaneous_fwd=[bands[("spontaneous", "time_fraction_fwd")]["lo"],
                             bands[("spontaneous", "time_fraction_fwd")]["hi"]],
            spontaneous_rev=[bands[("spontaneous", "time_fraction_rev")]["lo"],
                             bands[("spontaneous", "time_fraction_rev")]["hi"]],
            spontaneous_turn=[bands[("spontaneous", "time_fraction_turn")]["lo"],
                              bands[("spontaneous", "time_fraction_turn")]["hi"]]),
        escape=dict(
            n_trials=esc["n_trials"], tau_trans_ms=TAU_TRANS_MS,
            tau_trans_jit_ms=TAU_TRANS_JIT_MS, muscle_tau_ms=MUSCLE_TAU_MS,
            muscle_w_back=MUSCLE_W_BACK, c_fwd_baseline=C_FWD_BASELINE,
            nerve_latency_anchor="M3 m3_reflex_ref.npz latency_nerve 实测抽样（∈[5,20]）",
            nerve_window=[5.0, 20.0],
            behavior_window=[30.0, 50.0], behavior_tol=[25.0, 60.0],
            behavior_definition=("C_back ≥ 0.3·C_back_peak − touch（M3 同款）；"
                                 "behavior_v0 = 首个 v<0；行为 = τ_trans + 神经链 + 肌肉上升"),
            note=("M3 结构性落不到 [25,60]（无转导模型，m3_env_notes L7）；M5 补齐 τ_trans"
                  "（触→感觉电流）+ 显式肌肉上升 → 行为潜伏期 ~40ms 落 [30,50]。"
                  "τ_trans 建议由 P5 协议节点定稿写入 data/m5_worm_params.csv（本参考锚 23ms）。")),
        spontaneous=dict(
            n_trials=spont["n_trials"], t_total_ms=spont["t_total_ms"],
            dt_b_ms=spont["dt_b_ms"],
            bout_mean_ms=BOUT_MEAN_MS,
            cal_knobs=spont["cal_knobs"], out_of_band=spont["out_of_band"],
            classify_thresholds=spont["classify_thresholds"],
            note=("bout 半马尔可夫：嵌入链转移矩阵 + 指数 bout 时长（Srivastava 2013 量级）；"
                  "时间比例 = classify_state(v,ω,C_fwd,C_back) 测量（共用约定：B1d 的 "
                  "src/virtual_body.py 同签名同语义，P6 判定统一从单一实现调用）")),
        chemotaxis_reference="复用 data/m4_ref.npz（pirouette，本文件不重复）",
        note=("确定性：NEURON/cvode 与 numpy 均无随机性（seed=0 仅用于行为参考的试次/矩阵抽样）；"
              "实测坑与 P3/P5/P6 判据对照见执行节点回报 L23+（docs/m5_env_notes.md 待追加）"),
    )
    np.savez(out_npz, **out, meta=np.array(meta, dtype=object))
    return out_npz


if __name__ == "__main__":
    out = run_reference()
    d = np.load(out, allow_pickle=True)
    meta = d["meta"].item()
    print(f"参考解已写入: {out}")
    print(f"键数: {len([k for k in d.files if k != 'meta'])}")
    print("--- P3 咽部节律（Stage-B 缝隙网络；泵信号 = MCL/MCR/M4 发放池）---")
    for proto in ("no_food", "food"):
        f = float(d[f"pharynx_peak_freq_{proto}"][0])
        fw = float(d[f"pharynx_peak_freq_welch_{proto}"][0])
        fa = float(d[f"pharynx_peak_freq_acf_{proto}"][0])
        ev = float(d[f"pharynx_burst_rate_{proto}"][0])
        drift = float(d[f"pharynx_drift_{proto}"][0])
        band = meta["bands"]["pharynx_no_food" if proto == "no_food" else "pharynx_food"]
        n_sp = len(d[f"pharynx_pacemaker_spikes_{proto}"])
        print(f"  {proto:8s}: 主频={f:.3f}Hz (Welch={fw:.3f}, ACF={fa:.3f}, "
              f"event={ev:.2f}/s) 带{band} 落带={band[0]<=f<=band[1]} 漂移={drift:.3f} "
              f"起搏池发放={n_sp}")
    print("--- P5 逃避参考 ---")
    print(f"  神经潜伏期: {d['escape_ref_nerve_latency_ms'].mean():.2f}±"
          f"{d['escape_ref_nerve_latency_ms'].std():.2f}ms 入窗[5,20]="
          f"{d['escape_ref_nerve_in_band'][0]:.2f}")
    print(f"  行为潜伏期: {d['escape_ref_behavior_latency_ms'].mean():.2f}±"
          f"{d['escape_ref_behavior_latency_ms'].std():.2f}ms 入容差[25,60]="
          f"{d['escape_ref_behavior_in_band'][0]:.2f} 方向={set(d['escape_ref_direction'])} "
          f"概率={d['escape_ref_reaction_probability'][0]:.2f} D_peak={d['escape_ref_d_peak'][0]:.2f}")
    print("--- P6 自发行为参考 ---")
    print(f"  前进={d['spontaneous_ref_time_fraction_fwd_pct'][0]:.1f}% "
          f"后退={d['spontaneous_ref_time_fraction_rev_pct'][0]:.1f}% "
          f"转弯={d['spontaneous_ref_time_fraction_turn_pct'][0]:.1f}% "
          f"暂停={d['spontaneous_ref_time_fraction_pause_pct'][0]:.1f}%  "
          f"带[60-80]/[10-25]/[5-20] → "
          f"{d['spontaneous_ref_fwd_in_band'][0]}/{d['spontaneous_ref_rev_in_band'][0]}/"
          f"{d['spontaneous_ref_turn_in_band'][0]}")
    print("  bout 均值(ms):", {s: round(v, 1) for s, v in zip(
        d["spontaneous_ref_states"], d["spontaneous_ref_bout_mean_ms"])})
    print("  转移矩阵:\n", np.round(d["spontaneous_ref_transition_matrix"], 3))
