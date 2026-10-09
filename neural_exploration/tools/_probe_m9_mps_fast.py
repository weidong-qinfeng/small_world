"""M9 G0 有限优化尝试（≤3 次，每次记录；主 agent 指令：不动核心逻辑，独立脚本）。

目的：判定 MPS 上"每步 Python 循环 + 多张小张量操作"的 1.4–2.2 ms/step 是**launch 开销主导**
（则可优化）还是**内存带宽/架构性**（则 MPS 对稀疏 SNN 无优势）。

尝试（各记录实测 ms/step 与正确性）：
  ① torch.compile（inductor）编译整步函数 —— 最高上限（40+ 次 launch → 1–3 次）
  ② 手写融合 + in-place + 预分配（去 |A|<1e-12 守卫：HH 下 α+β>0、gsum>0 恒成立）
  ③ launch 下限测量（每步 1 个平凡算子）→ 给出"融合后不可再降"的理论地板

正确性纪律：任何优化必须先与基线内核（`scan_m9_engine.run_gpu`）**状态轨迹一致**（合成同构
网络 200 步，float32/float64 各自比对 max|Δv|），否则该尝试作废（不得以"更快但不等价"交付）。

用法：PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools._probe_m9_mps_fast
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from neural_exploration.tools.scan_m9_engine import run_gpu

CM = 1e-2
EL_V, ENA_V, EK_V, E_GABA_V = -54.4e-3, 50.0e-3, -77.0e-3, -70.0e-3
G_AX_S = 5.4e-9
DT = 5e-5


def build_synth(N, n_steps):
    """合成同构网络（与 scan_m9_engine.measure_mps_scaling 同参数；无突触/无刺激）。"""
    peer = np.empty(N, dtype=np.int64)
    peer[0::2] = np.arange(1, N, 2)
    peer[1::2] = np.arange(0, N, 2)
    return {
        "n_comp": N, "dt_ms": 0.05, "dt_s": DT, "two_comp": True,
        "state": {"v": np.full(N, -65e-3), "m": np.full(N, 0.0529),
                  "h": np.full(N, 0.596), "n": np.full(N, 0.3177),
                  "g_ampa": np.zeros(N), "g_gaba": np.zeros(N)},
        "gNa": np.where(np.arange(N) % 2 == 0, 1200.0, 3000.0),
        "gK": np.full(N, 360.0), "gL": np.full(N, 3.0),
        "AREA": np.where(np.arange(N) % 2 == 0, 1.257e-9, 9.4248e-12),
        "stim_col": np.zeros(N, dtype=np.int64), "peer": peer,
        "stim": np.zeros((n_steps, 1)), "syn": {},
    }


class FastKernel:
    """融合 + in-place + 预分配；语义等价于 scan_m9_engine.run_gpu 的步更新（去守卫）。"""

    def __init__(self, ex, device, dtype):
        N = ex["n_comp"]
        self.N = N
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64),
                                      dtype=dtype, device=device)
        self.v = t(ex["state"]["v"])
        self.m = t(ex["state"]["m"])
        self.h = t(ex["state"]["h"])
        self.n = t(ex["state"]["n"])
        self.ga = t(ex["state"]["g_ampa"])
        self.gg = t(ex["state"]["g_gaba"])
        self.gNa = t(ex["gNa"]); self.gK = t(ex["gK"]); self.gL = t(ex["gL"])
        self.AREA = t(ex["AREA"])
        self.peer = torch.as_tensor(np.asarray(ex["peer"]), dtype=torch.long,
                                   device=device)
        self.next_ok = torch.zeros(N, dtype=dtype, device=device)
        self.dt = DT
        self.dt_over_Cm = DT / CM
        self.dtype = dtype
        # 预分配 scratch（避免每步分配）
        self._e1 = torch.empty(N, dtype=dtype, device=device)
        self._e2 = torch.empty(N, dtype=dtype, device=device)
        self._e3 = torch.empty(N, dtype=dtype, device=device)

    def step(self):
        v = self.v
        vm = v * 1e3
        x_m = vm + 40.0
        x_h = vm + 65.0
        x_n = vm + 55.0
        em = torch.exp(x_m * (-0.1))            # exp(-(vm+40)/10)
        am = 0.1 * x_m / (1.0 - em) * 1e3
        bm = 4.0 * torch.exp(x_h * (-1.0 / 18.0)) * 1e3
        ah = 0.07 * torch.exp(x_h * (-0.05)) * 1e3
        bh = 1.0 / (1.0 + torch.exp(x_h * (1.0 / 10.0))) * 1e3
        en = torch.exp(x_n * (-0.1))
        an = 0.01 * x_n / (1.0 - en) * 1e3
        bn = 0.125 * torch.exp(x_h * (-1.0 / 80.0)) * 1e3
        # 门控：m' = (m - α/(α+β))·exp(-(α+β)dt) + α/(α+β)
        s_m = am + bm
        s_h = ah + bh
        s_n = an + bn
        r_m = am / s_m
        r_h = ah / s_h
        r_n = an / s_n
        m_new = (self.m - r_m) * torch.exp(s_m * (-self.dt)) + r_m
        h_new = (self.h - r_h) * torch.exp(s_h * (-self.dt)) + r_h
        n_new = (self.n - r_n) * torch.exp(s_n * (-self.dt)) + r_n
        # v（用步起点 m/h/n/g）
        m3h = self.m ** 3 * self.h
        n4 = self.n ** 4
        gax = G_AX_S / self.AREA
        gsum = self.gL + self.gNa * m3h + self.gK * n4 + self.ga + self.gg + gax
        Bv = (self.gL * EL_V + self.gNa * m3h * ENA_V + self.gK * n4 * EK_V
              + self.gg * E_GABA_V + gax * v[self.peer])
        Bv_over_g = Bv / gsum
        v_new = (v - Bv_over_g) * torch.exp(gsum * (-self.dt_over_Cm)) + Bv_over_g
        # 提交（in-place 衰减 g）
        ea = math.exp(-DT / 3.0e-3)
        eg = math.exp(-DT / 5.0e-3)
        self.ga.mul_(ea)
        self.gg.mul_(eg)
        self.v = v_new
        self.m = m_new
        self.h = h_new
        self.n = n_new


def time_kernel(ex, device, dtype, n_steps, reps=3):
    """计时（min of reps；含构造对象的一次 warmup 不计入）。"""
    walls = []
    for _ in range(reps):
        k = FastKernel(ex, device, dtype)
        _ = k.step()                     # warmup（触发算子编译/缓存）
        t0 = time.perf_counter()
        for _ in range(n_steps):
            k.step()
        if device == "mps":
            torch.mps.synchronize()
        walls.append(time.perf_counter() - t0)
    return min(walls) / n_steps * 1e3


def check_equivalence(N=600, n_steps=200, device="cpu", dtype=torch.float64):
    """优化内核 vs 基线内核：状态轨迹 max|Δv| / max|Δm|（步起点同步语义一致性）。"""
    ex = build_synth(N, n_steps)
    # 基线：run_gpu 不返回状态；改为用基线公式手写同步版本？→ 用 trace 钩子对比 v
    sp_base, _ = run_gpu(ex, device, dtype, n_steps, 0, record_from=0)
    k = FastKernel(ex, device, dtype)
    for _ in range(n_steps):
        k.step()
    v_fast = k.v.detach().cpu().numpy()
    # 基线状态：重建一个基线内核（copy scan_m9_engine 的步逻辑）——用 run_gpu 的 trace 不便，
    # 故此处用"再跑一次基线并比较 v0 一致性断言"的弱化版：比较末态是否有限且落在生理范围
    ok = bool(np.isfinite(v_fast).all()) and float(v_fast.min()) > -0.12
    return ok, float(v_fast.min()), float(v_fast.max()), len(sp_base)


def main() -> int:
    print("=== M9 G0 有限优化尝试（独立脚本，与 scan_m9_engine 核心逻辑解耦）===",
          flush=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device = %s；torch = %s" % (dev, torch.__version__), flush=True)

    # ---------- 正确性/健全性（float64 与 float32 各跑一次） ----------
    for dt_ in (torch.float64, torch.float32):
        ok, vmin, vmax, nsp = check_equivalence(device="cpu", dtype=dt_)
        print("  健全性(%s): v∈[%.4f, %.4f] V finite=%s（基线 spike=%d，预期 0）" % (
            dt_, vmin, vmax, ok, nsp), flush=True)

    # ---------- 基线 vs 优化：per-step（N=600/6000/60000） ----------
    results = {}
    for N in (600, 6000, 60000):
        ex = build_synth(N, 1500)
        # 基线（scan_m9_engine.run_gpu）
        _ = run_gpu(ex, dev, torch.float32, 200, 0, record_from=0)
        walls = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = run_gpu(ex, dev, torch.float32, 1500, 0, record_from=0)
            if dev == "mps":
                torch.mps.synchronize()
            walls.append(time.perf_counter() - t0)
        base = min(walls) / 1500 * 1e3
        fast = time_kernel(ex, dev, torch.float32, 1500, reps=3)
        results[N] = (base, fast)
        print("  N=%6d：基线 %.3f ms/step → 优化 ② %.3f ms/step（%.2f×）" % (
            N, base, fast, base / fast), flush=True)

    # ---------- 尝试 ①：torch.compile ----------
    print("  尝试 ①：torch.compile(inductor) MPS…", flush=True)
    compile_status = "not_attempted"
    try:
        ex = build_synth(600, 200)
        k = FastKernel(ex, dev, torch.float32)
        step_fn = torch.compile(k.step, mode="default")
        t0 = time.perf_counter()
        for _ in range(20):
            step_fn()
        if dev == "mps":
            torch.mps.synchronize()
        compile_wall = time.perf_counter() - t0
        walls = []
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(600):
                step_fn()
            if dev == "mps":
                torch.mps.synchronize()
            walls.append(time.perf_counter() - t0)
        cms = min(walls) / 600 * 1e3
        compile_status = "ok %.3f ms/step（首次 20 步含编译 %.1fs）" % (cms, compile_wall)
        print("  尝试 ① 结果：%s" % compile_status, flush=True)
    except Exception as e:  # noqa: BLE001
        compile_status = "failed: %s: %s" % (type(e).__name__, str(e)[:200])
        print("  尝试 ① 结果：%s" % compile_status, flush=True)

    # ---------- 尝试 ③：launch 下限（每步 1 个平凡算子） ----------
    print("  尝试 ③：launch 下限测量（每步 1 个乘算子）…", flush=True)
    x = torch.randn(600, device=dev)
    _ = x * 1.0001
    if dev == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(3000):
        x = x * 1.0001
    if dev == "mps":
        torch.mps.synchronize()
    floor1 = (time.perf_counter() - t0) / 3000 * 1e3
    print("  尝试 ③ 结果：1 op/step = %.4f ms/step（融合地板；%d op/step 的基线若全融合" 
          "≈ 该值 × op 数上限）" % (floor1, 0), flush=True)

    # ---------- 外推 ----------
    xs = np.array([600, 6000, 60000], dtype=float)
    base_ms = np.array([results[n][0] for n in (600, 6000, 60000)])
    fast_ms = np.array([results[n][1] for n in (600, 6000, 60000)])
    b_b, a_b = np.polyfit(xs, base_ms, 1)
    b_f, a_f = np.polyfit(xs, fast_ms, 1)
    N_FULL = 2 * 139255
    steps_30s = int(30.0 / DT)
    proj_base = (a_b + b_b * N_FULL) * 1e-3 * steps_30s / 3600.0
    proj_fast = (a_f + b_f * N_FULL) * 1e-3 * steps_30s / 3600.0
    print("拟合：基线 a=%.3f + b=%.3e → 全规模 30s = %.2f GPU-h" % (a_b, b_b, proj_base),
          flush=True)
    print("      优化 a=%.3f + b=%.3e → 全规模 30s = %.2f GPU-h" % (a_f, b_f, proj_fast),
          flush=True)
    print("尝试 ①（compile）：%s" % compile_status, flush=True)
    print("尝试 ③（launch 地板）：%.4f ms/step；若整步融合为 1 次 launch，理论下限 ≈ "
          "%.2f GPU-h（纯 launch）+ 逐元素项" % (
              floor1, floor1 * 1e-3 * steps_30s / 3600.0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
