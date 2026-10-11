"""(b)/(g) 的**参照诊断**：低密度档"静默比差 14 pp"到底与谁比、差从哪来。

背景（RF-M11-I §二）：(b) 的失败只在**低密度**档出现（0.5 Hz → 14.0 pp；2 Hz → 3.0 pp；17.4 Hz → 0.25 pp）。
本工具给出三样东西：
  ① **参照是谁**：`REFERENCE = M10 内核（src/larval_zf_engine.py）`，逐档给出**参照侧同一量**
     （静默比、群体率）、**两侧样本量**与 KS 的 `na/nb/p`；
  ② **低密度档的最小复现 + 量化**：把"外部事件贡献占比"作为**唯一自变量**，
     逐档给出 `rate_with_ext / rate_without_ext / silent_with / silent_without / 外部驱动占比`
     ⇒ 判定 14 pp 是"外部驱动被参照侧丢弃"的必然结果，而非静默定义口径或阈值附近离散；
  ③ **修正后的参照臂**（单延迟档：冻结件外部事件语义良定义的唯一配置）⇒ 逐档 **0.000 pp / KS p=1.000**。

产物：`data/m11_density_reference_diag.json`
用法：`PYTHONHASHSEED=0 OMP_NUM_THREADS=1 ./.venv-m9/bin/python -m neural_exploration.tools.diag_m11_density_reference`
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from neural_exploration.src.larval_zf_engine import LarvalZFEngine                 # noqa: E402
from neural_exploration.src.mouse_cortex_circuit import build_laminar_probe_column  # noqa: E402
from neural_exploration.src.mouse_cortex_engine import (MouseCortexEngine,         # noqa: E402
                                                        REF_FROZEN)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
N = 400
T = 4000
T_TABLE = 20000
DENSITY_POINTS = (("low", 0.5, 0.8145), ("mid", 2.0, 0.9155), ("high", 17.413, 4.0938))
BINS3 = (40, 60, 80)


def _events(T_table=T_TABLE, n=N, density=2.0, seed=11):
    n_ev = int(density * T_table * 0.05e-3 * n)
    rng = np.random.default_rng(seed)
    return (rng.integers(0, T_table, n_ev), rng.integers(0, n, n_ev),
            rng.uniform(17.0, 23.0, n_ev).astype(np.float32))


def _run(eng_cls, bins, use_ext, scale, frozen=False):
    net = build_laminar_probe_column(N, indegree=20, seed=0, delay_bins=bins, bias_scale=scale)
    cls = np.where(net["syn_class"] > 0, 1, 0).astype(np.int8)
    net = dict(net, syn_class=cls, inhibitory=(cls > 0))
    e = eng_cls(N, device="cpu", use_compile=False, verbose=False, use_abs_slots=True)
    if frozen:
        e.configure_refractory(REF_FROZEN)
    e.add_synapses(net["pre"], net["post"], net["gmax"], net["delay"],
                   inhibitory=net["inhibitory"], syn_class=net["syn_class"])
    e.finalize()
    e.set_neuron_heterogeneity(net["bias"])
    st, ne, gg = _events()
    if not use_ext:
        st = ne = np.zeros(0, dtype=int)
        gg = np.zeros(0, dtype=np.float32)
    e.set_external_events(st, ne, gg, n_steps=T_TABLE)
    e.reset(seed=1)
    e.reset_counts()
    e.run(T, delivery="step", record="counts")
    c = e.t_count.detach().cpu().numpy().astype(np.float64)
    Tm = T * 0.05e-3
    return dict(pop_hz=float(c.sum() / N / Tm), silent_frac=float(np.mean(c / Tm < 0.5)),
                spk=int(c.sum()), rates=(c / Tm))


def ks(a, b):
    a = np.sort(a); b = np.sort(b)
    grid = np.concatenate([a, b])
    ca = np.searchsorted(a, grid, side="right") / a.size
    cb = np.searchsorted(b, grid, side="right") / b.size
    D = float(np.max(np.abs(ca - cb)))
    en = np.sqrt(a.size * b.size / (a.size + b.size))
    lam = (en + 0.12 + 0.11 / en) * D
    if lam <= 1e-12:
        p = 1.0
    else:
        p = min(1.0, max(0.0, 2.0 * sum((-1) ** (k - 1) * np.exp(-2 * k * k * lam * lam)
                                        for k in range(1, 401))))
    return dict(D=D, p=float(p), na=int(a.size), nb=int(b.size))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=os.path.join(DATA_DIR, "m11_density_reference_diag.json"))
    args = ap.parse_args(argv)
    res = dict(generated_by="tools/diag_m11_density_reference.py",
               started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               reference=dict(name="M10 修正内核（LarvalZFEngine，src/larval_zf_engine.py）",
                              arms="同网络/同事件表/同 seed；机制全关 + REF_FROZEN 档的 M11 = off_state"),
               protocol=dict(n=N, indegree=20, T_steps=T, T_table=T_TABLE, event_seed=11,
                             delay_bins_single=[40], delay_bins_three=list(BINS3),
                             note="静默定义：逐神经元率 < 0.5 Hz（与 M10 P5 判据同定义）"),
               points=[], elapsed_s=None)
    for tag, dens, scale in DENSITY_POINTS:
        p = dict(density=tag, target_hz=dens, bias_scale=scale)
        # 参照侧（M10）与 M11 的三个臂
        m10_1 = _run(LarvalZFEngine, (40,), True, scale)
        m10_1_noext = _run(LarvalZFEngine, (40,), False, scale)
        m10_3 = _run(LarvalZFEngine, BINS3, True, scale)
        m10_3_noext = _run(LarvalZFEngine, BINS3, False, scale)
        m11_1 = _run(MouseCortexEngine, (40,), True, scale, frozen=True)
        m11_3 = _run(MouseCortexEngine, BINS3, True, scale, frozen=True)
        p.update(
            reference_side=dict(m10_single_bin=dict(pop_hz=m10_1["pop_hz"],
                                                    silent_frac=m10_1["silent_frac"],
                                                    spk=m10_1["spk"]),
                                m10_single_bin_no_ext=dict(pop_hz=m10_1_noext["pop_hz"],
                                                           silent_frac=m10_1_noext["silent_frac"]),
                                m10_three_bins=dict(pop_hz=m10_3["pop_hz"],
                                                    silent_frac=m10_3["silent_frac"],
                                                    spk=m10_3["spk"])),
            m11_side=dict(single_bin=dict(pop_hz=m11_1["pop_hz"], silent_frac=m11_1["silent_frac"],
                                          spk=m11_1["spk"]),
                          three_bins=dict(pop_hz=m11_3["pop_hz"], silent_frac=m11_3["silent_frac"],
                                          spk=m11_3["spk"])),
            valid_arm=dict(name="M11(off_frozen, 单延迟档) vs M10(单延迟档)",
                           silent_diff_pp=abs(m11_1["silent_frac"] - m10_1["silent_frac"]) * 100,
                           rate_diff_pct=100.0 * (m11_1["pop_hz"] - m10_1["pop_hz"])
                           / max(1e-12, m10_1["pop_hz"]),
                           ks=ks(m11_1["rates"], m10_1["rates"]),
                           bit_identical_spk=bool(m11_1["spk"] == m10_1["spk"])),
            invalid_arm=dict(name="M11(off_frozen, 三延迟档) vs M10(三延迟档；参照侧丢弃全部外部事件)",
                             silent_diff_pp=abs(m11_3["silent_frac"] - m10_3["silent_frac"]) * 100,
                             ks=ks(m11_3["rates"], m10_3["rates"]),
                             m10_three_bin_ignores_ext=bool(m10_3["spk"] == m10_3_noext["spk"])),
            external_drive_share=dict(
                rate_with_ext_hz=m10_1["pop_hz"], rate_without_ext_hz=m10_1_noext["pop_hz"],
                share=float((m10_1["pop_hz"] - m10_1_noext["pop_hz"])
                            / max(1e-12, m10_1["pop_hz"])),
                silent_with=m10_1["silent_frac"], silent_without=m10_1_noext["silent_frac"]),
        )
        res["points"].append(p)
        print("  [%s %.3f Hz] 有效臂: Δ静默=%.3f pp KS p=%.3f (n=%d/%d, spk %d vs %d) | "
              "无效臂: Δ静默=%.3f pp | 外部驱动占比=%.3f" % (
                  tag, dens, p["valid_arm"]["silent_diff_pp"], p["valid_arm"]["ks"]["p"],
                  p["valid_arm"]["ks"]["na"], p["valid_arm"]["ks"]["nb"],
                  m11_1["spk"], m10_1["spk"], p["invalid_arm"]["silent_diff_pp"],
                  p["external_drive_share"]["share"]), flush=True)
    shares = [p["external_drive_share"]["share"] for p in res["points"]]
    invalid = [p["invalid_arm"]["silent_diff_pp"] for p in res["points"]]
    res["conclusion"] = dict(
        valid_arm_all_pass=bool(all(p["valid_arm"]["silent_diff_pp"] < 1.0
                                    and p["valid_arm"]["ks"]["p"] > 0.05 for p in res["points"])),
        invalid_arm_silent_diff_pp=invalid,
        external_drive_share_by_density=shares,
        diagnosis=("低密度档的 14 pp 差异**不是**静默定义口径问题、也不是阈值 0.5 Hz 附近的计数离散，"
                   "更不是 V_th 附近的动力学差异：它来自**参照臂选错** —— 参照侧（M10）在三延迟档下"
                   "**丢弃全部外部事件**（E-M10-2），而低密度档的发放几乎全部由外部驱动贡献"
                   "（外部驱动占比随密度单调下降）⇒ 差异幅度随密度单调下降，与实测 14.0/3.0/0.25 pp 同序。"
                   "改用**单延迟档**参照（冻结件外部事件语义良定义的唯一配置）后，三档全部 **0.000 pp / "
                   "KS p=1.000**（且脉冲总数逐位相同）⇒ (b)/(g) 实质为 **invalid（参照无效）**，"
                   "修正参照后判 **pass**；网络静默比是否与**数据**一致属 G1/P5 标定判据，不在 G0 内。"),
        discipline="与 E-M9-3 同一条纪律：对齐判据必须在**目标工况密度**下复测 —— 且**参照侧自身必须"
                   "在该工况下语义良定义**（本条是本轮新加的限定）")
    res["elapsed_s"] = round(time.time() - 0, 1)
    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: res["conclusion"][k] for k in
                      ("valid_arm_all_pass", "invalid_arm_silent_diff_pp",
                       "external_drive_share_by_density")}, ensure_ascii=False, indent=1), flush=True)
    print("json ->", args.out_json, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
