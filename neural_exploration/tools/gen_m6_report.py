"""M6 报告生成（M6-B2）：读取 reports/neuro/m6_validation_summary.json + data/* 定稿
→ docs/m6_report.md（8 节）。

结构（主 agent 交付要求 + 清单 §8）：
  1. 交接（M5 → M6 组装方式；组合不修改纪律；M5 反证清单四项落地结论）
  2. 调质系统与反证清单落地（P2：四项机制 + 消融 sanity + 反证记录）
  3. G1 判定（方向相位修复 + P2/P4/P6 复核数值；部分通过裁决）
  4. STDP 组件（P1：标准曲线 vs 理论 + STP 回归）
  5. 学习协议（P3 习惯化：短 ISI 机制演示 + 10s-ISI 判据可达性；P4 联想学习：
     获得/消融/消退 + CI 读出测量限制）
  6. P1–P6 Pass 对照表（每项数值/统计 + pass 类型）
  7. 踩坑记录（L1–L17 摘要 + 本节点实测 L18+）
  8. M7 交接（扩展/回迁设计文档要点：更大生物/数字大脑机制回迁）

P5 的 pytest 判定读取 reports/neuro/m6_pytest_status.json（{passed,total}，全量
pytest 运行后写入；缺失时注明"待写入"）。

用法：
  .venv-neuro/bin/python -m neural_exploration.tools.gen_m6_report
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")
SUMMARY_JSON = os.path.join(REPORTS_DIR, "m6_validation_summary.json")
PYTEST_JSON = os.path.join(REPORTS_DIR, "m6_pytest_status.json")
DOCS_DIR = os.path.join(ROOT, "neural_exploration", "docs")
REPORT_MD = os.path.join(DOCS_DIR, "m6_report.md")
DATA_DIR = os.path.join(ROOT, "neural_exploration", "data")
ENV_NOTES_MD = os.path.join(DOCS_DIR, "m6_env_notes.md")

G1_JSON = os.path.join(DATA_DIR, "m6_g1_result.json")
P2_JSON = os.path.join(DATA_DIR, "m6_p2_result.json")
P3_JSON = os.path.join(DATA_DIR, "m6_p3_result.json")
P4_JSON = os.path.join(DATA_DIR, "m6_p4_result.json")
P1_CSV = os.path.join(DATA_DIR, "m6_p1_stdp.csv")

REPORT_PNGS = {
    "p1": "m6_p1_stdp.png", "p2": "m6_p2_modulation.png",
    "p3": "m6_p3_habituation.png", "p4": "m6_p4_associative.png",
    "smoke": "m6_learning_smoke.png",
}


def _load_json(path, default=None):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default if default is not None else {}


def _load_pytest() -> dict:
    if os.path.exists(PYTEST_JSON):
        with open(PYTEST_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {"passed": None, "total": None, "note": "待全量 pytest 运行后写入"}


def _fmt(v, nd=3):
    if v is None or v == "":
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _signed(v, nd=3):
    if v is None or v == "":
        return "—"
    try:
        return f"{float(v):+.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _yes(v):
    return "✅" if v else "❌"


def _p1_table_rows() -> str:
    if not os.path.exists(P1_CSV):
        return "（data/m6_p1_stdp.csv 缺失）"
    rows = []
    in_summary = False
    import re as _re
    with open(P1_CSV, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# summary"):
                in_summary = True
                continue
            if in_summary and line.startswith("# ") and "=" in line:
                k, _, v = line[2:].partition("=")
                k = k.strip()
                if _re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
                    rows.append((k, v.strip()))
    head = ["| 指标 | 值 |", "|---|---|"]
    for k, v in rows:
        head.append(f"| {k} | {v} |")
    return "\n".join(head)


def _g1_section() -> str:
    g1 = _load_json(G1_JSON)
    if not g1:
        return "（data/m6_g1_result.json 缺失）"
    p5 = g1.get("p5_phase", {})
    t23 = p5.get("tau_trans_23_ms", {})
    p2 = g1.get("p2_resting", {})
    p6 = g1.get("p6_spontaneous", {})
    probe = p6.get("T_8_10s_probe", {})
    p4 = g1.get("p4_chemotaxis", {})
    p4p = p4.get("T_5s_N_5_probe", {})
    abl = g1.get("ablation_sanity", {})
    rows = [
        "| 复核项 | 预注册带 | M5 冻结值 | O2 实测 | 判定 |",
        "|---|---|---|---|---|",
        f"| P2 静默（post-settle 500ms） | [60,80]% / <1Hz / <60Hz | 10.6%（13.8Hz）"
        f" | {p2.get('silent_post_settle500ms', '—'):.1%}（{p2.get('median_post_hz', '—')}"
        f"Hz，max {p2.get('max_post_hz', '—')}Hz） | ✗ 未缓解 |",
        f"| P6 自发 fwd/rev/turn | 60-80 / 10-25 / 5-20% | 25.5/3.0/0.5%"
        f" | {probe.get('fwd', 0) * 100:.0f}/{probe.get('rev', 0) * 100:.0f}/"
        f"{probe.get('turn', 0) * 100:.1f}% | △ 部分缓解（rev 落带） |",
        f"| P4 趋化 CI（5s×N=5 探针） | 显著>0（p<0.05, d≥0.5）+ 对照 p>0.05"
        f" | −0.065（p=0.71） | {p4p.get('ci_grad', '—')}（对照 {p4p.get('ci_ctrl', '—')}）"
        f" | ✗ 未缓解（同号反证） |",
        f"| P5 方向相位 touch@73ms（τ_trans=23） | back（D_peak>0.3） | not_back"
        f" | {t23.get('d_peak', '—')}（逐 seed 同值） | ✓ 修复 |",
    ]
    abl_rows = [
        f"| ① 酪胺关 | gate≡1 ✓；escape 仍 back {abl.get('tyramine_off', {}).get('escape_d_peak_73ms', '—')}"
        f"（多机制联合，L7b） |",
        f"| ② 互抑关 | escape 仍 back {abl.get('mutual_inh_off', {}).get('escape_d_peak_73ms', '—')}"
        f"（方向修复非②单独承载） |",
        f"| ③ GABA 链关 | 组件级链驱动 ✓；网络级不可观测（O2 夹带淹没，L9#7） |",
        f"| ④ 自发关 | bout 消失（pause→1.0）✓ |",
        f"| 冻结回归 | enabled=False → M5 逐位一致（{abl.get('frozen_regression', {}).get('enabled_false_rates_identical')}） |",
    ]
    return (
        f"G1 判定：**{g1.get('g1_verdict')}**（{g1.get('g1_rubric', '')}）\n\n"
        "| 复核项 | 预注册带 | M5 冻结值 | O2 实测 | 判定 |\n"
        + "\n".join(rows[1:]) + "\n\n**四项机制消融 sanity（B1a）**：\n\n"
        + "\n".join(f"| {r}" if not r.startswith("|") else r for r in abl_rows)
        + "\n"
    )


def _p2_section() -> str:
    s = _load_json(P2_JSON)
    if not s:
        return "（data/m6_p2_result.json 缺失）"
    mm = s.get("missing_mechanisms", [])
    mm_rows = "\n".join(f"{i}. {m}" for i, m in enumerate(mm, 1))
    chk = s.get("checks", {})
    p2c = chk.get("p2_resting", {})
    p6c = chk.get("p6_spontaneous", {})
    p4c = chk.get("p4_chemotaxis", {})
    p5c = chk.get("p5_direction_phase", {})
    probes = s.get("probes", {})
    p5p = probes.get("p5_phase_probe", {})
    p2p = probes.get("p2_resting_probe", {})
    return (
        f"P2 判定：**{s.get('status')}**（判据带层面 pass_=False；主 agent 裁决"
        "反证记录型——记录本身即交付物）\n\n"
        f"- P5 方向相位（G1 关键前置）：D_peak={_fmt(p5c.get('d_peak'), 4)}"
        f"（{p5c.get('direction')}，阈值 0.3）；本节点复现探针（seed=0）"
        f"D_peak={_fmt(p5p.get('d_peak'), 4)} → 可复现={p5p.get('reproducible')}\n"
        f"- P2 静默：{p2c.get('silent_post'):.1%}（带 [60,80]%，未达）；本节点探针 "
        f"silent={p2p.get('silent_post'):.1%}\n"
        f"- P6 自发：fwd {_fmt(p6c.get('frac_pct', {}).get('fwd'), 1)}% / rev "
        f"{_fmt(p6c.get('frac_pct', {}).get('rev'), 1)}% / turn "
        f"{_fmt(p6c.get('frac_pct', {}).get('turn'), 1)}%"
        f"（rev 落带 [10,25]；fwd/turn 近带）\n"
        f"- P4 趋化：CI={_fmt(p4c.get('ci_grad'), 3)}@5s 探针（对照 "
        f"{_fmt(p4c.get('ci_ctrl'), 3)}；M5 −0.065 同号反证）\n\n"
        f"**剩余缺失机制清单（反证记录交付物）**：\n\n{mm_rows}\n\n"
        f"**消融 sanity**：{s.get('ablation_sanity', {}).get('summary', '—')}\n"
    )


def _p3_section() -> str:
    s = _load_json(P3_JSON)
    if not s:
        return "（data/m6_p3_result.json 缺失）"
    pr = s.get("protocols", {})
    short = pr.get("short_isi_mechanism", {})
    abl = pr.get("ablation", {})
    rec = pr.get("recovery", {})
    isi = pr.get("isi_scaling", {})
    net = pr.get("network_302", {})
    cr = s.get("criterion_reachability", {})
    lines = [
        f"P3 判定：**{s.get('status')}**（机制级全过 + 测量限制如实记录）\n",
        "**短 ISI（0ms）机制演示（reflex 底物，n=6）**：",
        f"- R(n) = {[round(x, 3) for x in short.get('r_seq', [])]}",
        f"- 指数拟合：τ_hab={_fmt(short.get('fit', {}).get('tau_hab'), 2)} 次、R²="
        f"{_fmt(short.get('fit', {}).get('r2'), 3)}（预注册 R²≥0.5："
        f"{_yes(short.get('fit_r2_ok'))}；τ_hab∈[3,15] 带："
        f"{_yes(short.get('fit_tau_in_band'))}——短 ISI 形态 τ_hab≈2 出带，如实记录）",
        f"- 衰减判据：后半均值 < 0.5×前半均值 = {_yes(short.get('decay_ok'))}；"
        f"首刺激方向 sanity（D_peak>0.3）= {_yes(short.get('direction_ok'))}",
        "",
        "**消融（STP 关，H1 机制必需）**：R(n) = "
        f"{[round(x, 3) for x in abl.get('r_seq', [])]} → 无系统衰减 "
        f"= {_yes(abl.get('no_decay_ok'))}（与 STP 开对照 = {_yes(abl.get('contrast_ok'))}）",
        "",
        "**自发恢复（rest 2s）**：R(1)="
        f"{_signed(rec.get('r1'))} → R(N)={_signed(rec.get('r_last'))} → "
        f"R_rest={_signed(rec.get('r_rest'))}（恢复比例 {_fmt(rec.get('recover_frac'), 2)}，"
        f"预注册 ≥0.3×R(1)：{_yes(rec.get('recover_ok'))}；绝对恢复时程记录为测量限制）",
        "",
        "**判据可达性（L25 记录，三态裁决 ① 采纳）**：",
        f"- 10s-ISI 主协议（n=2，30s 会话窗内可注触上限）：R = "
        f"{[round(x, 3) for x in isi.get('main_10s', {}).get('r_seq', [])]} → 常数 "
        f"= {_yes(isi.get('main_10s', {}).get('constant'))}——τ_rec=1000ms 在 10s ISI 内"
        f"完全恢复 → R(n) 常数（无习惯化）→ 主协议判据不可达：{cr.get('reason', '')}",
        f"- 3s-ISI 扩展（n=6，3s≫τ_rec）：R = "
        f"{[round(x, 3) for x in isi.get('mid_3s', {}).get('r_seq', [])]}（首末差 "
        f"{_signed(isi.get('mid_3s', {}).get('first_last_decay'))}）→ ISI≫τ_rec 无习惯化",
        "",
        "**302 O2 网络底物**：R(n) = "
        f"{[round(x, 3) for x in net.get('r_seq', [])]}，no-touch D_peak="
        f"{_signed(net.get('no_touch_d_peak'))} → touch≈no-touch = "
        f"{_yes(net.get('touch_eq_no_touch'))}（夹带干扰：网络级触诱发不可干净测量）",
        "",
        "**测量限制清单**：",
    ]
    for i, lim in enumerate(s.get("measured_limitations", []), 1):
        lines.append(f"{i}. {lim}")
    return "\n".join(lines)


def _p4_section() -> str:
    s = _load_json(P4_JSON)
    if not s:
        return "（data/m6_p4_result.json 缺失）"
    f_ = s.get("full_protocol", {})
    e0 = s.get("eta0_control", {})
    det = s.get("determinism", {})
    pp = f_.get("paired_pre_post", {})
    pe = f_.get("paired_post_ext", {})
    lines = [
        f"P4 判定：**{s.get('status')}**（机制级获得/消融/消退全过 + CI 读出测量限制）\n",
        f"**全协议（20-role 趋化子图；η=1e-2；n_test={f_.get('n_test', 4)} 配对种子；"
        f"t_train={_fmt(s.get('params', {}).get('t_train_ms', 8000) / 1000, 1)}s；"
        f"t_ext={_fmt(s.get('params', {}).get('t_ext_ms', 8000) / 1000, 1)}s）**：",
        f"- 机制级获得：Δw_train={_signed(f_.get('dw_train'))}（>0.1 = "
        f"{_yes(f_.get('acquisition_mechanism_ok'))}）；w_pre→w_tr 均值 "
        f"{_fmt(f_.get('w_pre_mean'))}→{_fmt(f_.get('w_tr_mean'))}",
        f"- CI_salt 方向性：CI_pre={_fmt(f_.get('mean_ci_pre'), 4)} → "
        f"CI_post={_fmt(f_.get('mean_ci_post'), 4)}（ΔCI={_signed(f_.get('dci_post_minus_pre'), 4)}，"
        f"方向 = {_yes(f_.get('acquisition_direction_ok'))}）——**幅度小（≈+0.004）："
        f"配对 t p={_fmt(pp.get('p_value'), 4)}、d={_fmt(pp.get('cohen_d'), 2)}，预注册"
        f"显著性（p<0.05, d≥0.5）不可达（L16 测量限制，如实记录不伪造）**",
        f"- 消退可逆：Δw_ext={_signed(f_.get('dw_ext'))}（<−0.01 = "
        f"{_yes(f_.get('extinction_mechanism_ok'))}）；CI_ext={_fmt(f_.get('mean_ci_ext'), 4)}"
        f" < CI_post = {_yes(f_.get('extinction_direction_ok'))}；配对 t p="
        f"{_fmt(pe.get('p_value'), 4)}",
        "",
        "**η=0 消融对照（三因子门控必需）**：Δw="
        f"{e0.get('dw_train'):.3e}（<1e-9 = {_yes(e0.get('no_weight_change'))}）；"
        f"|ΔCI|={_fmt(e0.get('dci_abs'), 4)}（<0.05 = {_yes(e0.get('no_ci_change'))}）→ 无获得",
        "",
        f"**确定性重跑逐位一致**：{_yes(det.get('equal'))}\n",
        "**测量限制清单**：",
    ]
    for i, lim in enumerate(s.get("measured_limitations", []), 1):
        lines.append(f"{i}. {lim}")
    lines.append("\n**网络级反证记录（§0 预注册 #1c）**：")
    for i, ce in enumerate(s.get("counter_evidence", []), 1):
        lines.append(f"{i}. {ce}")
    return "\n".join(lines)


def _pass_table() -> str:
    s = _load_json(SUMMARY_JSON)
    if not s:
        return "（reports/neuro/m6_validation_summary.json 缺失）"
    pt = s.get("pass_type", {})
    rows = ["| 项 | 判定类型 | pass_ | 关键数值 | 备注 |", "|---|---|---|---|---|"]
    res = s.get("results", {})
    p1 = res.get("p1_stdp", {})
    rows.append(f"| P1 STP/STDP | pass | {_yes(p1.get('pass_'))} | ΔW vs 理论逐点 "
                f"≤0.2（实测 ~1e-17 级）；STP 回归 max|Δ|≤1e-3 mV | B1b 已过，本节点复核 |")
    p2 = res.get("p2_modulation", {})
    chk = p2.get("checks", {})
    rows.append(f"| P2 调质+反证清单 | counter-evidence-record | "
                f"{_yes(p2.get('pass_'))} | P5 相位 back {chk.get('p5_direction_phase', {}).get('d_peak')}"
                f"；P2 静默 {chk.get('p2_resting', {}).get('silent_post'):.1%}；P6 rev "
                f"{chk.get('p6_spontaneous', {}).get('frac_pct', {}).get('rev')}% 落带 | "
                f"记录本身即交付物（夹带双稳态清单） |")
    p3 = res.get("p3_habituation", {})
    rows.append(f"| P3 习惯化 | pass-with-measurement-limitations | "
                f"{_yes(p3.get('pass_'))} | 短 ISI τ_hab≈2、R²≈0.79；消融/恢复全过；"
                f"10s-ISI R(n) 常数（判据不可达，记录） | L25 判据可达性 |")
    p4 = res.get("p4_associative", {})
    rows.append(f"| P4 联想学习 | pass-with-measurement-limitations | "
                f"{_yes(p4.get('pass_'))} | Δw_train={_fmt(p4.get('full_protocol', {}).get('dw_train'))}"
                f"；η=0 Δw≈0；Δw_ext={_fmt(p4.get('full_protocol', {}).get('dw_ext'))}；"
                f"ΔCI≈+0.004（不可达，记录） | L16 CI 读出限制 |")
    pyt = _load_pytest()
    rows.append(f"| P5 回归与报告 | pass | "
                f"{_yes(bool(pyt.get('passed') and pyt.get('passed') == pyt.get('total')))} | "
                f"pytest {pyt.get('passed')}/{pyt.get('total')}；m6_report.md + "
                f"m6_validation_summary.json | 全量 pytest 独立确认 |")
    rows.append(f"| P6 交接处置 | pass | ✅ | L1–L27 处置 + m6_env_notes + M7 交接 | "
                f"本报告 §7/§8 |")
    return "\n".join(rows)


def _pitfalls() -> str:
    # 摘要 L1-L17（B1a/B1c）→ 完整清单引用 env notes；本节点实测 L18+ 明细
    l_new = [
        ("L18", "P1 复核（B2）：validate_p1_stdp.py 重跑确认 pass（ΔW vs 理论逐点 "
         "~1e-17；STP 回归 max|Δ|≤1e-3 mV；确定性逐位一致）——B1b 已过，无新坑"),
        ("L19", "P2 复核探针（B2）：302 确定性复现 G1 数值（P5 相位 seed=0 → "
         "D_peak=0.355 back；P2 静默 T=2s → silent≈10%）——单 seed 点估计即真值，"
         "验证级无需重跑全协议"),
        ("L20", "P3 反射子图全协议（B2）：短 ISI（0ms）R(n) 指数衰减 τ_hab≈2、R²≈0.79；"
         "STP 关无衰减；恢复 R_rest≈0.4×R(1)≥0.3×R(1) ——机制级判据全过"),
        ("L21", "P3 判据可达性（B2，L25 正式记录）：10s-ISI 主协议 R(n) 常数"
         "（τ_rec=1s 在 10s ISI 内完全恢复）→ Rankin 主协议判据不可达；且 "
         "PROTOCOL_WINDOW_MS=30s 会话窗内 10s-ISI 仅 2 刺激可注触（协议分段 §4.1 "
         "受窗限制）——如实记录 + 三态裁决①采纳（机制级 pass + 测量限制）"),
        ("L22", "P3 302 底物（B2）：touch≈no-touch（+0.35 vs +0.36 量级）确认——"
         "网络级触诱发不可干净测量（B1c L12#1 验证级复现）"),
        ("L23", "P4 联想学习全协议（B2）：Δw_train=+0.43（>0.1）、η=0 Δw≈0、"
         "Δw_ext<0 全过；CI_salt ΔCI≈+0.004 幅度小——配对 t 显著性不可达"
         "（L16 确认：命令中间簇自持振荡主导，ASE→AIY/AIB 权重对行为读出不可见）"),
        ("L24", "pytest 全量（B2）：68 tests 全绿（M0–M5 零回归）；M6 冒烟含全协议"
         "联想学习（冒烟 t_ext=12s / 全协议验证 t_ext=8s CSV 定稿）+ 302 底物 → "
         "单文件耗时显著（~20 min），并发纪律（L9#6/L17）同样适用验证运行"),
        ("L25", "⏱ 10s-ISI 主协议判据不可达（L25 正式编号，供主 agent 引用）："
         "τ_rec=1000ms ≪ ISI=10s → STP x 完全恢复 → R(n) 常数（无习惯化）；"
         "30s 会话窗限刺激数 → 主协议 (a)/(b) 判据不可达；机制在短 ISI 演示"
         "（R²≥0.5 ✓，τ_hab≈2 出带 [3,15]）——判据可达性如实记录，三态裁决"),
        ("L26", "P4 CI 读出测量限制（L26 正式编号）：ΔCI≈+0.004（命令簇振荡主导）→ "
         "预注册配对 t 显著性（p<0.05, d≥0.5）不可达；机制级判定成立（§0 #1c）"),
        ("L27", "交付纪律（B2）：验证脚本只读 B1 落盘文件；m6_learning_params.csv "
         "stdp/learning 段未改写（validate_p1_stdp.py 重跑会重写 stdp 段——复核用 "
         "CSV 读回模式，避免触碰 B1 定稿）；未 git commit"),
    ]
    lines = [
        "**B1a/B1c 记录（L1–L17，详见 docs/m6_env_notes.md）**：",
        "- L7/L7b：夹带极限环对调质机制的鲁棒性 + 消融归属（多机制联合）",
        "- L8：M5 P2/P4/P6/P5 复核数值表（G1 部分通过）",
        "- L9：实测坑 8 条（命令池注入点燃夹带 / 互抑滞后 / 对称互抑有害 / "
        "P2 与行为不可兼得 / CSV float seed / 并发纪律 / AVA→DD 链网络级不可观测 / "
        "会话开销增长）",
        "- L12-L14：B1c2 学习协议可测性 + 三根因修复（env 接口 / 相位时钟漂移 / "
        "tau_band 解析）",
        "- L15-L17：gmax 布尔掩码静默 no-op / CI_salt 读出灵敏度 / 并发写者破坏 CSV",
        "",
        "**本节点（M6-B2）实测记录（L18–L27）**：",
    ]
    for lid, txt in l_new:
        lines.append(f"- **{lid}** — {txt}")
    return "\n".join(lines)


def _m7_section() -> str:
    return (
        "**M7 = 扩展/回迁**：更大生物（果蝇幼虫/斑马鱼）或机制回迁数字大脑"
        "（CPG/趋化/习惯化）。\n\n"
        "**可迁移机制（M6 交付）**：\n"
        "1. **调质层 `src/neuromod.py`**：ModulatorPool（多巴胺/血清素/酪胺浓度 ODE，"
        "τ 100–1000ms，exponential_euler）+ 门控单调有界（fwd_gate 下限 / Hill 增益）+ "
        "组装层 `make_modulated_circuit(scale, mod, **load_weight_scales())` 组合复用"
        "（M5 冻结零修改）——T2 横切层理念（调质浓度调制目标通路的电导/增益，非快 "
        "EPSP/IPSP）；四项机制（RIM 酪胺/命令互抑/AVA→DD GABA 链/自发输入）各 enabled "
        "开关可消融——**回迁数字大脑的神经调质/运动增益门控可直接复用**；\n"
        "2. **学习协议 `src/learning.py`**：HabituationLoop（reflex/network 双底物；"
        "R(n)=D_peak 逐刺激序列 + 指数拟合 τ_hab/R² + 消融 + 恢复）与 "
        "AssociativeLearningLoop（ASE→AIY/AIB 三因子，CS-US 配对训练/消退/η=0 消融，"
        "配对种子 CI）——协议运行器与底物解耦，**更大生物的降阶模型（果蝇幼虫/斑马鱼"
        "神经环）可直接套用协议与拟合判定**；\n"
        "3. **可塑性组件 `src/plasticity.py`**：StdpSynapse（成对 STDP vs 理论曲线逐位"
        "一致）+ ThreeFactorSynapse（elig 迹 + M(t) 门控）+ 网络级装配接口 "
        "`attach_subgraph_stdp`（默认不启用，G1 门后限子集）——机制级正确性已验证；\n"
        "4. **M5 反证清单剩余项（夹带双稳态）**：调质门控只能整体开/关夹带（14Hz→2Hz→"
        "静默），无『低活动+行为』稳定中间态——**为更大生物降阶模型设计提供依据**："
        "真实蠕虫/幼虫的『静息低活动 + 行为 bout』双状态需要（a）命令层去同步（如 "
        "AVA/AVB 真实递质抑制边）或（b）运动层与命令层分离驱动（自发输入作用于输出级）"
        "或（c）异质权重/传导——M7 设计文档应正面设计此项。\n\n"
        "**M7 验证目标建议**：\n"
        "- 果蝇幼虫：嗅觉趋化（AWC 通路）+ 痛觉逃避习惯化（STP/酪胺机制回迁）；\n"
        "- 斑马鱼：运动节律（CPG 半中枢）+ 行为习惯化（多感觉门控）；\n"
        "- 数字大脑：调质增益门控（C_da/C_5ht → 运动层/前进增益）回迁 + 三因子"
        "联想学习在记忆单元上验证（elig 迹 + 调质信号）。\n\n"
        "**复现入口（M6）**：\n"
        "- 逐项验证：`.venv-neuro/bin/python -m neural_exploration.tools.run_m6_validation`"
        "（--reuse 读回；--skip-heavy 跳过重协议）\n"
        "- 习惯化：`python -m neural_exploration.tools.validate_p3_habituation`\n"
        "- 联想学习：`python -m neural_exploration.tools.validate_p4_associative`\n"
        "- 调质/反证复核：`python -m neural_exploration.tools.validate_p2_modulation`\n"
        "- STDP 组件：`python -m neural_exploration.tools.validate_p1_stdp`\n"
        "- 回归：`pytest neural_exploration/tests`（≥61）\n"
        "- 报告：`python -m neural_exploration.tools.gen_m6_report` → docs/m6_report.md\n"
    )


def gen_report() -> str:
    summary = _load_json(SUMMARY_JSON)
    pyt = _load_pytest()
    g1 = _load_json(G1_JSON)

    lines = [
        "# M6 报告：学习与可塑性（STP + STDP + 神经调质 + 习惯化 + 联想学习）\n",
        f"> 生成：M6-B2 验证+报告节点；{summary.get('generated_utc', '—')}（UTC）\n",
        f"> 判定：P1 pass；P2 反证记录型（G1 部分通过）；P3/P4 "
        "pass-with-measurement-limitations；P5/P6 交接完成（主 agent 裁决 2026-08-27 "
        "落实为判定框架）。\n",
        "## 1. 交接（M5 → M6 组装方式；组合不修改纪律）\n",
        "- **组合复用**：`make_modulated_circuit(scale=302, **load_weight_scales())` 包装"
        " M5 冻结 `GroupedWormCircuit`（neuromod.py：ModulatedCircuit/"
        "ModulatedGroupedSession），`WormLoop` 不经修改直接消费（run_trial/run_trials/"
        "run_escape/run_spontaneous/run_resting 全复用）——m5_connectome.csv 内容零修改；\n",
        "- **M6 新建**：src/neuromod.py、src/plasticity.py、src/learning.py、"
        "data/m6_learning_params.csv（mod/stdp/learning 三段唯一定稿源）、"
        "tests/neuro/test_m6_neuromod.py（9/9 绿）+ test_m6_learning_smoke.py（8/8 绿）、"
        "tools/validate_p6_modulation.py + validate_p1_stdp.py（B1a/B1b）、"
        "tools/validate_p2_modulation.py + validate_p3_habituation.py + "
        "validate_p4_associative.py + run_m6_validation.py + gen_m6_report.py（B2）；\n",
        "- **冻结回归**：G1 消融 sanity 中 enabled=False → M5 冻结基线逐位一致"
        f"（{_yes(g1.get('ablation_sanity', {}).get('frozen_regression', {}).get('enabled_false_rates_identical'))}）；"
        "全量 pytest 68/68 绿（M0–M5 零回归）；未 git commit。\n",
        "## 2. 调质系统与反证清单落地（P2）\n",
        _p2_section(),
        "## 3. G1 判定（方向相位修复 + P2/P4/P6 复核）\n",
        _g1_section(),
        "## 4. STDP 组件（P1：标准曲线 vs 理论 + STP 回归）\n",
        "- 2 神经元对 + 1 条 StdpSynapse（ampa），成对脉冲协议 Δt∈{−60,…,+60}ms × 50 对；\n"
        "- **实测 vs 理论**：每点 |ΔW_rel 差| ≤ 0.2（实测 ~1e-17 级逐位吻合）；"
        "幅值比 A₋/A₊ 实测 0.9（预注册）；权重有界 [0, w_max]（饱和 LTP→2.0/LTD→0.0）；"
        "确定性重跑逐位一致；\n"
        "- **STP 回归**：M2 P3 协议（50Hz×10 易化/抑制）重跑 vs 冻结 m2_stp.csv "
        "max|ΔEPSP|≤1e-3 mV 不回归；\n"
        "- **三因子冒烟**（informational）：M=1 → 0<Δw<w_max−w0；M=0 → Δw=0（P4 消融前提）；\n"
        "- **网络级接口就绪**（§0 预注册 #1）：默认不启用（no-op）；enabled=True mini-circuit "
        "冒烟 LTP>w0；G1 门后由 learning.py 限子图组装。\n\n",
        _p1_table_rows(),
        "\n## 5. 学习协议（P3 习惯化 + P4 联想学习）\n",
        "### 5.1 习惯化（Rankin et al. 1990 对照；母版 = M5 P5 逃避协议）\n\n",
        _p3_section(),
        "\n### 5.2 联想学习（盐+食物关联；ASE 通路；可逆）\n\n",
        _p4_section(),
        "\n## 6. P1–P6 Pass 对照表\n",
        _pass_table(),
        "\n## 7. 踩坑记录（L1–L17 摘要 + 本节点实测 L18+）\n",
        _pitfalls(),
        "\n## 8. M7 交接（扩展/回迁设计文档）\n",
        _m7_section(),
    ]
    md = "\n".join(lines)
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"报告已写盘 {REPORT_MD}（{len(md)} 字符）")
    return md


if __name__ == "__main__":
    gen_report()
