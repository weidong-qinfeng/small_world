"""M10 判据带定稿器（行为/标定/机制阶梯）——**运行前落盘，不事后调**（M10 §0.3 #6 / §0.7 #8）。

产出（唯一权威判据带 CSV，供 P4–P11 读取）：
  data/m10_behavior_reference.csv   行为学带（P4/P8/P9/P10/P11/P13）
  data/m10_calibration_targets.csv  标定带（P5，G1 门）
  data/m10_mechanism_ladder.csv     机制阶梯参数范围（P6，前置机制门）

**诚实性声明（必读）**：
  本文件的行来源分三类，逐行在 `provenance_class` 列登记：
    - `pre_registered_draft`：来自 M10 实施清单 §1.3/§4.1/§4.2 的**预注册草案**（本次定稿写盘，
      协议运行前，未事后调整）。清单原文即要求"以文献整理定稿为准"。
    - `measured_derived`：来自本次数据门实测（如 ZAPBench 成像统计 → `m10_imaging_reference.csv`）。
    - `literature_anchor`：来自可核对文献量级（**本次会话网络受限，DOI 级 provenance 化未完成**
      —— science.org 403 / cell.com 超时 / biorxiv 429，登记为数据门测量限制，见
      `docs/m10_env_notes.md` L13），交由 P4/P9–P11 执行时按 PMC 可得源逐条补齐。
  任何判据带不得在协议运行后修改；确需修正必须**新增行 + 保留原行**（M9 L36.3 纪律）。

用法：
    PYTHONHASHSEED=0 ./.venv-m10/bin/python -m neural_exploration.tools.build_m10_reference_bands
"""
from __future__ import annotations

import argparse
import csv
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

BEHAVIOR = [
    # key, value_lo, value_hi, unit, dpf, protocol, provenance_class, source, note
    ("swim_freq_lo", "20", "40", "Hz", "5-7", "fictive/自由游泳自发 bout",
     "pre_registered_draft", "M10 §1.3", "游泳模态弯曲波频率（swim）"),
    ("struggle_freq_lo", "40", "60", "Hz", "5-7", "强持续刺激", "pre_registered_draft",
     "M10 §1.3", "挣扎（struggle）"),
    ("escape_freq_lo", "40", "80", "Hz", "5-7", "突发听/触觉刺激", "pre_registered_draft",
     "M10 §1.3", "逃逸（escape）"),
    ("amplitude_order", "escape>struggle>swim", "", "ordering", "5-7", "三模态并测",
     "pre_registered_draft", "M10 §1.3", "模态间有序差异（峰值角）"),
    ("escape_latency_lo", "4", "10", "ms", "5-7", "听觉/触觉惊吓", "literature_anchor",
     "M10 §1.3 + 领域通行量级（Mauthner C-start）",
     "刺激→反应潜伏期；DOI 级 provenance 化待补（数据门限制 L13）"),
    ("escape_response_prob_lo", "0.80", "1.0", "", "5-7", "同上", "pre_registered_draft",
     "M10 §1.3 / §3.5", "反应概率下限（sanity）"),
    ("okr_gain_lo", "0.30", "0.90", "", "5-7", "光流条纹闭环",
     "pre_registered_draft", "M10 §1.3", "OKR 增益（<1），分光流速度档"),
    ("okr_phase_lag_lo", "5", "90", "deg", "5-7", "同上", "pre_registered_draft",
     "M10 §1.3", "相位滞后带（粗带，执行时按文献分档细化）"),
    ("phototaxis_turn_rate_lo", "0.05", "2.0", "1/s", "5-7", "亮/暗选择",
     "pre_registered_draft", "M10 §1.3", "转弯速率带 + 方向正确性"),
    ("prey_success_lo", "0.20", "0.90", "", ">=5", "小目标（猎物）刺激",
     "pre_registered_draft", "M10 §1.3", "捕食成功率（逐 dpf 登记）"),
    ("prey_approach_latency_lo", "0.1", "5.0", "s", ">=5", "同上", "pre_registered_draft",
     "M10 §1.3", "接近潜伏期带"),
    ("habituation_index_lo", "0.20", "0.90", "", "5-7", "重复同强度刺激",
     "pre_registered_draft", "M10 §1.3", "习惯化指数（响应递减）"),
    ("habituation_recovery_lo", "1.0", "600", "s", "5-7", "同上", "pre_registered_draft",
     "M10 §1.3", "恢复时间常数带"),
    ("rest_bout_median_lo", "10", "600", "s", "5-7", "自发（压缩协议）",
     "measured_derived", "m10_imaging_reference.csv ③（ZAPBench 实测 2 状态 HMM）",
     "静息 bout 时长中位带；成像实测低活动态 dwell 中位见 imaging reference"),
    ("active_bout_median_lo", "5", "300", "s", "5-7", "自发（压缩协议）",
     "measured_derived", "m10_imaging_reference.csv ③",
     "活动 bout 时长中位带"),
    ("state_occupancy_lo_lo", "0.10", "0.90", "", "5-7", "自发", "measured_derived",
     "m10_imaging_reference.csv ③（实测占用）", "两状态占用比例（低/高活动态皆须非零）"),
]

CALIB = [
    # key, value, unit, source, note
    ("rate_band_lo", "0.30", "Hz", "m10_imaging_reference.csv::p5_rate_band_lo",
     "P5 (a) 中位发放率带下界（文献量级 0.3–1.5 Hz；清单 §4.2 预注册）"),
    ("rate_band_hi", "1.50", "Hz", "m10_imaging_reference.csv::p5_rate_band_hi", "P5 (a) 上界"),
    ("silent_band_lo", "0.50", "", "M10 §4.2 (b) 预注册",
     "P5 (b) 静默比例带下界（草案，执行时以成像/电生理文献校准；成像事件率口径见 imaging reference）"),
    ("silent_band_hi", "0.95", "", "M10 §4.2 (b) 预注册", "P5 (b) 上界"),
    ("selfsustain_ratio_max", "0.10", "", "M10 §4.2 (c) 预注册（G1 门硬判据）",
     "零外部驱动 + 零背景率 → 群率/基准群率上限（M9 实测 0.9853 = 反证基线）"),
    ("drive_elasticity_min", "0.30", "", "M10 §4.2 (d) 预注册（G1 门硬判据）",
     "4× 驱动群率相对增幅下限（M9 实测 +0.0725 = 反证基线）"),
    ("drive_monotone_required", "1", "bool", "M10 §4.2 (d) 预注册",
     "驱动倍数 {0,0.5,1,2,4,8} 群率单调不减"),
    ("single_trial_gpu_h_max", "0.50", "GPU-h", "M10 §0.7 #2 预注册（以 G0 探针实测修正）",
     "P5 (f) 单试次墙钟预算；G0 探针后若实测偏离则**追加行**修正（不删原行）"),
    ("settle_window_ms", "500", "ms", "M10 §4.2 预注册（M5 L37#2 教训）",
     "t=0 初始化瞬态剔除窗；判据只看 settle 之后"),
    ("n_trials_min", "3", "", "M10 §4.2 预注册（固定 seed）", "P5 最少试次数"),
]

LADDER = [
    # level, mechanism, param, lo, hi, unit, source_note
    ("M-1", "STP 抑制型（Tsodyks–Markram）", "u0", "0.30", "0.70", "",
     "M10 §4.1.1；语义复用 M2 冻结件 src/synapse_model.py"),
    ("M-1", "STP 抑制型", "tau_rec_s", "0.10", "2.00", "s", "M10 §4.1.1"),
    ("M-1", "STP 抑制型", "tau_fac_s", "0.01", "0.10", "s", "M10 §4.1.1（短易化）"),
    ("M-2", "STP 易化型", "u0", "0.05", "0.30", "", "M10 §4.1.1"),
    ("M-2", "STP 易化型", "tau_fac_s", "0.10", "1.00", "s", "M10 §4.1.1"),
    ("M-2", "STP 易化型", "tau_rec_s", "0.05", "0.50", "s", "M10 §4.1.1"),
    ("M-3", "GABA 亚型异质（A 快 + B 慢）", "tau_A_ms", "5", "20", "ms",
     "M10 §4.1.1；**数据侧无亚型标注** → 类级文献先验 + 测量限制（§1.1.2 允许）"),
    ("M-3", "GABA 亚型异质", "tau_B_ms", "80", "400", "ms", "M10 §4.1.1"),
    ("M-3", "GABA 亚型异质", "frac_B", "0.05", "0.50", "",
     "亚型比例扫描范围（无标注 → 扫描 + 登记为非数据驱动）"),
    ("M-4", "调质增益（tonic→phasic）", "g_mod", "0.50", "2.00", "",
     "M10 §4.1.1；须作用于**状态变量**而非仅背景率（M9 R2 教训）"),
    ("M-4", "调质增益", "tau_mod_s", "0.50", "10.00", "s", "M10 §4.1.1"),
    ("M-5", "组合（M-1..M-4 最优子集）", "n_global_params_max", "8", "8", "",
     "M10 §4.1.1 组合上限（防组合爆炸式拟合）"),
    # 机制自检门三条（每级独立评估；逐级登记胜负）
    ("GATE", "自检门 1：无输入不得自持", "selfsustain_ratio_max", "0.10", "0.10", "",
     "零驱动群率/基准群率上限；M9 实测 0.9853 → 反证基线"),
    ("GATE", "自检门 2：驱动弹性", "drive_elasticity_min", "0.30", "0.30", "",
     "4× 驱动增幅下限；M9 实测 +0.0725 → 反证基线"),
    ("GATE", "自检门 3：双态存在", "bout_fraction_min", "0.05", "1.0", "",
     "活动 bout 占比须 >0（带下界 0.05）；M9 实测 0.0000 → 反证基线"),
    ("GATE", "现象级：慢变量相关", "slow_var_corr_min", "0.30", "1.0", "",
     "机制内部慢变量轨迹 vs 群率慢涨落 |r| 下限（机制归属证据链，双轨）"),
    ("GATE", "消融：关闭机制 → 双态消失", "ablation_bout_fraction_max", "0.00", "0.02", "",
     "关闭态须回到 M9 语义（bout≈0）且逐 spike 等价（§2.3 关闭态等价）"),
]


def write_csv(path: str, header, rows) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(f"# M10 判据带 —— 运行前定稿（生成器 tools/build_m10_reference_bands.py）；不事后调\n")
        f.write(f"# 修改纪律：只允许**新增行**，原行保留（M9 L36.3）\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(list(r))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    args = ap.parse_args(argv)
    out = {}
    out["behavior"] = write_csv(
        os.path.join(args.data_dir, "m10_behavior_reference.csv"),
        ["key", "value_lo", "value_hi", "unit", "dpf", "protocol", "provenance_class",
         "source", "note"], BEHAVIOR)
    out["calibration"] = write_csv(
        os.path.join(args.data_dir, "m10_calibration_targets.csv"),
        ["key", "value", "unit", "source", "note"], CALIB)
    out["ladder"] = write_csv(
        os.path.join(args.data_dir, "m10_mechanism_ladder.csv"),
        ["level", "mechanism", "param", "lo", "hi", "unit", "note"], LADDER)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
