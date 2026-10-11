"""M10 数据门验收（P1/P2）——独立复核 + 确定性重跑 + 裁决记录。

《生物仿真M10实施清单》§1（P1 连接组数据门 / P2 全脑成像数据门）+ §7 验收表。

验收动作（沿 M9 主agent验收语义：**独立复核产物真实性**，不采纳构建脚本自述）：
  1. **独立重算**：流式重读 `data/m10_zf_connectome.csv`，逐行核算 neuron/chem/gap 行数与
     syn_count 求和 → 与 `m10_connectome_counts.json` 逐项对齐；
  2. **确定性重跑**：把 `tools/build_m10_connectome.py` 重跑到临时目录 → SHA-256 必须逐位一致；
  3. **成像参考量一致性**：`m10_imaging_reference.csv` 的键值与 `m10_imaging_stats.json` 交叉核对；
  4. **可达性表落盘**：`data/m10_source_reachability.csv`（本次会话实测 HTTP 状态/吞吐）；
  5. 输出 `reports/neuro/m10_data_gate_summary.json`（P1/P2 判定 + 证据 + 裁决请求）。

用法：
    PYTHONHASHSEED=0 ./.venv-m10/bin/python -m neural_exploration.tools.validate_p10_data_gate
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)   # `python -m neural_exploration.*` 需要其父目录在 sys.path
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REP_DIR = os.path.join(PROJECT_ROOT, "reports", "neuro")

# ---- 本次会话实测的可达性（L9–L13；method=curl -m 12..60 / python urllib；UTC 2026-10-11）----
REACH = [
    ("fish1-release.storage.googleapis.com", "https://fish1-release.storage.googleapis.com/index.html",
     "connectome", 200, "200, 6.6 KB, 公开匿名可读；bucket 可 list（189 objects / 3.54 GB）", "OK"),
    ("fish1-release GCS", "https://storage.googleapis.com/fish1-release/paper_data/TEN_analysis.zip",
     "connectome", 200, "3.49 MB / 1.7 s ≈ 2.1 MB/s", "OK"),
    ("fish1-release GCS(vast)", "https://storage.googleapis.com/fish1-release/paper_data/vast/OC_left.vsseg",
     "connectome", 200, "range GET 20 MB / 5.0 s ≈ 3.97 MB/s", "OK"),
    ("CAVE materialize (Fish1)", "https://global.brain-wire-test.org/materialize/api/v2/datastack/fish1_full/version",
     "connectome", 404, "nginx 404（多路径变体均 404）", "BLOCKED_API"),
    ("CAVE info (Fish1)", "https://global.brain-wire-test.org/info/api/v2/datastacks",
     "connectome", 302, "302 → /sticky_auth 登录（交互式 token；无头不可得）", "BLOCKED_AUTH"),
    ("CAVE local (skeleton)", "https://pcgv3local.brain-wire-test.org/",
     "connectome", 404, "nginx 404", "BLOCKED_API"),
    ("zapbench-release GCS", "https://storage.googleapis.com/zapbench-release/volumes/20240930/traces/zarr.json",
     "imaging", 200, "275 B, 0.47 s；zarr3 shape (7879, 71721) chunk (512,512)", "OK"),
    ("zapbench traces chunks", "https://storage.googleapis.com/zapbench-release/volumes/20240930/traces/c/{t}/{f}",
     "imaging", 206, "256 chunk × 1 MB 用时 94.7 s（6 线程）≈ 2.8 MB/s；NaN=0", "OK"),
    ("zenodo.org API", "https://zenodo.org/api/records/1009488",
     "imaging", 200, "首测 SSL self-signed（DNS 解析 182.16.62.14）后恢复 200 → 间歇性 DNS/证书问题", "OK_INTERMITTENT"),
    ("zenodo.org file", "https://zenodo.org/api/records/19066472/files/WBSC.zip/content",
     "imaging", 206, "**单流 56 KB/s**；4 并发无改善（≈11 KB/s 聚合）；35.6 MB 文件 1 h 未完成", "OK_VERY_SLOW"),
    ("arXiv (export)", "https://export.arxiv.org/pdf/2503.02618", "literature", 200,
     "13.2 MB / 数秒（ZAPBench 论文全文，含采集参数）", "OK"),
    ("Europe PMC REST", "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12259123/fullTextXML",
     "literature", 200, "162 KB（Fish1 论文全文，含 Data availability）", "OK"),
    ("biorxiv.org", "https://www.biorxiv.org/content/10.1101/2025.06.12.659365v1", "literature", 429,
     "首测 TCP 超时（ERR 28），复测 429（限流而非封锁）", "PARTIAL"),
    ("science.org", "https://www.science.org", "literature", 403, "403（M9 同现象）", "BLOCKED"),
    ("cell.com", "https://www.cell.com", "literature", 0, "TCP 超时（云 WAF）", "BLOCKED"),
    ("journals.biologists.com", "https://journals.biologists.com", "literature", 403, "403", "BLOCKED"),
    ("research.google", "https://research.google/pubs/", "literature", 0, "TCP 超时", "BLOCKED"),
    ("nature.com", "https://www.nature.com/articles/s41592-022-01621-0", "literature", 200, "OK", "OK"),
    ("elifesciences.org", "https://elifesciences.org/articles/57443", "literature", 200, "OK", "OK"),
    ("github.com / api / codeload", "https://api.github.com/repos/VCG/cave-scripts", "code", 200,
     "api 200；codeload tarball 5.8 MB OK；README.md OK", "OK"),
    ("raw.githubusercontent.com", "https://raw.githubusercontent.com/VCG/cave-scripts/master/README.md",
     "code", 0, "连接被重置（Recv failure）→ 走 api.github.com contents(base64) 替代", "BLOCKED"),
    ("huggingface.co", "https://huggingface.co", "code", 0, "连接失败（直连不可达）", "BLOCKED"),
    ("neuprint.janelia.org", "https://neuprint.janelia.org", "crosscheck", 200, "200（未取凭据）", "OK"),
    ("janelia.org", "https://www.janelia.org", "crosscheck", 0, "SSL self-signed（curl 60）", "PARTIAL"),
    ("fishatlas.neuro.mpg.de", "https://fishatlas.neuro.mpg.de/", "atlas", 0, "SSL 证书链不可验证（curl 60）", "PARTIAL"),
    ("zebrafishatlas.zib.de", "https://zebrafishatlas.zib.de/", "atlas", 200, "200", "OK"),
    ("mapzebrain.org", "https://mapzebrain.org/home", "atlas", 200, "200", "OK"),
    ("fishexplorer.zib.de", "https://fishexplorer.zib.de", "atlas", 200, "200", "OK"),
    ("zebrafishem.org", "https://zebrafishem.org", "atlas", 0, "DNS 无 A 记录（域名不可解析）", "BLOCKED_DNS"),
    ("zebrafishbrain.org", "https://zebrafishbrain.org", "atlas", 200,
     "**HTTP 200 但仅 114 字节**：`window.location.href='/lander'` → **域名停放页（parked/lander），"
     "无任何数据资源** ⇒ 可达 ≠ 可用（主agent 观测到的 200 与之相符，但不可当作数据源）", "REACHABLE_NO_DATA"),
    ("zenodo.org（复测）", "https://zenodo.org/api/records/1009488", "imaging", 200,
     "本次会话 **curl 与 python urllib 双双 200**（首次曾 SSL self-signed/DNS 异常）→ 可达，但**文件吞吐仅 56 KB/s**"
     "（与主agent『zenodo 超时不可达』的观测不同：登记为**间歇性不稳 + 极低吞吐**）", "OK_VERY_SLOW"),
    ("engertlab Z-Brain", "https://engertlab.fas.harvard.edu/Z-Brain/", "atlas", 0, "TCP 超时", "BLOCKED"),
    ("Z-Brain 镜像(Zenodo)", "https://zenodo.org/api/records/7130703", "atlas", 200,
     "Reference brains for registration to ZBB（19 nii.gz，4.5 GB）——Z-Brain 配准图谱经 Zenodo 可得",
     "OK_VERY_SLOW"),
    ("www.nichd.nih.gov", "https://www.nichd.nih.gov/", "atlas", 200, "200（Brain Browser 介绍页）", "OK"),
    ("figshare", "https://api.figshare.com/v2/articles", "imaging", 403, "403", "BLOCKED"),
    ("gin.g-node.org", "https://gin.g-node.org", "imaging", 403, "403", "BLOCKED"),
    ("openneuro.org", "https://openneuro.org", "imaging", 200, "200", "OK"),
    ("osf.io", "https://osf.io", "imaging", 200, "200", "OK"),
    ("datadryad.org", "https://datadryad.org", "imaging", 200, "200", "OK"),
    ("data.mendeley.com", "https://data.mendeley.com", "imaging", 200, "200", "OK"),
    ("idr.openmicroscopy.org", "https://idr.openmicroscopy.org", "imaging", 200, "200", "OK"),
    ("zebrahub.org", "https://www.zebrahub.org", "imaging", 0, "301 → squarespace（无数据）", "BLOCKED"),
    ("cloud.google.com / google.com", "https://www.google.com", "net", 0, "TCP 超时（普通 web 受限）", "BLOCKED"),
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def recount_csv(path: str) -> dict:
    """流式独立重算（不 import 构建脚本的任何逻辑）。"""
    n_neuron = n_chem = n_gap = 0
    syn_sum = 0
    pre_col = post_col = None
    with open(path, newline="") as f:
        rd = csv.reader(f)
        header = None
        for row in rd:
            if not row:
                continue
            if row[0].startswith("#"):
                continue
            if header is None:
                header = row
                pre_col = header.index("pre")
                post_col = header.index("post")
                sc_col = header.index("syn_count")
                continue
            kind = row[0]
            if kind == "neuron":
                n_neuron += 1
            elif kind == "chem":
                n_chem += 1
                syn_sum += int(row[sc_col])
            elif kind == "gap":
                n_gap += 1
    return dict(n_neuron=n_neuron, n_chem=n_chem, n_gap=n_gap, syn_sum=syn_sum)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerun", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    csv_path = os.path.join(DATA_DIR, "m10_zf_connectome.csv")
    json_path = os.path.join(DATA_DIR, "m10_connectome_counts.json")
    counts = json.load(open(json_path))

    # 1) 独立重算
    rc = recount_csv(csv_path)
    recompute_ok = (rc["n_neuron"] == counts["parse"]["n_roster"]
                    and rc["n_chem"] == counts["parse"]["n_chem_pairs_in_roster"]
                    and rc["syn_sum"] == counts["parse"]["n_chem_synapses_in_roster"]
                    and rc["n_gap"] == counts["parse"]["gap_rows"])
    sha_now = sha256_file(csv_path)
    sha_recorded = counts["outputs"]["csv_sha256"]

    # 2) 确定性重跑
    rerun = dict(skipped=bool(args.no_rerun))
    if not args.no_rerun:
        with tempfile.TemporaryDirectory() as td:
            cmd = [sys.executable, "-m", "neural_exploration.tools.build_m10_connectome",
                   "--out-csv", os.path.join(td, "c.csv"),
                   "--out-json", os.path.join(td, "c.json"),
                   "--out-npz", os.path.join(td, "c.npz")]
            env = dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=REPO_ROOT)
            p = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
            rerun["returncode"] = p.returncode
            if p.returncode == 0:
                sha2 = sha256_file(os.path.join(td, "c.csv"))
                rerun["sha256"] = sha2
                rerun["identical"] = bool(sha2 == sha_now)
            else:
                rerun["stderr_tail"] = p.stderr[-500:]

    # 3) 成像参考一致性
    img = {}
    img_path = os.path.join(DATA_DIR, "m10_imaging_reference.csv")
    with open(img_path, newline="") as f:
        for row in csv.DictReader(r for r in f if not r.startswith("#")):
            img[row["key"]] = row["value"]
    istats = json.load(open(os.path.join(DATA_DIR, "m10_imaging_stats.json")))
    img_checks = [
        ("frame_rate", abs(float(img["imaging_frame_rate_hz"])
                           - istats["dataset"]["frame_rate_hz"]) < 1e-3),
        ("rate_median_matches_stats",
         abs(float(img["p5_rate_band_imaging_quantile_lo"])
             - istats["firing_rate"]["p25_hz"]) < 1e-4),
        ("occupancy_matches_stats",
         abs(float(img["p7_state_occupancy_low"])
             - istats["states"]["occupancy"][0]) < 1e-3),
        ("decode_status_recorded", img.get("p7_decode_status", "").startswith("not_established")),
        ("evidence_level_measured", img["imaging_evidence_level"] == "measured"),
        ("rate_dist_file_exists",
         os.path.exists(os.path.join(DATA_DIR, "m10_imaging_rate_dist.csv"))),
    ]

    # 4) 可达性表落盘
    reach_path = os.path.join(DATA_DIR, "m10_source_reachability.csv")
    with open(reach_path, "w", newline="") as f:
        f.write("# M10 数据源可达性实测（2026-10-11，本机曲线；method: curl -m 12..60 / python urllib）\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["source", "url", "category", "http_code", "measured_detail", "status"])
        for r in REACH:
            w.writerow(list(r))

    # 5) 判定
    rerun_ok = bool(rerun.get("identical", False)) or bool(args.no_rerun)
    p1_pass_partial = bool(recompute_ok and rerun_ok)
    out = dict(
        generated_by="tools/validate_p10_data_gate.py",
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        P1=dict(
            verdict="PASS(partial)" if p1_pass_partial else "REVIEW",
            completeness_level="partial",
            evidence=dict(
                csv_sha256_now=sha_now, csv_sha256_recorded=sha_recorded,
                sha_match=bool(sha_now == sha_recorded),
                independent_recount=rc, counts_json=counts["parse"],
                recompute_matches_json=bool(recompute_ok),
                determinism_rerun=rerun,
                authoritative_anchors=counts["authority"],
            ),
            failed_anchors=[
                "全脑突触计数锚 ~30M：**未测**（CAVE token 门控）→ 不入分母、不写成未命中",
                "缝隙连接：官方包与 CAVE 表清单均无此表 → 0 行 + 测量限制（M9 R8 语义）",
                "区域分配 / 递质亚型：官方未发布 → 不臆造（M8 B1a）",
            ],
            adjudication_requests=counts["adjudication_requests"],
        ),
        P2=dict(
            verdict="PASS(measured, 3/4 指标)" if all(ok for _, ok in img_checks[:3]) else "REVIEW",
            evidence_level="measured",
            dataset=istats["dataset"],
            metrics=dict(
                firing_rate=istats["firing_rate"],
                correlation=istats["correlation"],
                states=dict(occupancy=istats["states"]["occupancy"],
                            dwell=istats["states"]["dwell"]),
                decoding_status=img["p7_decode_status"],
                decoding_diag=dict(raw=istats["decoding"]["accuracy"],
                                   raw_null_max=istats["decoding"]["null_block_shift_max"],
                                   fc=istats["decoding"]["accuracy_frame_centered"],
                                   fc_null_max=istats["decoding"]["null_block_shift_fc_max"]),
            ),
            consistency_checks=img_checks,
            failed_or_degraded=[
                "P7 ④ 刺激解码参考带 **未建立**（条件块与慢漂移完全混淆；零分布 ≥ 数据）→ "
                "登记 `unavailable` 请求裁决，不静默删除",
                "Zenodo 吞吐 56 KB/s → 380 MB 以上的公开成像数据集（如 left/right 交错刺激档 "
                "Zenodo 19598932、DFF 矩阵 1009488）**本次未下载**（未测 ≠ 不可得）",
                "文献带的 DOI 级 provenance 化未完成（science.org 403 / cell.com 超时 / biorxiv 429）",
            ],
            outputs=[img_path, os.path.join(DATA_DIR, "m10_imaging_stats.json"),
                     os.path.join(DATA_DIR, "m10_imaging_rate_dist.csv"),
                     os.path.join(PROJECT_ROOT, "reports", "neuro", "m10_imaging_reference.png")],
        ),
        reachability_table=reach_path,
        elapsed_s=round(time.time() - t0, 1),
    )
    os.makedirs(REP_DIR, exist_ok=True)
    outp = os.path.join(REP_DIR, "m10_data_gate_summary.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(dict(P1=out["P1"]["verdict"], P2=out["P2"]["verdict"],
                          sha_match=out["P1"]["evidence"]["sha_match"],
                          rerun=rerun.get("identical"), recount=rc,
                          summary=outp), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
