"""M9 §1.1 FlyWire 数据获取（可达性实测 + 断点续传；provenance 登记）。

《生物仿真M9实施清单》§1.1 权威源 + §1.5 回退预案；M8 L16 网络受限教训沿用。

实测可达性（2026-08-31，照 m8_env_notes L16 格式）：
  ✅ 可用：zenodo.org API（/api/records/{id}，1s 内 200；文件走
     /api/records/{id}/files/{key}/content → 302 对象存储；实测 852MB ≈2min ≈7MB/s，
     之后限速到 ~10–100 KB/s：**大文件不可靠**）、codex.flywire.ai（静态页/健康检查；
     交互应用需 Google 登录）、neuprint.janelia.org（可达但 **API 需凭据 401**）、
     nature.com（论文全文可达，含 Data availability）、
     storage.googleapis.com/flywire-data（bucket 可列，无连接组数据）。
  ❌ 不可用：github.com（30s 超时 000）、api.flywire.ai（000，CAVE 程序化访问不可达）、
     cave.flywire.ai（000）、biorxiv.org（429）、science.org（M8 已知 403）。

数据清单（Zenodo 10676866 = v783，CC-BY-4.0；主 agent 裁决见 m9_env_notes L14.1）：
  - proofread_connections_783.feather（852MB）**必需**（G2 主数据：逐连接 NT 平均概率）
  - proofread_root_ids_783.npy（1.1MB）**必需**（139,255 神经元权威 roster）
  - per_neuron_neuropil_count_{post,pre}_783.feather（233MB/17MB）——**弃用**（ID 空间与
    roster 不相交；见 L14.2）但保留作 provenance
  - flywire_synapses_783.feather（9.49GB）——**主 agent 裁决非 G2 必需**（逐突触级覆盖率
    登记为测量限制；网络限速 10min≈0.8MB 不可行）
  - Schlegel 2024（Zenodo 10877326）nblast_flywire_hemibrain_min_comp.feather（212MB）
    ——§1.3 映射率用；实测 99/212MB 未完成 → 测量限制
  - Eckstein 2024（Zenodo 10593546）hemibrain-v1.2-tbar-neurotransmitters.feather.bz2
    （442MB）——§1.3 递质一致性用；未下载 → 测量限制

用法：
  PYTHONHASHSEED=0 MPLBACKEND=Agg .venv-m9/bin/python -m neural_exploration.tools.fetch_m9_flywire [key ...]
  key：v783_core（默认；proofread_connections + root_ids）/ v783_all / nblast_hb / hb_nt
  **注意**：下载是网络行为，不进 build 管线（build 只读本地 m9_raw，确定性重跑不联网）。
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "m9_raw", "zenodo")

# key -> (record_id, filename, expected_bytes, required)
SOURCES = {
    "v783_core": (10676866, "proofread_connections_783.feather", 852022274, True),
    "v783_roots": (10676866, "proofread_root_ids_783.npy", 1114168, True),
    "v783_pre": (10676866, "per_neuron_neuropil_count_pre_783.feather", 16853770, False),
    "v783_post": (10676866, "per_neuron_neuropil_count_post_783.feather", 233843050, False),
    "v783_syn": (10676866, "flywire_synapses_783.feather", 9492998242, False),
    "nblast_hb": (10877326, "nblast_flywire_hemibrain_min_comp.feather", 212095362, False),
    "hb_nt": (10593546, "hemibrain-v1.2-tbar-neurotransmitters.feather.bz2",
              442710628, False),
}
DEFAULT_KEYS = ["v783_core", "v783_roots"]


def fetch(record_id: int, filename: str, expected: int) -> bool:
    """断点续传（Range）；返回是否完整。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, filename)
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have == expected:
        print("  [skip] %s 已完整（%d B）" % (filename, have), flush=True)
        return True
    url = "https://zenodo.org/api/records/%d/files/%s/content" % (record_id, filename)
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", "bytes=%d-" % have)
    print("  [get ] %s（已有 %d / %d B）" % (filename, have, expected), flush=True)
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r, open(path, "ab") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        got = os.path.getsize(path)
        ok = got == expected
        print("  [%s] %s %d B（%.1fs）" % ("ok" if ok else "part", filename, got,
                                          time.time() - t0), flush=True)
        return ok
    except Exception as e:  # noqa: BLE001 —— 网络受限如实报告（不静默）
        print("  [err ] %s: %s（已有 %d B，可重跑续传）" % (
            filename, e, os.path.getsize(path) if os.path.exists(path) else 0),
            flush=True)
        return False


def main() -> int:
    keys = sys.argv[1:] or DEFAULT_KEYS
    print("=== M9 §1.1 FlyWire 数据获取（Zenodo 官方发布；断点续传）===", flush=True)
    status = {}
    for k in keys:
        if k not in SOURCES:
            print("未知 key：%s（可选：%s）" % (k, "、".join(SOURCES)), flush=True)
            continue
        rec, fn, exp, required = SOURCES[k]
        ok = fetch(rec, fn, exp)
        status[k] = (ok, required)
        if not ok and required:
            print("  ⚠ 必需文件未完整 → build_m9_connectome.py 将失败（如实报告，"
                  "不伪造 P1 通过）", flush=True)
    print("--- 汇总 ---", flush=True)
    for k, (ok, req) in status.items():
        print("  %-12s %s%s" % (k, "完整" if ok else "未完成",
                                "（必需）" if req else "（可选/测量限制）"), flush=True)
    return 0 if all(ok for ok, req in status.values() if req) else 1


if __name__ == "__main__":
    raise SystemExit(main())
