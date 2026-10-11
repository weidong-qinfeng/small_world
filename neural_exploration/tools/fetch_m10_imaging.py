"""M10 数据门 P2：ZAPBench（全脑光片功能成像）最小必要子集获取工具。

《生物仿真M10实施清单》§1.2（全脑活动成像数据门，P2 验证对象）。

数据源（L1 首选，实测可达性见 `docs/m10_env_notes.md` L9+）：
  ZAPBench = Zebrafish Activity Prediction Benchmark（Google Research + HHMI Janelia + Harvard）
  - 论文：arXiv:2503.02618（ICLR 2025）；代码：https://github.com/google-research/zapbench
  - 数据：`gs://zapbench-release/`（公开匿名可读；HTTP 等价 `https://storage.googleapis.com/zapbench-release/`）
  - 许可：CC-BY-4.0；功能记录由 Ahrens lab（HHMI Janelia）采集
  - 内容：斑马鱼幼鱼全脑 71,721 神经元 × 7,879 时间步钙活动（zarr3，chunk 512×512 float32）
          + 9 个视觉刺激条件的时间段标注（`zapbench/constants.py::CONDITION_OFFSETS/NAMES`）

本工具只做**确定性、可复现的最小拉取**：按 zarr3 默认 chunk 键 `c/{ti}/{fi}` 逐块 GET，
写入 `.npy`（大端无关，float32 memmap）。**不修改任何冻结件**（M10 §0.3 #9）。

用法：
    PYTHONHASHSEED=0 ./.venv-m10/bin/python -m neural_exploration.tools.fetch_m10_imaging \
        --n-neurons 8192 --out data/m10_raw/zapbench_traces_8192.f32.npy
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BUCKET = "zapbench-release"
BASE = f"https://storage.googleapis.com/{BUCKET}"
TRACES_PREFIX = "volumes/20240930/traces"
ZARR_META = f"{TRACES_PREFIX}/zarr.json"
# 数据源自身元数据（zarr.json 读出，硬编码仅作断言上界）
TRACE_SHAPE = (7879, 71721)
CHUNK_SHAPE = (512, 512)


def curl_bytes(url: str, tries: int = 6, timeout: int = 120) -> bytes:
    """经 `curl` 子进程取字节。

    **环境实测（M10 L11）**：本机 Python 进程（venv-m10 的 LibreSSL/`urllib`）对
    `storage.googleapis.com` 的 TLS 握手会超时（`_ssl.c:1112 handshake operation timed out`），
    而 `curl` 同 URL 正常（HTTP/2）→ 数据门一律以 curl 为传输后端（可复现、不静默）。
    """
    out = None
    for i in range(tries):
        p = subprocess.run(["curl", "-sS", "-m", str(timeout), "--retry", "2", url],
                           capture_output=True)
        if p.returncode == 0 and p.stdout:
            return p.stdout
        out = f"rc={p.returncode} err={p.stderr.decode('utf8', 'replace')[:200]}"
        time.sleep(min(2 ** i, 20))
    raise RuntimeError(f"curl failed after {tries} tries: {url}: {out}")


def zarr_meta() -> dict:
    return json.loads(curl_bytes(f"{BASE}/{ZARR_META}"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-neurons", type=int, default=8192)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "m10_raw",
                                                  "zapbench_traces_8192.f32.npy"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--provenance", default=os.path.join(DATA_DIR, "m10_raw",
                                                         "zapbench_provenance.json"))
    args = ap.parse_args(argv)

    meta = zarr_meta()
    shape = tuple(meta["shape"])
    chunk = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
    # 读回断言（M9 教训 B3：参数必须读回核对，避免静默失效）
    assert shape == TRACE_SHAPE, f"traces shape 读回 {shape} != 预注册 {TRACE_SHAPE}"
    assert chunk == CHUNK_SHAPE, f"chunk shape 读回 {chunk} != 预注册 {CHUNK_SHAPE}"
    assert meta["data_type"] == "float32" and meta["zarr_format"] == 3
    assert meta["codecs"][0]["name"] == "bytes", "仅支持无压缩（bytes codec）chunk"

    nt = int(np.ceil(shape[0] / chunk[0]))
    nf_keep = int(np.ceil(args.n_neurons / chunk[1]))
    nf_keep = min(nf_keep, int(np.ceil(shape[1] / chunk[1])))
    n_cols = min(args.n_neurons, shape[1])
    print(f"[fetch] shape={shape} chunk={chunk} t-chunks={nt} f-chunks={nf_keep} cols={n_cols}",
          flush=True)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32,
                                    shape=(shape[0], n_cols))
    arr[:] = np.nan
    tasks = [(ti, fi) for ti in range(nt) for fi in range(nf_keep)]
    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def worker(tasks_):
        for ti, fi in tasks_:
            url = f"{BASE}/{TRACES_PREFIX}/c/{ti}/{fi}"
            raw = curl_bytes(url)
            exp = chunk[0] * chunk[1] * 4
            if len(raw) != exp:
                raise RuntimeError(f"chunk {ti}/{fi} 大小 {len(raw)} != {exp}")
            blk = np.frombuffer(raw, dtype="<f4").reshape(chunk)
            r0, r1 = ti * chunk[0], min((ti + 1) * chunk[0], shape[0])
            c0 = fi * chunk[1]
            c1 = min(c0 + chunk[1], n_cols)
            arr[r0:r1, c0:c1] = blk[: r1 - r0, : c1 - c0]
            with lock:
                done[0] += 1
                if done[0] % 32 == 0 or done[0] == len(tasks):
                    el = time.time() - t0
                    print(f"[fetch] {done[0]}/{len(tasks)} chunks  {el:.1f}s "
                          f"({done[0] * chunk[0] * chunk[1] * 4 / el / 1e6:.1f} MB/s)", flush=True)

    per = max(1, len(tasks) // max(1, args.threads))
    ths = []
    for i in range(args.threads):
        seg = tasks[i * per:(i + 1) * per] if i < args.threads - 1 else tasks[i * per:]
        if not seg:
            continue
        t = threading.Thread(target=worker, args=(seg,))
        t.start()
        ths.append(t)
    for t in ths:
        t.join()
    arr.flush()
    del arr

    # 完整性断言：不允许 NaN（未拉取区段会留下 NaN）
    mm = np.load(out, mmap_mode="r")
    n_nan = int(np.isnan(mm).sum())
    assert n_nan == 0, f"存在 {n_nan} 个 NaN —— 拉取不完整"
    prov = dict(
        generated_by="tools/fetch_m10_imaging.py",
        source=dict(bucket=BUCKET, prefix=TRACES_PREFIX, zarr_json=meta, license="CC-BY-4.0",
                    citation="ZAPBench (arXiv:2503.02618); recordings by Ahrens lab, HHMI Janelia"),
        array=dict(path=out, dtype="float32", shape=[int(mm.shape[0]), int(mm.shape[1])],
                   n_neurons_kept=int(n_cols), n_time_steps=int(mm.shape[0])),
        checks=dict(nan_count=n_nan, min=float(np.nanmin(mm[:200])), max=float(np.nanmax(mm[:200]))),
        elapsed_s=round(time.time() - t0, 1),
        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(args.provenance, "w") as f:
        json.dump(prov, f, indent=1)
    print(f"[fetch] done -> {out}  {prov['elapsed_s']}s  NaN={n_nan}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
