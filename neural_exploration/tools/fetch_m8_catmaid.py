"""M8 数据获取：L1EM CATMAID（Virtual Fly Brain 存档实例）神经元清单/标注拉取。

权威源：Winding et al. 2023 (Science 379:eadd9330) 论文 Data availability 指定的
CATMAID 接口（L1 Larval CNS）：https://l1em.catmaid.virtualflybrain.org/（VFB 存档）。
本脚本用 CATMAID REST API（CSRF cookie 会话）拉取：
  1) 论文分析图 roster 标注（mw brain and inputs / mw brain accessory neurons）
  2) 排除标注（mw brain very incomplete / mw partially differentiated / mw motor）
  3) 递质标注（chol/gaba/glut/DA/5HT/OA/TA/肽能 + mw 变体）
  4) 区域/类标注（Brain/VNC/SEZ/SOG/A1-A9/T1-T3/Sensories/motorneurons/...）
  5) 命名神经元（Brain 下 type=neuron 的 annotation 实体 → 名称 + skeleton_ids）

输出：data/m8_raw/catmaid/<tag>.json（原始响应，供 build_m8_connectome.py 解析）。
用法：.venv-neuro/bin/python tools/fetch_m8_catmaid.py [tag ...]
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.parse
import http.cookiejar

BASE = "https://l1em.catmaid.virtualflybrain.org"
PID = "1"
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "m8_raw", "catmaid"
)

# (tag, [annotation names]) —— 一组标注一次性查询
QUERIES = {
    # roster 由 S1 矩阵 2952 + excluded 推导（见 build_m8_connectome.py），不做大查询
    "excluded": ["mw brain very incomplete", "mw partially differentiated", "mw motor"],
    "accessory": ["mw brain accessory neurons"],
    "nt": [
        "Cholinergic", "GABAergic", "Glutamatergic", "Dopaminergic", "Serotonergic",
        "Octopaminergic", "peptidergic", "neuropeptidergic", "acetylcholine", "glutamate",
        "dopamine", "octopamine", "GABA", "mw cholinergic", "mw GABAergic", "mw glutamatergic",
        "mw dopaminergic", "mw octopaminergic", "mw cholinergic and glutamatergic",
        "mw GABAergic and glutamatergic", "potential GABA (segregation index)",
    ],
    "region": [
        "Brain", "VNC", "SEZ", "SOG", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9",
        "T1", "T2", "T3", "Left", "Right", "Unpaired", "unpaired",
    ],
    "class": [
        "Sensories", "sensory", "motorneurons", "MN motor neurons", "Interneurons",
        "Local Interneurons", "Kenyon cells", "class IV", "ORN", "ORNs right", "ORNs left",
        "Odor Projection Neurons (1 to 1)", "Projection Neurons from VNC", "MBON", "DAN",
        "Output", "output", "ascending", "DN-VNC", "DN-SEZ", "RGN",
    ],
    "brain_named": ["Brain"],
}


def make_opener():
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    # 先 GET 主页拿 CSRF cookie
    opener.open(BASE + "/", timeout=120).read()
    csrf = None
    for c in cj:
        if c.name.startswith("csrftoken"):
            csrf = c.value
    return opener, csrf


def query_targets(opener, csrf, annotations, tag):
    url = "%s/%s/annotations/query-targets" % (BASE, PID)
    body = json.dumps(
        {
            "annotated_objects": [{"annotation": a} for a in annotations],
            "with_annotations": True,
            "with_names": True,
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Referer", BASE + "/")
    if csrf:
        req.add_header("X-CSRFToken", csrf)
    t0 = time.time()
    with opener.open(req, timeout=1800) as resp:
        data = resp.read()
    out = os.path.join(OUT_DIR, "%s.json" % tag)
    with open(out, "wb") as f:
        f.write(data)
    # 校验可解析
    try:
        parsed = json.loads(data.decode("utf-8"))
        n = len(parsed.get("entities", []))
    except Exception as e:
        n = "PARSE_ERR %s" % e
    print("[%s] %s bytes, entities=%s, %.1fs -> %s" % (tag, len(data), n, time.time() - t0, out), flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tags = sys.argv[1:] or list(QUERIES.keys())
    opener, csrf = make_opener()
    print("csrf ok:", bool(csrf), flush=True)
    for tag in tags:
        anns = QUERIES[tag]
        try:
            query_targets(opener, csrf, anns, tag)
        except Exception as e:
            print("[%s] ERROR: %s" % (tag, e), flush=True)
            # 若被限流，等 30s 重试一次
            time.sleep(30)
            try:
                query_targets(opener, csrf, anns, tag)
            except Exception as e2:
                print("[%s] RETRY ERROR: %s" % (tag, e2), flush=True)


if __name__ == "__main__":
    main()
