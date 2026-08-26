# data/m5_raw/ —— M5 连接组原始数据归档（provenance 记录）

> 本目录为 `tools/build_m5_connectome.py` 的唯一直读输入（确定性重跑不依赖网络）。
> 全部文件 2026-08-25 经 GitHub API / 镜像下载（沙箱 raw.githubusercontent.com 不可达，见 docs/m5_env_notes.md L10）。

## 连接组数据

| 文件 | 来源 | 说明 |
|---|---|---|
| `herm_full_edgelist.csv` | github.com/openworm/c302 (MIT) c302/data/ | **PRIMARY**：Cook et al. 2019 *Nature* 571:63-71 雌雄同体全连接组 full edge list（Source, Target, Weight=突触计数, Type=chemical/electrical）；245,463 B |
| `herm_full_edgelist_MODIFIED.csv` | 同上（c302 运行用版本） | 对照（+15 化学对差异） |
| `SI5_adjacency.xlsx` | 同上 c302/data/（=Cook 2019 SI5 官方补充） | 官方邻接矩阵：hermaphrodite chemical / herm gap jn symmetric / herm gap jn asymmetric；4,171,572 B；逐对核对与 PRIMARY 一致 |
| `SI5_parsed.json` | 本节点解析产物 | SI5 矩阵 → dict（rows/cols/data），管线备用 |
| `herm_chem_syn.csv/` `herm_gap_syn.csv/` | networks.skewed.de/net/celegans_2019（ICON 镜像，官方 Cook 2019 synapse list） | edges.csv（突触计数）+ **nodes.csv（node_type 分类权威源：SENSORY NEURONS/INTERNEURONS/MOTOR NEURONS/PHARYNX/SEX-SPECIFIC CELLS）** |
| `aconnectome_white_1986_whole.csv` | c302 仓库 c302/data/ | White 1986 数字化（pre/post/type/synapses；对照源；神经元-神经元化学 7,914/缝隙 971） |
| `aconnectome_white_1986_A.csv` `_L4.csv` | 同上 | White 1986 前部/L4 子集（对照） |
| `wormwiring_N2U.txt` | c302 仓库 c302/data/ | wormwiring N2U 连接（对照） |

## 标注数据

| 文件 | 来源 | 说明 |
|---|---|---|
| `owmeta_cache.json` | c302 仓库 c302/data/（OpenWorm owmeta） | **302 神经元规范 roster + class + neurotransmitter（文献编译：Pereira 2015 / Serrano-Saiz 2013 等）** |
| `CElegansNeuronTables.xls` | c302 仓库 c302/data/ | Connectome/NeuronsToMuscle/Sensory 三表（肌肉→递质，对照） |
| `Bentley_et_al_2016_expression.csv` | c302 仓库 c302/data/ | 神经肽表达（辅助标注） |

## 复现

```bash
cd /Users/weidong/ai/small_world
.venv-neuro/bin/python -m neural_exploration.tools.build_m5_connectome
```

输出：data/m5_connectome.csv（sha256 稳定）+ data/m5_connectome_counts.json + data/m5_crosscheck_m3m4.csv + data/m5_{pharynx,command,chemotaxis}_subgraph.csv
