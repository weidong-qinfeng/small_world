# M0 环境前置确认报告（第一版）

> 对应《生物仿真M0实施清单》§1 前置确认。
> 检查时间：2026-08-22 14:32 CST
> 检查人：M0 开工（自动化执行）
> 状态：**C1–C5 全部通过 → 步骤 1 走「在线安装」方案**

---

## 1. 结论速览

| # | 检查项 | 结果 | 结论 |
|---|---|---|---|
| C1 | 芯片架构 | `arm64`（Apple M1 Pro） | 用 arm64 wheel；NEURON 官方提供 macosx arm64 wheel，pip 直装可行 |
| C2 | 网络可用性 | ✅ pypi.org / github.com 均可达，`pip download numpy` 成功 | **走在线安装**，不需要离线 wheel 目录 |
| C3 | 系统 Python | `/usr/bin/python3` = 3.9.6（Xcode 系统 Python，无 brew 版本） | **必须用 venv**，禁止直接装系统 Python |
| C4 | 磁盘空间 | 409 GiB 可用（总 460 GiB，已用 8%） | ✅ 远超 2 GiB 需求 |
| C5 | git 状态 | `main` 分支领先 origin 2 个提交；`neural_exploration/` 未跟踪 | M0 产物天然隔离，可独立提交，不混入 M3 改动 |

**最终结论**：环境前置无阻塞项。pypi 网络通畅（清单 §1 记录的「github 443 超时」已知风险当前已消失），**步骤 1（venv + 依赖安装）可以直接按在线方案开工**。

---

## 2. C1 — 芯片架构

```text
uname -m          → arm64
uname -a          → Darwin 23.6.0 (macOS 14.8.7)
machdep.cpu.brand → Apple M1 Pro
```

**影响**：
- 所有包使用 `arm64` wheel（macOS 14 目标，如 `macosx_14_0_arm64`）。
- NEURON：官方 PyPI wheel 支持 macOS arm64，预期 `pip install neuron` 成功（无需源码编译）。
- NEST：macOS arm64 仍大概率需源码编译（cmake），按清单 §8 只做「安装可行性」评估（10 分钟超时放弃，打 1 分）。

---

## 3. C2 — 网络可用性

| 探测 | 结果 |
|---|---|
| `curl -I https://pypi.org/simple/` | HTTP/2 200 |
| `curl -I https://github.com` | HTTP/2 200（清单记录的 443 超时风险**已解除**） |
| `pip download --no-deps numpy -d /tmp/m0_pypi_test` | ✅ 成功：`numpy-2.0.2-cp39-cp39-macosx_14_0_arm64.whl`（5.3 MB） |

**结论**：pypi 直连可用 → 步骤 1 采用**在线安装**方案（§8 风险表中的备用方案①镜像源、②离线 wheel 暂不需要，留作故障预案）。

---

## 4. C3 — 系统 Python 位置

```text
which -a python3  → /usr/bin/python3
python3 --version → Python 3.9.6 (Xcode 系统 Python)
pip3 --version    → pip 21.2.4 (来自 Xcode Python3.framework)
brew              → 未安装（无 /opt/homebrew/bin/python3*）
```

**结论**：
- 全机唯一 Python 是 Xcode 系统 Python 3.9.6 → **必须建 venv**（清单 §2.1 的 `.venv-neuro`）。
- ⚠️ **风险记录**：Python 3.9.6 已接近官方 EOL（安全补丁已停更），且 pip 21.2.4 较旧。M0 不阻塞（numpy 2.0.2 等 cp39 wheel 仍可得），但建议 M0 结束后评估安装独立 Python 3.11/3.12（python.org 安装包或先装 Homebrew）。
- venv 内第一步 `python -m pip install --upgrade pip`（清单 §2.1）必要——升级到 26.x 避免旧 pip 解析问题。

---

## 5. C4 — 磁盘空间

```text
df -h ~  → /dev/disk3s1  460Gi  35Gi  409Gi  8%  /System/Volumes/Data
```

**结论**：409 GiB 可用，远超「≥2 GiB」需求（引擎 wheel + 仿真数据 + 报告），无压力。

---

## 6. C5 — git 状态

```text
分支：main，领先 origin/main 2 个提交（最近：8cbe831 feat(M3): SRL 语义角色标注…）

工作区：
  modified:   .DS_Store                                  ← 已被 git 跟踪的历史遗留，与 M0 无关
  modified:   digital_brain/src/core/pattern/event_state_machine.py  ← M3 未提交改动，与 M0 无关
  untracked:  neural_exploration/                        ← M0 全部产物所在
```

**结论**：
- `neural_exploration/` 是未跟踪目录 → M0 产物与未推送的 M3 改动**天然隔离**，满足清单 C5 要求。
- **建议**：M0 阶段只 `git add neural_exploration/` 独立提交，不触碰 `.DS_Store` 与 `digital_brain/` 的 M3 改动。
- 注：`.gitignore` 已含 `.DS_Store` 但该文件历史上已被跟踪，故仍显示 modified；不影响 M0。

---

## 7. 对后续步骤的影响

1. **步骤 1（venv + 依赖）**：可直接开工，在线安装。命令照抄清单 §2：
   ```bash
   cd /Users/weidong/ai/small_world
   python3 -m venv .venv-neuro
   source .venv-neuro/bin/activate
   python -m pip install --upgrade pip
   pip install numpy scipy matplotlib pytest
   ```
2. **引擎安装预期**（M0 实测前为预期值，装完回填基准表 §3）：
   - Brian2：纯 Python，预期成功（在线）。
   - NEURON：有 macosx arm64 wheel，预期成功。
   - NEST：预期需源码编译，仅做 10 分钟可行性评估。
3. **版本锁定**：步骤 1 完成后 `pip freeze > neural_exploration/docs/m0_requirements.lock`（M0 结束时更新）。

---

## 8. 遗留风险与预案

| 风险 | 状态 | 预案 |
|---|---|---|
| Python 3.9.6 接近 EOL | ⚠️ 不阻塞 M0 | M0 后装独立 3.11/3.12（python.org / Homebrew） |
| pip 21.2.4 过旧 | ⚠️ 不阻塞 | venv 内先 `pip install --upgrade pip` |
| github.com 443 曾超时 | ✅ 已恢复 | 若再超时：git 操作走 ssh / 镜像，装包不受影响（pypi 独立） |
| pypi 中断 | ✅ 当前通畅 | 预案①清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`；预案②离线 wheel 目录 |

---

## 9. 步骤 1 执行结果（M0 实施回填，2026-08-22）

| 项 | 结果 |
|---|---|
| venv | `.venv-neuro`（Python 3.9.6）创建成功，pip 升级 21.2.4 → 26.0.1 |
| 基础栈 | numpy 1.26.4 / scipy 1.13.1 / matplotlib 3.9.4 / pytest 8.4.2 ✅ |
| Brian2 | 2.6.0 ✅（pip 秒装） |
| NEURON | 9.0.1 ✅（官方 arm64 wheel，pip 秒装） |
| NEST | ❌ 源码编译失败（本机无 cmake），按清单 §8 打 1 分，待 M5 评估 |
| 版本锁定 | `docs/m0_requirements.lock`（29 包，一条命令重建） |

**踩坑记录（已解决，供复现参考）**：
1. numpy 2.0.2 与 brian2 2.6.0 预编译扩展（cythonspikequeue.so）二进制不兼容 → 锁定 numpy 1.26.4。
2. brian2 2.6.0 不认 Cython 3.x（判为"不可用"）→ 锁定 Cython 0.29.37。
3. Brian2 HH 神经元若无 reset/refractory，阈值条件持续为真时每个时间步发尖峰（官方 HH 示例均带 refractory）→ 统一加 refractory=2ms。
4. Brian2 方程单位：门控速率需 `(0.1/mV)*…` 化无量纲；变量声明只能用基本单位（siemens/meter**2 而非 mS/cm2）。

---

*第一版完成：C1–C5 结论已记录。下一步：按清单 §2 执行步骤 1（建 venv + 装依赖），随后回填 `m0_engine_benchmark.md`。*
