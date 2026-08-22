"""M0 最小闭环冒烟测试（清单 §6）：用主线引擎（Brian2）实现。

    外部刺激（电流注入）
      → 感觉神经元（HH，产生动作电位）
      → 兴奋性化学突触（EPSP）
      → 运动神经元（HH，跨阈值发放）
      → 虚拟肌肉（积分器：发放 → 收缩量）

确定性铁律：本模块不含任何随机性 → 同参数重跑结果逐位一致。

用法：
  python -m neural_exploration.src.smoke_loop            # 跑默认刺激 + 出图 + 打印摘要
  python -m neural_exploration.src.smoke_loop --amp 0    # 无刺激对照
"""

import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from neural_exploration.tools.hh_spec import CM, DT, EK, EL, ENA, GK, GL, GNA, T_TOTAL, V0, steady_state  # noqa: E402

REPORTS_DIR = os.path.join(ROOT, "neural_exploration", "reports", "neuro")

# --- 闭环参数 ---
TAU_SYN = 2.0        # ms，突触时间常数
G_SYN = 1.0          # mS/cm²，突触电导上限
EREV_EXC = 0.0       # mV，兴奋性突触反转电位
W_SYN = 0.15         # 突触权重（每次发放 s 增量）
TAU_MUSCLE = 20.0    # ms，虚拟肌肉收缩衰减时间常数
T_STIM_START_LOOP = 5.0   # ms
T_STIM_END_LOOP = 55.0    # ms（刺激持续 50ms）


@dataclass
class SmokeResult:
    motor_spikes: int = 0
    muscle_contraction: float = 0.0
    muscle_max: float = 0.0
    sensory_spikes: int = 0
    t: np.ndarray = field(default_factory=lambda: np.array([]))
    v_sensory: np.ndarray = field(default_factory=lambda: np.array([]))
    v_motor: np.ndarray = field(default_factory=lambda: np.array([]))
    contraction: np.ndarray = field(default_factory=lambda: np.array([]))

    def __eq__(self, other):
        """确定性验证用：数值逐位比较（含数组）。"""
        if not isinstance(other, SmokeResult):
            return NotImplemented
        return (
            self.motor_spikes == other.motor_spikes
            and self.muscle_contraction == other.muscle_contraction
            and self.muscle_max == other.muscle_max
            and self.sensory_spikes == other.sensory_spikes
            and np.array_equal(self.t, other.t)
            and np.array_equal(self.v_sensory, other.v_sensory)
            and np.array_equal(self.v_motor, other.v_motor)
            and np.array_equal(self.contraction, other.contraction)
        )


def run_smoke_loop(stimulus_amp=10.0, plot=False):
    """运行最小闭环。stimulus_amp：刺激幅度 µA/cm²；0 为无刺激对照。"""
    try:
        from brian2 import (NeuronGroup, StateMonitor, Synapses, TimedArray,
                            amp, cm, defaultclock, meter, ms, mS, mV, run, start_scope, uF, volt, Hz)
    except ImportError as e:
        raise RuntimeError("需要主线引擎 Brian2（pip install brian2）") from e

    start_scope()
    from brian2 import defaultclock
    defaultclock.dt = DT * ms

    def _hh_eqs(stim_expr, with_syn=False):
        syn_term = f" - ({G_SYN}*mS/cm2)*s_syn*(v-({EREV_EXC}*mV))" if with_syn else ""
        syn_decl = f"\n        ds_syn/dt = -s_syn/({TAU_SYN}*ms) : 1" if with_syn else ""
        return f"""
        dv/dt = ({stim_expr} - ({GNA}*mS/cm2)*m**3*h*(v-({ENA}*mV)) - ({GK}*mS/cm2)*n**4*(v-({EK}*mV)) - ({GL}*mS/cm2)*(v-({EL}*mV)){syn_term}) / ({CM}*uF/cm2) : volt
        dm/dt = alpham*(1-m)-betam*m : 1
        dh/dt = alphah*(1-h)-betah*h : 1
        dn/dt = alphan*(1-n)-betan*n : 1
        alpham = (0.1/mV)*(v+40*mV)/(1-exp(-(v+40*mV)/(10*mV)))/ms : Hz
        betam = 4*exp(-(v+65*mV)/(18*mV))/ms : Hz
        alphah = 0.07*exp(-(v+65*mV)/(20*mV))/ms : Hz
        betah = 1/(1+exp(-(v+35*mV)/(10*mV)))/ms : Hz
        alphan = (0.01/mV)*(v+55*mV)/(1-exp(-(v+55*mV)/(10*mV)))/ms : Hz
        betan = 0.125*exp(-(v+65*mV)/(80*mV))/ms : Hz
        Iinj : amp/meter**2{syn_decl}
        """
    n_steps = int(round(T_TOTAL / DT))
    i_am2 = stimulus_amp * 1e-6 * amp / cm ** 2
    t_stim = np.arange(n_steps) * DT
    i_values = np.where((t_stim >= T_STIM_START_LOOP) & (t_stim < T_STIM_END_LOOP), 1.0, 0.0) * i_am2
    stim = TimedArray(i_values, dt=DT * ms)

    sensory = NeuronGroup(1, _hh_eqs("stim(t)"), method="rk4",
                          threshold="v > -20*mV", refractory=2 * ms)
    motor = NeuronGroup(1, _hh_eqs("Iinj", with_syn=True), method="rk4",
                        threshold="v > -20*mV", refractory=2 * ms)
    m0, h0, n0 = steady_state(V0)
    for g in (sensory, motor):
        g.v = V0 * mV
        g.m = m0
        g.h = h0
        g.n = n0
        g.Iinj = 0.0 * amp / meter ** 2

    # 兴奋性化学突触：感觉 → 运动（EPSP；on_pre 直接累加到运动神经元 s_syn，
    # 衰减由运动神经元自身方程处理——避免 Brian2 突触变量重名限制）
    syn = Synapses(sensory, motor, on_pre="s_syn += W_SYN")
    syn.connect()
    syn.delay = 0.5 * ms

    # 虚拟肌肉：运动神经元发放 → 收缩量（积分器，带指数衰减）
    muscle_eqs = f"dcontraction/dt = -contraction/({TAU_MUSCLE}*ms) : 1"
    muscle = NeuronGroup(1, muscle_eqs, method="euler")
    muscle.contraction = 0.0
    m_syn = Synapses(motor, muscle, on_pre="contraction += 1.0")
    m_syn.connect()

    mon_s = StateMonitor(sensory, "v", record=True, dt=DT * ms)
    mon_m = StateMonitor(motor, "v", record=True, dt=DT * ms)
    mon_c = StateMonitor(muscle, "contraction", record=True, dt=DT * ms)

    run(T_TOTAL * ms)

    v_s = np.array(mon_s.v[0] / mV)
    v_m = np.array(mon_m.v[0] / mV)
    contraction = np.array(mon_c.contraction[0])
    t = np.array(mon_s.t / ms)

    from neural_exploration.tools.metrics import spike_count
    result = SmokeResult(
        motor_spikes=spike_count(v_m),
        muscle_contraction=float(contraction[-1]),
        muscle_max=float(contraction.max()),
        sensory_spikes=spike_count(v_s),
        t=t, v_sensory=v_s, v_motor=v_m, contraction=contraction,
    )

    if plot:
        plot_smoke(result, os.path.join(REPORTS_DIR, "m0_smoke.png"))
    return result


def plot_smoke(result, out_png=None):
    """刺激/响应叠加图：感觉 V、运动 V、肌肉收缩。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_png = out_png or os.path.join(REPORTS_DIR, "m0_smoke.png")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(result.t, result.v_sensory, lw=1.0, color="#1f77b4")
    axes[0].set_ylabel("sensory V (mV)")
    axes[0].set_title(f"M0 smoke loop: stimulus->sensory->synapse->motor->muscle "
                      f"(motor_spikes={result.motor_spikes})")
    axes[1].plot(result.t, result.v_motor, lw=1.0, color="#d62728")
    axes[1].axhline(-20.0, color="gray", ls="--", lw=0.8)
    axes[1].set_ylabel("motor V (mV)")
    axes[2].plot(result.t, result.contraction, lw=1.0, color="#2ca02c")
    axes[2].set_ylabel("muscle contraction")
    axes[2].set_xlabel("t (ms)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", type=float, default=10.0)
    ap.add_argument("--noplot", action="store_true")
    args = ap.parse_args()
    r = run_smoke_loop(stimulus_amp=args.amp, plot=not args.noplot)
    print(f"sensory_spikes={r.sensory_spikes}  motor_spikes={r.motor_spikes}  "
          f"muscle_contraction={r.muscle_contraction:.4f}  muscle_max={r.muscle_max:.4f}")
    if not args.noplot:
        print(f"图已生成: {os.path.join(REPORTS_DIR, 'm0_smoke.png')}")
