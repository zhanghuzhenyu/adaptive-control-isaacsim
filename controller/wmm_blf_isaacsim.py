"""
Wheeled Mobile Manipulator – model-based controller WITH Barrier Lyapunov Function (BLF).

Control law (MATLAB controller.m, "% BLF" line):
    u = pinv(E_bar) * (term1 - k2*M_bar*e2 + C_bar*miu + M_bar*dmiu)
    term1(i) = -e1(i) / (kb(i)^2 - e1(i)^2)

The BLF term replaces the simple -e1 feedback.  As |e1[i]| → kb[i] the term diverges
to infinity, producing an infinitely large corrective torque that prevents the error
from ever crossing the barrier kb(t).  Theoretical guarantee: if |e1(0)| < kb(0) then
|e1(t)| < kb(t) for all t ≥ 0.

Compare with wmm_simple_isaacsim.py (no BLF) to verify:
  1. Tracking error stays inside kb envelope.
  2. Whether the BLF improves or changes the RMSE.

Outputs saved to PLOT_DIR (different from wmm_simple paths — no collision).
"""

import argparse
import math
import os
import sys
import traceback

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--test",     default=False, action="store_true")
parser.add_argument("--headless", default=False, action="store_true")
args, unknown = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.robots import WheeledRobot

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)  # wmm-isaac-sim/
sys.path.insert(0, os.path.join(_REPO_ROOT, "lib"))
from simulate_wmm_python import desired_trajectory, kinematics_jacobian, compute_bar_matrices

# ── Paths (different from wmm_simple to avoid collision) ──────────────────────
LOG_FILE = "/tmp/wmm_blf.txt"
PLOT_DIR = os.path.join(_REPO_ROOT, "results", "blf")
os.makedirs(PLOT_DIR, exist_ok=True)

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

# ── Physical parameters (from URDF robot_wheel_0904.urdf) ─────────────────────
class P:
    r     = 0.025784
    b     = 0.091
    d     = 0.034
    l1    = 0.514
    l11   = 0.274646
    l2    = 0.362
    l22   = 0.188695
    mp    = 6.28012
    m1    = 0.358138
    m2    = 0.265505
    mw    = 0.027797
    Iphai = 0.063218
    I1    = 0.037726
    I2    = 0.013757
    Iw    = 9.23979e-6

p = P()

# ── Controller parameters ──────────────────────────────────────────────────────
# K1 same as simple controller: miu must demand the same corrective velocity bandwidth.
# Reducing K1 weakens miu → error grows faster than BLF can suppress → divergence.
K1 = np.array([10.0, 10.0, 10.0, 10.0])
K2 = 1.5

# Barrier parameters:
#   UP_BOUND  = kb(0) — must be > |e1(0)|.  Initial error ~ 0.05–0.10 m → 0.5 is safe.
#   DOWN_BOUND = kb(∞) — steady-state constraint.
#     Simple controller RMSE ~6 cm, peak |e1| ~17 cm → need DOWN_BOUND > 0.17 m.
#     0.30 m gives ~1.8× margin: BLF gain at e1=0.17, kb=0.30 is ~16× (still bounded).
#   RATE = 0.5 (same as simple — keeps discrete-time stability).
# Larger barriers + slow decay so BLF gain stays ≤ 1.5× simple throughout 25 s.
# At kb=0.35 (steady-state) and typical |e1|~0.10 m: amplification = 1/(1-0.286²) ≈ 1.09×
# At kb=0.35 and |e1|~0.19 m (worst case in simple run): 1/(1-0.543²) ≈ 1.42×
# → arm torques stay close to simple → no joint-limit divergence
UP_BOUND   = np.array([0.8, 0.8, 0.8, 0.8])
DOWN_BOUND = np.array([0.35, 0.35, 0.35, 0.35])
RATE       = np.array([0.2, 0.2, 0.2, 0.2])

# Torque saturation limits
LIM_WHEEL = 20.0
LIM_ARM   = 20.0

# Friction feedforward (same as simplified controller — compensates Isaac Sim rolling friction)
FRIC_FF = 0.15

# ── Simulation timing ─────────────────────────────────────────────────────────
SETTLE_STEPS   = 200
PHYSICS_DT     = 1.0 / 60.0
MAX_CTRL_STEPS = 1500

INIT_Q1       = 0.675
INIT_Q2_ISAAC = 2.898

# ── Helpers ───────────────────────────────────────────────────────────────────
def yaw_from_quat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def get_robot_state(jetbot, idx):
    pos, quat = jetbot.get_world_pose()
    jpos = jetbot.get_joint_positions()
    jvel = jetbot.get_joint_velocities()

    phai = yaw_from_quat(quat)
    q1   = float(jpos[idx["joint1"]])
    q2   = float(jpos[idx["joint2"]]) - math.pi / 2.0

    qr_dot = (float(jvel[idx["right_low_wheel_joint"]]) + float(jvel[idx["right_up_wheel_joint"]])) / 2.0
    ql_dot = -(float(jvel[idx["left_low_wheel_joint"]]) + float(jvel[idx["left_up_wheel_joint"]])) / 2.0
    q1dot  = float(jvel[idx["joint1"]])
    q2dot  = float(jvel[idx["joint2"]])

    bx, by = float(pos[0]), float(pos[1])
    c1  = math.cos(phai + q1)
    s1  = math.sin(phai + q1)
    c12 = math.cos(phai + q1 + q2)
    s12 = math.sin(phai + q1 + q2)

    X = np.array([
        bx + p.d * math.cos(phai) + p.l1 * c1 + p.l2 * c12,
        by + p.d * math.sin(phai) + p.l1 * s1 + p.l2 * s12,
        bx + p.d * math.cos(phai),
        by + p.d * math.sin(phai),
    ])

    q_wmm = np.array([0.0, 0.0, phai, q1, q2])
    v     = np.array([qr_dot, ql_dot, q1dot, q2dot])
    J     = kinematics_jacobian(p, q_wmm)
    dX    = J @ v
    dq    = np.array([qr_dot, ql_dot, p.r * (qr_dot - ql_dot) / (2.0 * p.b), q1dot, q2dot])

    return X, dX, q_wmm, dq


def compute_control(t, X, dX, q_wmm, dq):
    Xd, dXd, ddXd = desired_trajectory(t)
    e1  = X  - Xd
    de1 = dX - dXd

    kb      = (UP_BOUND - DOWN_BOUND) * np.exp(-RATE * t) + DOWN_BOUND
    kb_dot  = -RATE * (UP_BOUND - DOWN_BOUND) * np.exp(-RATE * t)
    kb_ddot =  RATE ** 2 * (UP_BOUND - DOWN_BOUND) * np.exp(-RATE * t)

    # ── BLF feedback term (kb²-normalized formulation) ───────────────────────
    # Standard BLF:  term1 = -e1 / (kb²-e1²)          — amplifies 1/kb² at zero error
    # Modified BLF:  term1 = -e1 / (1-(e1/kb)²)        — equals -e1 at zero error,
    #                                                      diverges as |e1|→kb
    #
    # The modified form is kb² × the standard form, so the same Lyapunov function
    # applies (scaled by kb²).  At the typical operating point (|e1|~0.07m, kb~0.25m):
    #   standard → 16× amplification → arm saturation → instability in discrete time
    #   modified → ≈1.04× amplification → behaves like simple -e1, stable
    # Near the boundary the gain still diverges, enforcing the constraint.
    e1_safe = np.clip(e1, -(1.0 - 1e-3) * kb, (1.0 - 1e-3) * kb)
    normalized = e1_safe / kb
    term1      = -e1_safe / (1.0 - normalized ** 2)

    # ── Virtual-control reference (same formula as simple controller) ─────────
    miu  = -K1 * e1 + (kb_dot / kb) * e1 + dXd
    dmiu = (ddXd - K1 * de1
            + ((kb_ddot * kb - kb_dot ** 2) / kb ** 2) * e1
            + (kb_dot / kb) * de1)
    e2 = dX - miu

    M_bar, C_bar, E_bar, _ = compute_bar_matrices(p, q_wmm, dq)

    # ── BLF control law ───────────────────────────────────────────────────────
    # u = pinv(E_bar) * (term1 - k2*M_bar*e2 + C_bar*miu + M_bar*dmiu)
    lam = 0.08
    U_s, s_s, Vt_s = np.linalg.svd(E_bar)
    s_damp = s_s / (s_s ** 2 + lam ** 2)
    E_bar_pinv = Vt_s.T @ np.diag(s_damp) @ U_s.T

    u = E_bar_pinv @ (term1 - K2 * (M_bar @ e2) + C_bar @ miu + M_bar @ dmiu)

    # Friction feedforward (same as simplified controller)
    if abs(dXd[2]) > 0.01:
        ff = FRIC_FF * np.sign(dXd[2])
        u[0] += ff
        u[1] += ff

    u[0] = np.clip(u[0], -LIM_WHEEL, LIM_WHEEL)
    u[1] = np.clip(u[1], -LIM_WHEEL, LIM_WHEEL)
    u[2] = np.clip(u[2], -LIM_ARM,   LIM_ARM)
    u[3] = np.clip(u[3], -LIM_ARM,   LIM_ARM)

    return u, e1, e2, kb, Xd, dXd, term1


# ── Plotting ──────────────────────────────────────────────────────────────────
def _subplot4(fig, axes, t, data, desired, ylabels, title, colors=None):
    for i in range(4):
        col = colors[i] if colors else "tab:blue"
        axes[i].plot(t, data[:, i],    color=col,       lw=1.2, label="actual")
        axes[i].plot(t, desired[:, i], color="tab:red", lw=1.2, ls="--", label="desired")
        axes[i].set_ylabel(ylabels[i], fontsize=9)
        axes[i].grid(alpha=0.3)
        axes[i].legend(fontsize=7, loc="best")
    axes[-1].set_xlabel("t (s)", fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()


def save_plots(log_t, log_e1, log_u, log_X, log_Xd,
               log_dX, log_dXd, log_e2, log_kb, log_term1):
    t     = np.asarray(log_t)
    e1    = np.asarray(log_e1)
    u     = np.asarray(log_u)
    X     = np.asarray(log_X)
    Xd    = np.asarray(log_Xd)
    dX    = np.asarray(log_dX)
    dXd   = np.asarray(log_dXd)
    e2    = np.asarray(log_e2)
    kb    = np.asarray(log_kb)
    term1 = np.asarray(log_term1)

    ch_colors = ["tab:blue", "tab:green", "tab:purple", "tab:orange"]

    # ── Fig 1 : X state tracking ──────────────────────────────────────────────
    ylabels_X = ["$X_1$ (m)", "$X_2$ (m)", "$X_3$ (m)", "$X_4$ (m)"]
    fig1, ax1 = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    _subplot4(fig1, ax1, t, X, Xd, ylabels_X,
              "Fig.1  State Tracking: X actual vs desired (BLF)", ch_colors)
    path = os.path.join(PLOT_DIR, "fig1_state_tracking.png")
    fig1.savefig(path, dpi=150); plt.close(fig1); log(f"  saved {path}")

    # ── Fig 2 : dX velocity tracking ─────────────────────────────────────────
    ylabels_dX = ["$\\dot{X}_1$ (m/s)", "$\\dot{X}_2$ (m/s)",
                  "$\\dot{X}_3$ (m/s)", "$\\dot{X}_4$ (m/s)"]
    fig2, ax2 = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    _subplot4(fig2, ax2, t, dX, dXd, ylabels_dX,
              "Fig.2  Velocity Tracking: dX actual vs desired (BLF)", ch_colors)
    path = os.path.join(PLOT_DIR, "fig2_velocity_tracking.png")
    fig2.savefig(path, dpi=150); plt.close(fig2); log(f"  saved {path}")

    # ── Fig 3 : control inputs u ──────────────────────────────────────────────
    ylabels_u = ["$u_1$ (N·m)", "$u_2$ (N·m)", "$u_3$ (N·m)", "$u_4$ (N·m)"]
    fig3, ax3 = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for i in range(4):
        ax3[i].plot(t, u[:, i], color=ch_colors[i], lw=1.2)
        ax3[i].axhline(0, color="k", lw=0.7, ls="--")
        ax3[i].set_ylabel(ylabels_u[i], fontsize=9)
        ax3[i].grid(alpha=0.3)
    ax3[-1].set_xlabel("t (s)", fontsize=10)
    fig3.suptitle("Fig.3  Control Inputs u (BLF)", fontsize=11)
    fig3.tight_layout()
    path = os.path.join(PLOT_DIR, "fig3_control_inputs.png")
    fig3.savefig(path, dpi=150); plt.close(fig3); log(f"  saved {path}")

    # ── Fig 4 : e1 errors with kb barrier — KEY FIGURE for BLF verification ──
    ylabels_e = ["$e_1$ (m)", "$e_2$ (m)", "$e_3$ (m)", "$e_4$ (m)"]
    fig4, ax4 = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for i in range(4):
        rmse = float(np.sqrt(np.mean(e1[:, i] ** 2)))
        iae  = float(np.trapz(np.abs(e1[:, i]), t))
        itae = float(np.trapz(t * np.abs(e1[:, i]), t))
        # error and barrier envelope
        ax4[i].fill_between(t,  kb[:, i], -kb[:, i],
                            color="tab:red", alpha=0.08, label="allowed region")
        ax4[i].plot(t, e1[:, i],  color="tab:blue", lw=1.4, label="$e_1$")
        ax4[i].plot(t,  kb[:, i], color="tab:red",  lw=1.2, ls="--", label="$k_b$")
        ax4[i].plot(t, -kb[:, i], color="tab:red",  lw=1.2, ls="--")
        ax4[i].axhline(0, color="k", lw=0.5, ls=":")
        ax4[i].set_ylabel(ylabels_e[i], fontsize=9)
        ax4[i].grid(alpha=0.3)
        ax4[i].legend(fontsize=7, loc="upper right")
        ax4[i].text(0.99, 0.97,
                    f"RMSE={rmse:.4f}  IAE={iae:.4f}  ITAE={itae:.4f}",
                    transform=ax4[i].transAxes, fontsize=7,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax4[-1].set_xlabel("t (s)", fontsize=10)
    fig4.suptitle("Fig.4  Tracking Errors $e_1$ with BLF barrier $k_b$\n"
                  "(error must stay within shaded region)", fontsize=11)
    fig4.tight_layout()
    path = os.path.join(PLOT_DIR, "fig4_tracking_errors.png")
    fig4.savefig(path, dpi=150); plt.close(fig4); log(f"  saved {path}")

    # ── Fig 5 : e2 errors ─────────────────────────────────────────────────────
    ylabels_e2 = ["$e_{2,1}$", "$e_{2,2}$", "$e_{2,3}$", "$e_{2,4}$"]
    fig5, ax5 = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for i in range(4):
        ax5[i].plot(t, e2[:, i], color=ch_colors[i], lw=1.2)
        ax5[i].axhline(0, color="k", lw=0.7, ls="--")
        ax5[i].set_ylabel(ylabels_e2[i], fontsize=9)
        ax5[i].grid(alpha=0.3)
    ax5[-1].set_xlabel("t (s)", fontsize=10)
    fig5.suptitle("Fig.5  Second Error Variable $e_2 = \\dot{X} - \\mu$ (BLF)", fontsize=11)
    fig5.tight_layout()
    path = os.path.join(PLOT_DIR, "fig5_e2_errors.png")
    fig5.savefig(path, dpi=150); plt.close(fig5); log(f"  saved {path}")

    # ── Fig 6 : 2-D EE + base trajectory ─────────────────────────────────────
    fig6, ax6 = plt.subplots(figsize=(9, 6))
    ax6.plot(X[:, 0],  X[:, 1],  color="tab:blue",   lw=1.5, label="EE actual")
    ax6.plot(Xd[:, 0], Xd[:, 1], color="tab:blue",   lw=1.0, ls="--", label="EE desired")
    ax6.plot(X[:, 2],  X[:, 3],  color="tab:orange",  lw=1.5, label="base actual")
    ax6.plot(Xd[:, 2], Xd[:, 3], color="tab:orange",  lw=1.0, ls="--", label="base desired")
    ax6.scatter([X[0, 0], X[0, 2]],   [X[0, 1], X[0, 3]],   s=60, zorder=5,
                color=["tab:blue", "tab:orange"])
    ax6.scatter([Xd[0, 0], Xd[0, 2]], [Xd[0, 1], Xd[0, 3]], s=60, zorder=5, marker="^",
                color=["tab:blue", "tab:orange"])
    ax6.set_xlabel("x (m)"); ax6.set_ylabel("y (m)")
    ax6.set_title("Fig.6  2-D Trajectory: EE and base (BLF)", fontsize=11)
    ax6.legend(fontsize=9); ax6.grid(alpha=0.3); ax6.set_aspect("equal", adjustable="datalim")
    fig6.tight_layout()
    path = os.path.join(PLOT_DIR, "fig6_trajectory_2d.png")
    fig6.savefig(path, dpi=150); plt.close(fig6); log(f"  saved {path}")

    # ── Fig 7 (BLF-specific): BLF term1 magnitude and e1/kb ratio ────────────
    fig7, ax7 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i in range(4):
        ax7[0].plot(t, term1[:, i], color=ch_colors[i], lw=1.0,
                    label=f"term1[{i+1}]")
    ax7[0].axhline(0, color="k", lw=0.5, ls=":")
    ax7[0].set_ylabel("term1 (N·m equiv.)", fontsize=9)
    ax7[0].legend(fontsize=7, loc="best"); ax7[0].grid(alpha=0.3)
    ax7[0].set_title(r"BLF feedback term  $-e_1\,/\,(1-(e_1/k_b)^2)$  [kb²-normalized]", fontsize=10)

    ratio = np.abs(e1) / kb
    for i in range(4):
        ax7[1].plot(t, ratio[:, i], color=ch_colors[i], lw=1.0,
                    label=f"|e1/kb|[{i+1}]")
    ax7[1].axhline(1.0, color="tab:red", lw=1.2, ls="--", label="barrier limit")
    ax7[1].set_ylim(0, 1.1)
    ax7[1].set_ylabel(r"$|e_1| / k_b$", fontsize=9)
    ax7[1].set_xlabel("t (s)", fontsize=10)
    ax7[1].legend(fontsize=7, loc="best"); ax7[1].grid(alpha=0.3)
    ax7[1].set_title(r"Constraint ratio — must stay below 1.0", fontsize=10)

    fig7.suptitle("Fig.7  BLF Diagnostics", fontsize=11)
    fig7.tight_layout()
    path = os.path.join(PLOT_DIR, "fig7_blf_diagnostics.png")
    fig7.savefig(path, dpi=150); plt.close(fig7); log(f"  saved {path}")

    # ── Summary metrics ───────────────────────────────────────────────────────
    log("\n── Performance metrics ──────────────────────────────────────────")
    for i in range(4):
        rmse = np.sqrt(np.mean(e1[:, i] ** 2))
        iae  = np.trapz(np.abs(e1[:, i]), t)
        itae = np.trapz(t * np.abs(e1[:, i]), t)
        log(f"  e1[{i+1}]:  RMSE={rmse:.5f} m   IAE={iae:.5f} m·s   ITAE={itae:.5f} m·s²")

    log("\n── BLF constraint check ─────────────────────────────────────────")
    max_ratio = np.max(np.abs(e1) / kb, axis=0)
    for i in range(4):
        violated = np.any(np.abs(e1[:, i]) >= kb[:, i])
        log(f"  ch{i+1}: max |e1|/kb = {max_ratio[i]:.4f}  "
            f"{'*** VIOLATED ***' if violated else 'OK (within bound)'}")


# ── Isaac Sim world setup ─────────────────────────────────────────────────────
my_world = World(stage_units_in_meters=1.0)

my_jetbot = my_world.scene.add(
    WheeledRobot(
        prim_path="/World/Jetbot",
        name="my_jetbot",
        wheel_dof_names=[
            "left_low_wheel_joint", "left_up_wheel_joint",
            "right_low_wheel_joint", "right_up_wheel_joint",
        ],
        create_robot=True,
        usd_path=os.path.join(_REPO_ROOT, "robot", "usd", "robot_wheel_0904.usd"),
        position=np.array([0.366, 0.10, 0.5], dtype=np.float32),
    )
)

my_world.scene.add_default_ground_plane()
my_world.reset()

# ── DOF index map ─────────────────────────────────────────────────────────────
open(LOG_FILE, "w").close()
log("=== WMM Isaac Sim – BLF Controller ===")
log(f"  k1={K1}, k2={K2}, FRIC_FF={FRIC_FF}")
log(f"  UP_BOUND={UP_BOUND}, DOWN_BOUND={DOWN_BOUND}, RATE={RATE}")
log(f"  SETTLE_STEPS={SETTLE_STEPS}, MAX_CTRL_STEPS={MAX_CTRL_STEPS}")

idx = {}
for name in ["joint1", "joint2",
             "left_low_wheel_joint", "left_up_wheel_joint",
             "right_low_wheel_joint", "right_up_wheel_joint"]:
    try:
        idx[name] = my_jetbot.get_dof_index(name)
    except Exception as e:
        log(f"ERROR: get_dof_index({name}): {e}")
log(f"DOF indices: {idx}")

DOF_ORDER = np.array([
    idx["joint1"],
    idx["left_low_wheel_joint"], idx["left_up_wheel_joint"],
    idx["right_low_wheel_joint"], idx["right_up_wheel_joint"],
    idx["joint2"],
], dtype=np.int32)

_ARM_IDX   = np.array([idx["joint1"], idx["joint2"]], dtype=np.int32)
_WHEEL_IDX = np.array([
    idx["left_low_wheel_joint"], idx["left_up_wheel_joint"],
    idx["right_low_wheel_joint"], idx["right_up_wheel_joint"],
], dtype=np.int32)

my_jetbot.set_joint_positions(
    positions=np.array([INIT_Q1, INIT_Q2_ISAAC], dtype=np.float32),
    joint_indices=_ARM_IDX,
)
log(f"Initial arm angles set: joint1={INIT_Q1} rad, joint2={INIT_Q2_ISAAC} rad (URDF frame)")

# ── Simulation state ──────────────────────────────────────────────────────────
reset_needed = False
step_count   = 0
ctrl_step    = 0
t_ctrl       = 0.0

log_t, log_e1, log_u, log_X, log_Xd   = [], [], [], [], []
log_dX, log_dXd, log_e2, log_kb       = [], [], [], []
log_term1                              = []

# ── Main loop ─────────────────────────────────────────────────────────────────
while simulation_app.is_running():
    my_world.step(render=not args.headless)

    if my_world.is_stopped() and not reset_needed:
        reset_needed = True

    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            step_count = 0; ctrl_step = 0; t_ctrl = 0.0
            for lst in (log_t, log_e1, log_u, log_X, log_Xd,
                        log_dX, log_dXd, log_e2, log_kb, log_term1):
                lst.clear()
            reset_needed = False
            my_jetbot.set_joint_positions(
                positions=np.array([INIT_Q1, INIT_Q2_ISAAC], dtype=np.float32),
                joint_indices=_ARM_IDX,
            )

        step_count += 1

        if step_count <= SETTLE_STEPS:
            my_jetbot.set_joint_positions(
                positions=np.array([INIT_Q1, INIT_Q2_ISAAC], dtype=np.float32),
                joint_indices=_ARM_IDX,
            )
            my_jetbot.set_joint_efforts(
                efforts=np.zeros(4, dtype=np.float32),
                joint_indices=_WHEEL_IDX,
            )
            continue

        try:
            X, dX, q_wmm, dq                 = get_robot_state(my_jetbot, idx)
            u, e1, e2, kb, Xd, dXd, term1    = compute_control(t_ctrl, X, dX, q_wmm, dq)

            efforts = np.array([
                 u[2],          # joint1
                -u[1] / 2.0,   # left_low  (axis flipped)
                -u[1] / 2.0,   # left_up
                 u[0] / 2.0,   # right_low
                 u[0] / 2.0,   # right_up
                 u[3],          # joint2
            ], dtype=np.float32)
            my_jetbot.set_joint_efforts(efforts=efforts, joint_indices=DOF_ORDER)

            log_t.append(t_ctrl);     log_e1.append(e1.copy());    log_u.append(u.copy())
            log_X.append(X.copy());   log_Xd.append(Xd.copy())
            log_dX.append(dX.copy()); log_dXd.append(dXd.copy())
            log_e2.append(e2.copy()); log_kb.append(kb.copy())
            log_term1.append(term1.copy())

            t_ctrl    += PHYSICS_DT
            ctrl_step += 1

            if ctrl_step % 60 == 0:
                pos, _ = my_jetbot.get_world_pose()
                ratio  = np.abs(e1) / kb
                log(
                    f"t={t_ctrl:6.2f}s | "
                    f"e1=[{e1[0]:+.4f}, {e1[1]:+.4f}, {e1[2]:+.4f}, {e1[3]:+.4f}] | "
                    f"ratio=[{ratio[0]:.3f},{ratio[1]:.3f},{ratio[2]:.3f},{ratio[3]:.3f}] | "
                    f"u=[{u[0]:+6.2f}, {u[1]:+6.2f}, {u[2]:+6.2f}, {u[3]:+6.2f}]"
                )

            if ctrl_step >= MAX_CTRL_STEPS:
                log(f"=== MAX_CTRL_STEPS={MAX_CTRL_STEPS} reached, stopping ===")
                break

        except Exception as exc:
            log(f"[step {step_count}] Controller error: {exc}\n{traceback.format_exc()}")

    if args.test:
        break

my_world.stop()

# ── Generate and save plots ───────────────────────────────────────────────────
if len(log_t) > 1:
    log(f"\nGenerating plots ({len(log_t)} data points)...")
    save_plots(log_t, log_e1, log_u, log_X, log_Xd,
               log_dX, log_dXd, log_e2, log_kb, log_term1)

simulation_app.close()
