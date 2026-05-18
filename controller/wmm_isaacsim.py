"""
Wheeled Mobile Manipulator – adaptive controller in Isaac Sim.

Controller:  adaptive_ctrl/control_script/simulate_wmm_python.py
Robot USD:   adaptive_ctrl/robot_wheel/urdf/robot_wheel_0904/robot_wheel_0904.usd

DOF order (confirmed):  joint1(0)  left_low(1)  left_up(2)  right_low(3)  right_up(4)  joint2(5)
WMM control vector u:   [u_r, u_l, u_q1, u_q2]   (right-wheel, left-wheel, joint1, joint2)

Torque mapping  (S_q.T @ B_actual = I_4, so u = physical joint torques):
  joint1    (DOF 0) =  u[2]
  left_low  (DOF 1) = -u[1] / 2   # left wheel axis flipped vs. WMM convention
  left_up   (DOF 2) = -u[1] / 2
  right_low (DOF 3) = +u[0] / 2
  right_up  (DOF 4) = +u[0] / 2
  joint2    (DOF 5) =  u[3]

Initial conditions chosen so |e1(0)| < up_bound = 0.5 for all components:
  robot base  (0.0, 0.305, 0.5)  →  drops to ground during settle phase
  q1 = pi/3,  q2 = -pi/3
  → X_init ≈ [0.65, 0.75, 0.034, 0.305],  Xd(0) = [0.65, 0.75, 0.40, 0.10]
  → e1(0) ≈ [0, 0, -0.37, 0.21]  (all < 0.5 ✓)
"""

import argparse
import math
import sys
import traceback

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true")
args, unknown = parser.parse_known_args()

simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.robots import WheeledRobot

# ── Import adaptive controller ────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)  # wmm-isaac-sim/
sys.path.insert(0, os.path.join(_REPO_ROOT, "lib"))
from simulate_wmm_python import (
    WMMParams,
    desired_trajectory,
    rbf_scalar,
    eps_clip,
    kinematics_jacobian,
    compute_bar_matrices,
)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = "/tmp/wmm_isaacsim.txt"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

# ── Robot parameters (all values from URDF) ───────────────────────────────────
class RobotParams(WMMParams):
    # Kinematics
    r    = 0.0254      # wheel radius (m), from mesh extent
    b    = 0.091       # half-wheelbase (m), wheel joint y-offset
    d    = 0.034       # joint1 x-offset from base_link origin (m)
    l1   = 0.514       # arm link-1 length: joint2 x-offset from joint1 (m)
    l11  = 0.274646    # link-1 CoM x-distance from joint1 (m), from URDF
    l2   = 0.362       # arm link-2 length (m), from original paper (no URDF tip)
    l22  = 0.189       # link-2 CoM distance from joint2 (m), |y| from URDF CoM
    # Dynamics — from URDF masses + Izz inertias (parallel-axis where needed)
    mp    = 6.28012    # base_link mass (kg)
    m1    = 0.358138   # joint1_link mass (kg)
    m2    = 0.265505   # joint2_link mass (kg)
    mw    = 0.027797   # single wheel mass (kg)
    Iphai = 0.06322    # base_link Izz about CoM (kg·m²)
    I1    = 0.03760    # joint1_link Izz about joint1 axis = 0.01071 + m1*l11² (kg·m²)
    I2    = 0.01379    # joint2_link Izz about joint2 axis = 0.00430 + m2*l22² (kg·m²)
    Iw    = 9.24e-6    # wheel Izz about spin axis (kg·m²)
    # Controller gains — tuned for physical 60 Hz loop (original gains for 100 Hz ODE)
    k1   = np.array([5.0, 5.0, 5.0, 5.0])   # position error gain (reduced from 50)
    k2   = 1.0                                 # velocity error gain  (reduced from 5)
    # Barrier: up_bound larger, rate much smaller → barrier shrinks over ~20 s, not 1 s
    up_bound   = np.array([1.0, 1.0, 1.0, 1.0])
    down_bound = np.array([0.1, 0.1, 0.1, 0.1])
    rate       = np.array([0.15, 0.15, 0.15, 0.15])

p = RobotParams()
nc = p.cij.size
na = p.c_a.size

SETTLE_STEPS     = 200          # steps for robot to drop and settle (~3.3 s at 60 Hz)
PHYSICS_DT       = 1.0 / 60.0  # s per step
MAX_CTRL_STEPS   = 1500        # ~25 s of control
TORQUE_LIM_WHEEL = 20.0        # N·m saturation per wheel side
TORQUE_LIM_ARM   = 10.0        # N·m saturation per arm joint

# Initial arm angles chosen so X_init ≈ Xd(0) = [0.65, 0.75, 0.40, 0.10].
# Verified: with base at (0.366, 0.10) and phai=0:
#   q2_wmm = INIT_Q2_ISAAC - pi/2 = 2.898 - 1.5708 = 1.327 rad
#   X_init ≈ [0.6499, 0.7500, 0.4000, 0.1000]  → e1(0) ≈ 0  → barrier valid.
# The arm joints are horizontal-plane rotations, so gravity does not disturb
# them during the settle phase (zero torques = zero drift).
INIT_Q1       = 0.675
INIT_Q2_ISAAC = 2.898

# ── Utility: yaw angle from quaternion [w, x, y, z] ──────────────────────────
def yaw_from_quat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

# ── State extraction from Isaac Sim → WMM coordinates ────────────────────────
def get_robot_state(jetbot):
    pos, quat = jetbot.get_world_pose()
    jpos      = jetbot.get_joint_positions()
    jvel      = jetbot.get_joint_velocities()

    phai = yaw_from_quat(quat)
    q1   = float(jpos[idx["joint1"]])
    # WMM: q2=0 → link2 forward. URDF: q2_isaac=0 → link2 in -y direction.
    # Correction: q2_wmm = q2_isaac - pi/2
    q2   = float(jpos[idx["joint2"]]) - math.pi / 2.0

    # Wheel velocities
    # Right wheels: positive vel → forward, matches WMM qr_dot > 0
    qr_dot = (float(jvel[idx["right_low_wheel_joint"]]) + float(jvel[idx["right_up_wheel_joint"]])) / 2.0
    # Left wheels: positive vel → backward; negate for WMM ql_dot
    ql_dot = -(float(jvel[idx["left_low_wheel_joint"]]) + float(jvel[idx["left_up_wheel_joint"]])) / 2.0
    q1dot  = float(jvel[idx["joint1"]])
    q2dot  = float(jvel[idx["joint2"]])   # angular velocity is frame-independent

    bx = float(pos[0])
    by = float(pos[1])

    cp  = math.cos(phai)
    sp  = math.sin(phai)
    c1  = math.cos(phai + q1)
    s1  = math.sin(phai + q1)
    c12 = math.cos(phai + q1 + q2)
    s12 = math.sin(phai + q1 + q2)

    # Task-space state (4D): [ee_x, ee_y, arm_base_x, arm_base_y]
    X = np.array([
        bx + p.d * cp + p.l1 * c1 + p.l2 * c12,
        by + p.d * sp + p.l1 * s1 + p.l2 * s12,
        bx + p.d * cp,
        by + p.d * sp,
    ])

    # q_wmm = [qr, ql, phai, q1, q2]; qr/ql positions unused by Jacobian
    q_wmm = np.array([0.0, 0.0, phai, q1, q2])
    v     = np.array([qr_dot, ql_dot, q1dot, q2dot])
    J     = kinematics_jacobian(p, q_wmm)
    dX    = J @ v

    # Full joint velocity vector for compute_bar_matrices
    dq = np.array([
        qr_dot, ql_dot,
        p.r * (qr_dot - ql_dot) / (2.0 * p.b),
        q1dot, q2dot,
    ])

    return X, dX, q_wmm, dq

# ── One-step adaptive controller (extracted from rhs, no ODE integration) ─────
def compute_control(t_ctrl, X, dX, q_wmm, dq, Wc, W1, W2, W3, W4):
    Xd, dXd, ddXd = desired_trajectory(t_ctrl)
    e1  = X  - Xd
    de1 = dX - dXd

    kb      = (p.up_bound - p.down_bound) * np.exp(-p.rate * t_ctrl) + p.down_bound
    kb_dot  = -p.rate * (p.up_bound - p.down_bound) * np.exp(-p.rate * t_ctrl)
    kb_ddot =  p.rate ** 2 * (p.up_bound - p.down_bound) * np.exp(-p.rate * t_ctrl)

    miu  = -p.k1 * e1 + (kb_dot / kb) * e1 + dXd
    dmiu = (ddXd - p.k1 * de1
            + ((kb_ddot * kb - kb_dot ** 2) / kb ** 2) * e1
            + (kb_dot / kb) * de1)
    e2 = dX - miu

    # Critic basis functions
    Sc = np.exp(
        -np.sum((e1[:, None] - p.cij[None, :]) ** 2, axis=0) / p.width_c ** 2
    )
    A = np.array([
        -(Sc[i] / p.phi)
        + (-2.0 * Sc[i] * np.sum(e1 - c) / p.width_c ** 2) * np.sum(de1)
        for i, c in enumerate(p.cij)
    ])

    # Actor basis functions
    Sa1 = rbf_scalar(e2[0], p.c_a, p.width_a)
    Sa2 = rbf_scalar(e2[1], p.c_a, p.width_a)
    Sa3 = rbf_scalar(e2[2], p.c_a, p.width_a)
    Sa4 = rbf_scalar(e2[3], p.c_a, p.width_a)
    Fnn = np.array([W1 @ Sa1, W2 @ Sa2, W3 @ Sa3, W4 @ Sa4])

    M_bar, C_bar, E_bar, _ = compute_bar_matrices(p, q_wmm, dq)

    # Control law (BLF + adaptive feedforward)
    term1 = -e1 / eps_clip(kb ** 2 - e1 ** 2, 1e-6)
    u = np.linalg.pinv(E_bar) @ (
        term1 - p.k2 * (M_bar @ e2) + C_bar @ miu + M_bar @ dmiu + Fnn
    )

    # Saturate to prevent physics blow-up during initial transient
    u[0] = np.clip(u[0], -TORQUE_LIM_WHEEL, TORQUE_LIM_WHEEL)
    u[1] = np.clip(u[1], -TORQUE_LIM_WHEEL, TORQUE_LIM_WHEEL)
    u[2] = np.clip(u[2], -TORQUE_LIM_ARM,   TORQUE_LIM_ARM)
    u[3] = np.clip(u[3], -TORQUE_LIM_ARM,   TORQUE_LIM_ARM)

    # Critic weight update
    inst_reward = e1.T @ p.D @ e1 + u.T @ p.R @ u
    grad_c      = (inst_reward + Wc @ A) * A
    wc_norm     = np.linalg.norm(Wc)
    if wc_norm < p.Wc_norm_limit or (
        np.isclose(wc_norm, p.Wc_norm_limit) and Wc @ grad_c > 0
    ):
        dWc = -p.lrc * grad_c
    else:
        dWc = -p.lrc * grad_c + p.lrc * (Wc @ grad_c / max(wc_norm ** 2, 1e-9)) * Wc

    # Actor weight updates
    def actor_dw(W, Sa, z, lr):
        prho   = Sa * z
        norm_W = np.linalg.norm(W)
        if norm_W < p.Wa_norm_limit or (
            np.isclose(norm_W, p.Wa_norm_limit) and W @ prho > 0
        ):
            return -lr * prho
        return -lr * prho + lr * (W @ prho / max(norm_W ** 2, 1e-9)) * W

    dW1 = actor_dw(W1, Sa1, e2[0], p.lra[0])
    dW2 = actor_dw(W2, Sa2, e2[1], p.lra[1])
    dW3 = actor_dw(W3, Sa3, e2[2], p.lra[2])
    dW4 = actor_dw(W4, Sa4, e2[3], p.lra[3])

    return u, dWc, dW1, dW2, dW3, dW4, e1, Xd, kb

# ── Isaac Sim setup ───────────────────────────────────────────────────────────
my_world = World(stage_units_in_meters=1.0)

my_jetbot = my_world.scene.add(
    WheeledRobot(
        prim_path="/World/Jetbot",
        name="my_jetbot",
        wheel_dof_names=[
            "left_low_wheel_joint",
            "left_up_wheel_joint",
            "right_low_wheel_joint",
            "right_up_wheel_joint",
        ],
        create_robot=True,
        usd_path=os.path.join(_REPO_ROOT, "robot", "usd", "robot_wheel_0904.usd"),
        # Initial position so that X_init = Xd(0) = [0.65, 0.75, 0.40, 0.10] exactly.
        # base_x=0.366 → X[2]=0.366+0.034=0.400; base_y=0.10 → X[3]=0.10
        # q1=0.675 rad, q2_isaac=2.898 rad → X[0]≈0.65, X[1]≈0.75
        # All e1(0) ≈ 0  →  barrier function valid with full margin
        position=np.array([0.366, 0.10, 0.5], dtype=np.float32),
    )
)

my_world.scene.add_default_ground_plane()
my_world.reset()

# ── DOF indices ───────────────────────────────────────────────────────────────
open(LOG_FILE, "w").close()
log("=== WMM Isaac Sim ===")

idx = {}
for name in ["joint1", "joint2",
             "left_low_wheel_joint", "left_up_wheel_joint",
             "right_low_wheel_joint", "right_up_wheel_joint"]:
    try:
        idx[name] = my_jetbot.get_dof_index(name)
    except Exception as e:
        log(f"ERROR: get_dof_index({name}): {e}")

log(f"DOF indices: {idx}")

# Fixed effort ordering: [joint1, ll, lu, rl, ru, joint2]
DOF_ORDER = np.array([
    idx["joint1"],
    idx["left_low_wheel_joint"],
    idx["left_up_wheel_joint"],
    idx["right_low_wheel_joint"],
    idx["right_up_wheel_joint"],
    idx["joint2"],
], dtype=np.int32)

# ── Set initial joint angles right after reset (before first step) ───────────
# reset_needed block only fires on MANUAL stop/restart; this covers first run.
_ARM_IDX = np.array([idx["joint1"], idx["joint2"]], dtype=np.int32)
_WHEEL_IDX = np.array([
    idx["left_low_wheel_joint"], idx["left_up_wheel_joint"],
    idx["right_low_wheel_joint"], idx["right_up_wheel_joint"],
], dtype=np.int32)
my_jetbot.set_joint_positions(
    positions=np.array([INIT_Q1, INIT_Q2_ISAAC], dtype=np.float32),
    joint_indices=_ARM_IDX,
)
log(f"Initial arm positions set: joint1={INIT_Q1} rad, joint2={INIT_Q2_ISAAC} rad")

# ── Simulation state ──────────────────────────────────────────────────────────
reset_needed = False
step_count   = 0
ctrl_step    = 0
t_ctrl       = 0.0

Wc = np.zeros(nc)
W1 = np.zeros(na)
W2 = np.zeros(na)
W3 = np.zeros(na)
W4 = np.zeros(na)

ZERO_EFFORTS = np.zeros(6, dtype=np.float32)

# ── Main loop ─────────────────────────────────────────────────────────────────
while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_stopped() and not reset_needed:
        reset_needed = True

    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            step_count = 0
            ctrl_step  = 0
            t_ctrl     = 0.0
            Wc = np.zeros(nc)
            W1 = np.zeros(na)
            W2 = np.zeros(na)
            W3 = np.zeros(na)
            W4 = np.zeros(na)
            reset_needed = False
            # Reset arm joints to initial config on manual restart
            my_jetbot.set_joint_positions(
                positions=np.array([INIT_Q1, INIT_Q2_ISAAC], dtype=np.float32),
                joint_indices=_ARM_IDX,
            )

        step_count += 1

        # ── Settle phase: base drops to ground, arm held at initial config ───────
        if step_count <= SETTLE_STEPS:
            # Force arm joints to INIT angles every step (guards against USD stiffness drift)
            my_jetbot.set_joint_positions(
                positions=np.array([INIT_Q1, INIT_Q2_ISAAC], dtype=np.float32),
                joint_indices=_ARM_IDX,
            )
            # Zero torque on wheel joints only
            my_jetbot.set_joint_efforts(
                efforts=np.zeros(4, dtype=np.float32),
                joint_indices=_WHEEL_IDX,
            )
            continue

        # ── Adaptive control phase ────────────────────────────────────────────
        try:
            X, dX, q_wmm, dq = get_robot_state(my_jetbot)

            u, dWc, dW1, dW2, dW3, dW4, e1, Xd, kb = compute_control(
                t_ctrl, X, dX, q_wmm, dq, Wc, W1, W2, W3, W4
            )

            # Euler weight update
            Wc += dWc * PHYSICS_DT
            W1 += dW1 * PHYSICS_DT
            W2 += dW2 * PHYSICS_DT
            W3 += dW3 * PHYSICS_DT
            W4 += dW4 * PHYSICS_DT

            # Map u → Isaac Sim joint efforts
            # u = [u_r, u_l, u_q1, u_q2]  (physical torques from S_q.T@B=I proof)
            # Left-wheel sign flip: Isaac Sim left wheel +vel = backward (WMM convention reversed)
            efforts = np.array([
                u[2],          # joint1   (DOF 0)
               -u[1] / 2.0,   # left_low  (DOF 1)
               -u[1] / 2.0,   # left_up   (DOF 2)
                u[0] / 2.0,   # right_low (DOF 3)
                u[0] / 2.0,   # right_up  (DOF 4)
                u[3],          # joint2   (DOF 5)
            ], dtype=np.float32)

            my_jetbot.set_joint_efforts(efforts=efforts, joint_indices=DOF_ORDER)

            t_ctrl  += PHYSICS_DT
            ctrl_step += 1

            if ctrl_step % 60 == 0:
                pos, _ = my_jetbot.get_world_pose()
                phai   = q_wmm[2]
                log(
                    f"t={t_ctrl:.2f}s | "
                    f"X=[{X[0]:.3f},{X[1]:.3f},{X[2]:.3f},{X[3]:.3f}] "
                    f"Xd=[{Xd[0]:.3f},{Xd[1]:.3f},{Xd[2]:.3f},{Xd[3]:.3f}] "
                    f"e1=[{e1[0]:.3f},{e1[1]:.3f},{e1[2]:.3f},{e1[3]:.3f}] "
                    f"u=[{u[0]:.2f},{u[1]:.2f},{u[2]:.2f},{u[3]:.2f}] "
                    f"phai={math.degrees(phai):.1f}deg "
                    f"base=({float(pos[0]):.3f},{float(pos[1]):.3f})"
                )

            if ctrl_step >= MAX_CTRL_STEPS:
                log("=== MAX_CTRL_STEPS reached, stopping ===")
                break

        except Exception as e:
            log(f"[step {step_count}] Controller error: {e}\n{traceback.format_exc()}")

    if args.test:
        break

my_world.stop()
simulation_app.close()
