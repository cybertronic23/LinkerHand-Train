# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class HandrlEnvCfg(DirectRLEnvCfg):
    """Five-finger joint position reaching task (Direct RL)."""

    # -----------------
    # env params
    # -----------------
    decimation = 2
    episode_length_s = 5.0
    is_finite_horizon = False

    # -----------------
    # joints to control
    # -----------------
    control_joint_names = [
        "thumb_cmc_pitch",
        "index_mcp_pitch",
        "middle_mcp_pitch",
        "ring_mcp_pitch",
        "pinky_mcp_pitch",
    ]

    # per-joint target range (rad)
    target_pos_ranges = {
        "thumb_cmc_pitch": (0.0, 0.99),
        "index_mcp_pitch": (0.0, 1.26),
        "middle_mcp_pitch": (0.0, 1.26),
        "ring_mcp_pitch": (0.0, 1.26),
        "pinky_mcp_pitch": (0.0, 1.26),
    }

    # -----------------
    # spaces
    # obs = [q, qd, (q-target)] for each joint => 5 * 3 = 15
    # -----------------
    action_space = 5
    observation_space = 15
    state_space = 0

    # -----------------
    # simulation
    # -----------------
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # -----------------
    # scene
    # -----------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=2.0,
        replicate_physics=True,
        filter_collisions=True,
    )

    # -----------------
    # rewards / termination
    # -----------------
    rew_pos = 1.0
    rew_vel = 0.01
    rew_action = 0.001
    success_bonus = 1.0
    success_thresh = 0.02  # rad, all controlled joints within this => success

    # -----------------
    # robot
    # -----------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/chen/myfiles/linkerhand-urdf-main/l6/right/linkerhand_l6_right/linkerhand_l6_right.usd",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.2),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "hand_act": ImplicitActuatorCfg(
                joint_names_expr=[
                    "thumb_cmc_pitch",
                    "index_mcp_pitch",
                    "middle_mcp_pitch",
                    "ring_mcp_pitch",
                    "pinky_mcp_pitch",
                ],
                stiffness=80.0,
                damping=8.0,
                effort_limit_sim=10.0,     # 不够力就调大：20~30
                velocity_limit_sim=20.0,
            )
        },
    )