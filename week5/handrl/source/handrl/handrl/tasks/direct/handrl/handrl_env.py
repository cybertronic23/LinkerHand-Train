# SPDX-License-Identifier: BSD-3-Clause

import torch
import isaaclab.sim as sim_utils

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation


class HandrlEnv(DirectRLEnv):
    """Direct RL env: control 5 finger joints to reach per-joint target positions."""

    def _setup_scene(self):
        # ground (optional but recommended)
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/groundPlane", ground_cfg)

        # create robot and register into scene
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        # clone envs
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # resolve joint ids (IsaacLab 0.36.1: safest via joint_names.index)
        joint_names = self._robot.data.joint_names
        for name in self.cfg.control_joint_names:
            if name not in joint_names:
                raise ValueError(f"Joint '{name}' not found. Available joints: {joint_names}")

        self._joint_ids = [joint_names.index(n) for n in self.cfg.control_joint_names]
        self._num_ctrl = len(self._joint_ids)  # should be 5

        # buffers: [num_envs, num_ctrl]
        self._q_des = torch.zeros((self.num_envs, self._num_ctrl), device=self.device)
        self._target_q = torch.zeros((self.num_envs, self._num_ctrl), device=self.device)

        # (optional) debug once
        # print("Controlled joints:", [(n, i) for n, i in zip(self.cfg.control_joint_names, self._joint_ids)])
        # print("joint_pos_target shape:", self._robot.data.joint_pos_target.shape)

    # -------------------------
    # RL hooks
    # -------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        # actions: [N, 5] in [-1, 1]
        actions = actions.clamp(-1.0, 1.0)

        # map each action dim to its joint's absolute target range
        for i, name in enumerate(self.cfg.control_joint_names):
            lo, hi = self.cfg.target_pos_ranges[name]
            self._q_des[:, i] = (actions[:, i] + 1.0) * 0.5 * (hi - lo) + lo  # [N]

    def _apply_action(self):
    # Some IsaacLab builds/assets store targets as [N, J] or [N, J, 1]
        if self._robot.data.joint_pos_target.ndim == 3:
            targets = self._q_des.unsqueeze(-1)   # [N,5,1]
        else:
            targets = self._q_des                # [N,5]
        self._robot.set_joint_position_target(targets, joint_ids=self._joint_ids)


    def _get_observations(self):
        # q/qd for controlled joints
        q = self._robot.data.joint_pos[:, self._joint_ids]
        qd = self._robot.data.joint_vel[:, self._joint_ids]

        # some builds store as [N, K, 1]
        if q.ndim == 3:
            q = q.squeeze(-1)
        if qd.ndim == 3:
            qd = qd.squeeze(-1)

        err = q - self._target_q  # [N, 5]
        obs = torch.cat([q, qd, err], dim=-1)  # [N, 15]
        return {"policy": obs}

    def _get_rewards(self):
        q = self._robot.data.joint_pos[:, self._joint_ids]
        qd = self._robot.data.joint_vel[:, self._joint_ids]
        if q.ndim == 3:
            q = q.squeeze(-1)
        if qd.ndim == 3:
            qd = qd.squeeze(-1)

        err = q - self._target_q  # [N, 5]

        # reward components (mean keeps scale stable w.r.t. #joints)
        rew = -self.cfg.rew_pos * torch.abs(err).mean(dim=-1)
        rew += -self.cfg.rew_vel * torch.abs(qd).mean(dim=-1)
        rew += -self.cfg.rew_action * torch.sum(self.actions**2, dim=-1)

        success = (torch.abs(err) < self.cfg.success_thresh).all(dim=-1)
        rew = rew + success.float() * self.cfg.success_bonus
        return rew

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        q = self._robot.data.joint_pos[:, self._joint_ids]
        if q.ndim == 3:
            q = q.squeeze(-1)
        err = q - self._target_q
        terminated = (torch.abs(err) < self.cfg.success_thresh).all(dim=-1)
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        # sample per-joint targets
        for i, name in enumerate(self.cfg.control_joint_names):
            lo, hi = self.cfg.target_pos_ranges[name]
            self._target_q[env_ids, i] = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)

        # reset robot joint state (all joints), set controlled joints to 0
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()

        joint_pos[:, self._joint_ids] = 0.0
        joint_vel[:, self._joint_ids] = 0.0

        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)