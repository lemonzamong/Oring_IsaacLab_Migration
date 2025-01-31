# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.assets import SoftObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import subtract_frame_transforms, euler_xyz_from_quat

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLDOTaskEnv
    # from omni.isaac.orbit_tasks.Soft.disentangle.disentangle_env import DisentangleEnv #FIXME


# def ee_positions(
#         env: RLTaskEnv,
#         robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#         goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_pole"),
#     ) -> torch.Tensor:
#     """Current end-effector position of the arm. + relate rigid pole"""
#     robot: RigidObject = env.scene[robot_cfg.name]
#     goal: RigidObject = env.scene[goal_cfg.name]
#     ee_position = robot.data.ee_state_w[:, :3] - env.envs_positions #FIXME: env.envs_positions?
#     return ee_position - env.init_rigid_pos # relative postion 

# def ee_orientations_cos(
#         env: RLTaskEnv,
#         robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#         goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_pole"),
#     ) -> torch.Tensor:
#     """Current end-effector orientation of the arm."""
#     robot: RigidObject = env.scene[robot_cfg.name]
#     goal: RigidObject = env.scene[goal_cfg.name]
#     # make the first element positive
#     quat_w = robot.data.ee_state_w[:, 3:7]
#     quat_w[quat_w[:, 0] < 0] *= -1
#     _,_, yaw = euler_xyz_from_quat(quat_w)
#     cos_yaw= np.cos(yaw)
#     return cos_yaw.reshape(env.num_envs, 1)

# def ee_orientations_sin(
#         env: RLTaskEnv,
#         robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#         goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_pole"),
#     ) -> torch.Tensor:
#     """Current end-effector orientation of the arm."""
#     robot: RigidObject = env.scene[robot_cfg.name]
#     goal: RigidObject = env.scene[goal_cfg.name]
#     """Current end-effector orientation of the arm."""
#     # make the first element positive
#     quat_w = robot.data.ee_state_w[:, 3:7]
#     quat_w[quat_w[:, 0] < 0] *= -1
#     _,_, yaw = euler_xyz_from_quat(quat_w)
#     sin_yaw= np.cos(yaw)
#     return sin_yaw.reshape(env.num_envs, 1)

# # - OBJECT
# # --- oring state
# def object_embedding(
#         env: RLTaskEnv,
#         object_cfg: SceneEntityCfg = SceneEntityCfg("oring"),
#     ) -> torch.Tensor:
#     """Oring latent vectors"""
#     object: SoftObject = env.scene[object_cfg.name] 
    
#     return env.latent_vectors #FIXME: latent_vectors to Soft object? depends on pcd_util implementation

# def object_transformation(
#         env: RLTaskEnv,
#         object_cfg: SceneEntityCfg = SceneEntityCfg("oring"),
#     ) -> torch.Tensor:
#     """Oring pose info. factor"""
#     object: SoftObject = env.scene[object_cfg.name] 

#     return env.scale_factors #FIXME: scale_factors to Soft object? depends on pcd_util implementation


# # - ETC
# # -- phase state
# def z_move_states(
#         env: RLTaskEnv,
#         robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#         object_cfg: SceneEntityCfg = SceneEntityCfg("oring"),
#         goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_pole"),
#     ) -> torch.Tensor:
#     """ Get z move up[1]/down[0]"""
        
#     return env.z_phase

# def aux_pole_position(
#         env: RLTaskEnv,
#         aux_cfg: SceneEntityCfg = SceneEntityCfg("aux_pole"),
#     ) -> torch.Tensor:
#     """ position of the auxiliary pole"""
#     aux_pole: RigidObject = env.scene[aux_cfg.name]
#     return env.aux_pole.pos
