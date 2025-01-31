# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.assets import SoftObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import FrameTransformer
from omni.isaac.orbit.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv
    # from omni.isaac.orbit_tasks.oring.disentangle.disentangle_env import DisentangleEnv #FIXME


# def object_is_lifted(
#     env: RLTaskEnv, 
#     minimal_height: float, 
#     object_cfg: SceneEntityCfg = SceneEntityCfg("oring")
# ) -> torch.Tensor:
#     """Reward the agent for lifting the object above the minimal height."""
#     object: SoftObject = env.scene[object_cfg.name]
#     return torch.where(max(object.data.particle_pos) > minimal_height, 1.0, 0.0) #FIXME: particle positions?



def ee_goal_distance(
    env: RLTaskEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("oring"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_pole"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: SoftObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]
    command = env.command_manager.get_command(command_name)

    #FIXME:
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold


    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

# def object_chamfer_distance(
#           env: RLTaskEnv):
#         return torch.as_tensor(env.chamfer_dists) #FIXME
