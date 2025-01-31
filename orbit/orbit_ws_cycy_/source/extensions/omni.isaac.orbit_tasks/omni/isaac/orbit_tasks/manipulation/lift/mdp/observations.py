# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import subtract_frame_transforms
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.isrl import ISRL_DIR
from omni.isaac.orbit.isrl import utils as isrl_utils
from omni.isaac.orbit.isrl import NeuralProcessImplicit3DHypernet as iSRLModel
from omni.isaac.orbit.envs.mdp.observations import joint_vel_rel

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv

isrl_cfg_path = f'{ISRL_DIR}/configs/primitives_final.yaml'
isrl_cfg = isrl_utils.load_config(isrl_cfg_path)
isrl_model: iSRLModel = iSRLModel(isrl_cfg['model'])
isrl_ckpt_path = f'{ISRL_DIR}/checkpoints/primitives_final.pth'
isrl_model.load_state_dict(torch.load(isrl_ckpt_path))
isrl_model.cuda()
isrl_model.eval()

def joint_vel_rel_discrete(
    env: BaseEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.00001
) -> torch.Tensor:
    """The joint velocities of the asset w.r.t. the default joint velocities."""
    """joint vel > 0 + threshold -> 1, joint vel < 0 - threshold -> -1, else 0"""
    joint_vel = joint_vel_rel(env, asset_cfg)
    joint_vel_discrete = torch.where(joint_vel > threshold, 1.0, torch.where(joint_vel < -threshold, -1.0, 0.0))

    return joint_vel_discrete


def object_position_in_robot_root_frame(
    env: RLTaskEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

def isrl_emb(
    env: RLTaskEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
) -> torch.Tensor:
    
    ADD_NOISE = False
    

    """Get point cloud of the object scanned from camera."""
    # object: RigidObject = env.scene[object_cfg.name]
    camera = env.scene[camera_cfg.name]
    raw_pcd = camera.data.output["pointcloud"]
    """Downsample point cloud."""
    num_points = 1024
    idx = torch.randperm(num_points)
    sampled_pcd = raw_pcd[:,idx]

    """Normalize point cloud to the object's bounding box."""
    centroid = torch.mean(sampled_pcd, keepdim=True, dim=1) # (num_envs, 3)
    normalized_pcd = sampled_pcd - centroid 
    furthest_distance = torch.max(torch.linalg.norm(normalized_pcd, dim=-1, keepdim=True), dim=1, keepdim=True).values
    furthest_distance *= 2.0 # FIXME: This is a hack to make the object fit in the bounding box
    normalized_pcd /= furthest_distance

    if ADD_NOISE:
        noise = torch.randn_like(normalized_pcd) * 0.05
        normalized_pcd += noise


    """Get the ISRL embedding for the object."""
    object: RigidObject = env.scene[object_cfg.name]

    # if isrl_model is None:
    #     return torch.zeros(1, 256, device=object.device)

    """Get centroid in robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    centroid, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], centroid.squeeze(dim=1)
    )

    """ True object position"""
    # object_pos_b = object_position_in_robot_root_frame(env, robot_cfg, object_cfg)
    # print(torch.linalg.norm(object_pos_b- centroid, dim=-1))

    latent_vec = isrl_model.encode(normalized_pcd)
    if any(torch.isnan(latent_vec).flatten()):
        print("latent_vec contains NaNs!!")
        """replace row with NaNs with zeros"""
        nan_mask = torch.isnan(latent_vec).any(dim=1)
        latent_vec[nan_mask] = 0.0

    embedding = torch.cat((latent_vec, centroid, furthest_distance.squeeze(dim=1)), dim=1)

    return embedding

