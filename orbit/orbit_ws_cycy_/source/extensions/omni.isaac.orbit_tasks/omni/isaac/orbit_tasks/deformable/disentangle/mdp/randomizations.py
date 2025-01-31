# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# 24.01.17
# Chanyoung Ahn

"""
Randomization for O-Ring Manipulation

Common functions that can be used to enable different randomizations.

Randomization includes anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`omni.isaac.orbit.managers.RandomizationTermCfg` object to enable
the randomization introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

# from omni.isaac.orbit.assets import ArticulationCfg
from omni.isaac.orbit.assets import Articulation, RigidObject, SoftObject, AssetBase
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import quat_from_euler_xyz, sample_uniform
from omni.isaac.orbit_assets import LOCAL_ASSETS_DIR
from omni.isaac.orbit.compat.markers import StaticMarker
# from omni.isaac.core.utils.stage import get_current_stage
# from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade
# import omni.isaac.core.utils.prims as prim_utils
# from omni.isaac.core.utils.stage import get_current_stage

# from omni.usd.commands import DeletePrimsCommand
from omni.isaac.orbit.assets import ArticulationCfg
from omni.isaac.orbit_assets.franka_gripper import PANDA_CFG  # isort: skip
# from omni.isaac.orbit.scene import InteractiveScene
from omni.isaac.core.utils.xforms import get_world_pose
from omni.isaac.core.utils.prims import find_matching_prim_paths

#test
import omni.isaac.orbit.compat.utils.kit as kit_utils

import numpy as np
import os 
import glob 
if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv

#TODO: goal pole distange randomization, dummy step, oring parameter randomization


"""
O-RING RANDOMIZATION
"""

# TODO: Change for diverse O-Ring spwan
def randomize_init_oring_shape(
    env: BaseEnv,
    env_ids: torch.Tensor | None,
    oring_type: list[tuple[int, str]], # [(0, "07_015"), (1, "07_01")]
    asset_cfg: SceneEntityCfg,
    ):
    # print("random oring")
    asset: SoftObject = env.scene[asset_cfg.name]
    num_envs = env.scene.num_envs
    if env_ids is None:
        env_ids = torch.arange(num_envs)
    # indices = env_ids
    # if oring_type is None:
    rand_paths = glob.glob(os.path.join(LOCAL_ASSETS_DIR, "disentangle", "spawn", "mesh", "*.txt"))
    oring_07_01 = []
    rand_names = []
    for rand_path in rand_paths:
        data = np.loadtxt(rand_path)
        oring_07_01.append(data)
        rand_names.append(os.path.basename(rand_path))
    oring_07_01 = torch.tensor(oring_07_01, device="cuda:0")
    
    # rand_paths = 
    oring_idx = asset.set_simulation_mesh_nodal_positions(positions=oring_07_01)
    
    _set_init_position(env, env_ids, rand_names, oring_idx)
        
    
"""
Init Pole RANDOMIZATION
"""


# TODO: Add various randomization range for various O-Ring Shape 
def set_init_pole_position(
    env: BaseEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("init_pole"),
):
    # print("set init_random")
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


# TODO: Add various randomization range for various O-Ring Shape 
def randomize_init_pole_position(
    env: BaseEnv,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("init_pole"),
):
    # print("set init_random")
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.default_joint_pos[env_ids].clone()

    if asset_cfg.joint_names is None:
        joint_indices = range(len(joint_pos))
    else:
        joint_indices = [asset.joint_indices[name] for name in asset_cfg.joint_names]
        
    # env.sim.pause()
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    
    for idx in joint_indices:
        joint_pos[:, idx] += sample_uniform(*position_range, joint_pos[:, idx].shape, joint_pos.device)
    asset.set_joint_position_target(target=joint_pos, env_ids=env_ids)
    env.scene.write_data_to_sim()
    for _ in range(30):
        env.sim.step()
        env.scene.update(0.1)
        
"""
Franka Gripper RANDOMIZATION
"""     
# spawn new position
def _set_init_position(
    env: BaseEnv,
    env_ids: torch.Tensor,
    rand_names,
    oring_idx
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene["robot"]
    env_paths = find_matching_prim_paths("/World/envs/env_.*")
    
    for rand_name in rand_names:
        rand_path = glob.glob(os.path.join(LOCAL_ASSETS_DIR, "disentangle", "spawn", "positions", rand_name))
        data = np.loadtxt(rand_path)
        print("d")
    # for rand_path in rand_paths:
    #     data = np.loadtxt(rand_path)
    #     oring_07_01.append(data)

    root_pos = asset.data.root_state_w[:, :7]
    for i in range(env_ids):
        env_position, _ = get_world_pose(env_paths[i])
        
        root_pos[i, :3] = + torch.tensor(env_position)
        # print(f"env_{i} reset {rand_ints[i]}")
    # set into the physics simulation
    asset.write_root_pose_to_sim(root_pos, env_ids)

# manual gripp        
def set_manual_grip(
    env: BaseEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    marker_vis: bool = False,
):
    
    print("set init_random")
    asset: Articulation = env.scene[asset_cfg.name]
    
    if marker_vis:
        marker = StaticMarker(prim_path = "/World/Visuals/goal", 
                              count = len(env_ids),
                              scale=(0.5, 0.5, 0.5))
        
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    if asset_cfg.joint_names is None:
        idx = asset.joint_indices["joint_z"] 
    else:
        idx = asset.joint_indices[asset_cfg.joint_names[0]] #?
        
    # manual move -z axis
    joint_vel[:, idx] = -30.
    asset.set_joint_velocity_target(target=joint_vel, env_ids=env_ids)
    for _ in range(5): # FIX
        if marker_vis:
            position = asset.data.body_pos_w[:,0,:]
            marker.set_world_poses(position)
            
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(0.01)
    
    joint_vel[:, idx] = -0.1
    asset.set_joint_velocity_target(target=joint_vel, env_ids=env_ids)
    env.scene.write_data_to_sim()
    

    # manual grapsing 
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_pos[:, -2:] = 0. 
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    env.scene.write_data_to_sim()
        
    for _ in range(10):
    # while True:
        env.sim.step()
        env.scene.update(0.01)
