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


def object_out_of_sight(
    env: RLTaskEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
) -> torch.Tensor:
    
    camera = env.scene[camera_cfg.name]
    raw_pcd = camera.data.output["pointcloud"] 

    zero_mask = (raw_pcd == 0).all(dim=(1, 2))

    return zero_mask

