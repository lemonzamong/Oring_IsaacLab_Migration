#TODO: goal pole distange randomization, dummy step, oring parameter randomization

# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different randomizations.

Randomization includes anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`omni.isaac.orbit.managers.RandomizationTermCfg` object to enable
the randomization introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject, DeformObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def randomize_deform_body_parameter(
    env: RLTaskEnv,
    env_ids: torch.Tensor | None,
    siffness: tuple[float, float],
    youngs: tuple[float, float],
    asset_cfg: SceneEntityCfg,
):
    #TODO: uniform random? Normal random?
    pass
    

def randomize_aux_pole_position(
    env: RLTaskEnv,
    env_ids: torch.Tensor | None,
    position: tuple[float, float],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the position of the auxiliary pole."""
    #TODO: dummy step
    pass