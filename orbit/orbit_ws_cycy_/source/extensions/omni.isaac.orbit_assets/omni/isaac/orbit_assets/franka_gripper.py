# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 24.01.14 
# Chanyoung Ahn

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`PANDA_CFG`: Panda hand gripper
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Panda robot with Panda hand with stiffer PD control
"""

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import ImplicitActuatorCfg
from omni.isaac.orbit.assets.articulation import ArticulationCfg
# from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR

import os
current_directory = os.path.dirname(os.path.abspath(__file__))
_FRANKA_INSTANCEABLE_USD = os.path.join(current_directory, 'usd', 'disentangle', 'franka_gripper_fix.usd')

##
# Configuration
##

PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_FRANKA_INSTANCEABLE_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos=(0.15, 0.25, 0.2),
        pos=(0.15, 0.25, 2.),
 
        rot=(-0.707, 0, 0, -0.707),
        joint_pos={
            "joint_x": 0.0,
            "joint_y": 0.0,
            "joint_z": 0.0,
            "rev_x": 0.0,
            "rev_y": 0.0,
            "rev_z": 0.0,
        },
    ),
    actuators={

        "wrist_trans": ImplicitActuatorCfg(
            joint_names_expr=["joint_[x-z]"],
            effort_limit=1000,
            velocity_limit=1e7,
            stiffness=8e6,
            damping=1e4,
        ),
        
        "wrist_rev": ImplicitActuatorCfg(
            joint_names_expr=["rev_[x-z]"],
            effort_limit=1000,
            velocity_limit=1e7,
            stiffness=8e6,
            damping=1e4,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


# FRANKA_PANDA_HIGH_PD_CFG = PANDA_CFG.copy()
# FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
# FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
# FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
