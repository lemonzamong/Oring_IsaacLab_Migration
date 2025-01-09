# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 24.01.31
# Chanyoung Ahn
# 25.01.08
# Hyeokjun Kwon


"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`PANDA_CFG`: Panda hand gripper
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Panda robot with Panda hand with stiffer PD control
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
# from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

import os
current_directory = os.path.dirname(os.path.abspath(__file__))
_FRANKA_INSTANCEABLE_USD = os.path.join(current_directory, 'usd', 'disentangle', 'franka_gripper_fix_v2.usd')

##
# Configuration
##

PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_FRANKA_INSTANCEABLE_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, # True
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
        pos=(0.152, 0.3, 0.6), 
        rot=(-0.707, 0, 0, -0.707),
        joint_pos={
            "joint_x": 0.0,
            "joint_y": 0.0,
            "joint_z": 0.0,
            "rev_x": 0.0,
            "rev_y": 0.0,
            "rev_z": 0.0,
            "left_gripper": 0.15, # [0, 0.3]
            "right_gripper": -0.15, # [-0.3, 0]
             
        },
    ),
    actuators={

        "wrist_trans": ImplicitActuatorCfg(
            joint_names_expr=["joint_[x-z]"],
            effort_limit=10000,
            velocity_limit=1e7,
            stiffness=10.,
            damping=1e2,
        ),
        
        "fix_rev": ImplicitActuatorCfg(
            joint_names_expr=["rev_[x-y]"],
            effort_limit=10000,
            velocity_limit=1e7,
            stiffness=1e4,
            damping=0.,
        ),
        
        "wrist_rev": ImplicitActuatorCfg(
            joint_names_expr=["rev_z"],
            effort_limit=10000,
            velocity_limit=1e7,
            stiffness=10,
            damping=1e2,
        ),
        
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_gripper", "right_gripper"],
            effort_limit=10000,
            velocity_limit=1e7,
            stiffness=1e3,          
            damping=100,
        
        )
    },
    soft_joint_pos_limit_factor=1.0,
    rl_control_type = "vel"
)
"""Configuration of Franka Emika Panda robot."""


# FRANKA_PANDA_HIGH_PD_CFG = PANDA_CFG.copy()
# FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
# FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
# FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
