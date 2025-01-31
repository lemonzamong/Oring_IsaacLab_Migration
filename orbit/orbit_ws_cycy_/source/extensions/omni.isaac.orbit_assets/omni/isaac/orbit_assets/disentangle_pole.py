# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 24.01.21
# Chanyoung Ahn

"""Configuration for init pole in disentangle task.

The following configurations are available:

* :obj:`POLE_CFG`: Distentangle pole configure
"""

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import ImplicitActuatorCfg
from omni.isaac.orbit.assets.articulation import ArticulationCfg

import os
current_directory = os.path.dirname(os.path.abspath(__file__))
_POLE_INSTANCEABLE_USD = os.path.join(current_directory, 'usd', 'disentangle', 'init_nail_mover.usd')

##
# Configuration
##

POLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_POLE_INSTANCEABLE_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0, #?
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0., -0.8, 0.),
        joint_pos={
            "joint_x": 0.0,
            "joint_y": 0.0,
        },
    ),
    actuators={

        "trans_joint": ImplicitActuatorCfg(
            joint_names_expr=["joint_[x-y]"],
            effort_limit=1e4, # ?
            velocity_limit=1e7,
            stiffness=1e8,
            damping=8e6,
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
