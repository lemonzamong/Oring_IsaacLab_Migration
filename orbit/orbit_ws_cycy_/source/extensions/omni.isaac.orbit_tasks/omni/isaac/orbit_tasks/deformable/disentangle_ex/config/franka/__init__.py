# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# 24.01.14 
# Chanyoung Ahn

import gymnasium as gym

from . import agents, joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

# gym.register(
#     id="Isaac-Disentangle-Franka-v0",
#     entry_point="omni.isaac.orbit.envs:RLDOTaskEnv", 
#     kwargs={
#         "env_cfg_entry_point": joint_pos_env_cfg.FrankaOringDisentangleEnvCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DisentanlgePPORunnerCfg,
#     },
#     disable_env_checker=True,
# )

