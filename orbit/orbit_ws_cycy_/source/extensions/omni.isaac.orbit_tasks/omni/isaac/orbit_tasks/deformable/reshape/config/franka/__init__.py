# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym

from . import agents, ik_abs_env_cfg, ik_rel_env_cfg, joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##
# Isaac-Reshape-Franka-v1

# gym.register(
#     id="Isaac-Reshape-Oring-Franka-v0",
#     entry_point="omni.isaac.orbit.envs:RLDOTaskEnv",
#     kwargs={
#         "env_cfg_entry_point": joint_pos_env_cfg.FrankaCubeLiftEnvCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftCubePPORunnerCfg,
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Reshape-Oring-Franka-Play-v0",
#     entry_point="omni.isaac.orbit.envs:RLDOTaskEnv",
#     kwargs={
#         "env_cfg_entry_point": joint_pos_env_cfg.FrankaCubeLiftEnvCfg_PLAY,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftCubePPORunnerCfg,
#     },
#     disable_env_checker=True,
# )

##
# Inverse Kinematics - Absolute Pose Control
##

# gym.register(
#     id="Isaac-Reshape-Oring-Franka-IK-Abs-v0",
#     entry_point="omni.isaac.orbit.envs:RLDOTaskEnv",
#     kwargs={
#         "env_cfg_entry_point": ik_abs_env_cfg.FrankaCubeLiftEnvCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftCubePPORunnerCfg,
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-Reshape-Oring-Franka-IK-Abs-Play-v0",
#     entry_point="omni.isaac.orbit.envs:RLDOTaskEnv",
#     kwargs={
#         "env_cfg_entry_point": ik_abs_env_cfg.FrankaCubeLiftEnvCfg_PLAY,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftCubePPORunnerCfg,
#     },
#     disable_env_checker=True,
# )

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Reshape-Oring-Franka-IK-Rel-v0",
    entry_point="omni.isaac.orbit.envs:RLDOTaskEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.FrankaOringReshapeEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.ReshapePPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Reshape-Oring-Franka-IK-Rel-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLDOTaskEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.FrankaOringReshapeEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.ReshapePPORunnerCfg,
    },
    disable_env_checker=True,
)
