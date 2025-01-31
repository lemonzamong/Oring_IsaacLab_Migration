# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# 24.01.16
# Chanyoung AHn

"""Script to an environment with random action agent. and visualize pcd"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Orbit environments.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--vis", type=str, default=True, help="Visualize pcd sensors.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch
import traceback

import carb

import omni.isaac.contrib_tasks  # noqa: F401
import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import parse_env_cfg

import random
import numpy as np
import omni.isaac.debug_draw._debug_draw as omni_debug_draw

# Acquire draw interface
draw_interface = omni_debug_draw.acquire_debug_draw_interface()

def main():
    """Random actions agent with Orbit environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():

            
            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # actions[:, 2] = 1
            # apply actions
            # print("actions", actions[0])
            env.step(actions)
            
            # Draw pointcloud
            if args_cli.vis:
                # partially point cloud 
                env.scene.sensors["camera"].update(dt=0.01)
                points_3d_world = env.scene.sensors["camera"].data.output["pointcloud"]
                # points_3d_world = camera.data.output["pointcloud"]
                # Convert to numpy for visualization
                if not isinstance(points_3d_world, np.ndarray):
                    points_3d_world = points_3d_world.cpu().numpy()
                # Clear any existing points
                draw_interface.clear_points()
                # Obtain drawing settings
                num_batch = points_3d_world.shape[0]
                num_points = points_3d_world.shape[1]
                
                # points_size = [1.25] * num_points
                points_size = [5] * num_points
                
                # Fix random seed
                random.seed(0)
                # Visualize the points
                for index in range(num_batch):
                    # generate random color
                    color = [random.random() for _ in range(3)]
                    color += [1.0]
                    # plain color for points
                    points_color = [color] * num_points
                    draw_interface.draw_points(points_3d_world[index].tolist(), points_color, points_size)

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
