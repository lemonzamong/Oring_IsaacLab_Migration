# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 24.02.19
# Chanyoung Ahn

"""
    This script demonstrates provide partial pcd of deformables for checking reconstruction model.
    If you want to add new shape of deformables, change objects in 'test_twist.usd'

.. code-block:: bash

    # Usage
    ./orbit.sh -p source/standalone/tutorials/06_deformables/get_twist_pcd.py --vis

"""


from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU device for camera output.")
parser.add_argument("--draw", action="store_true", default=True, help="Draw the obtained pointcloud on viewport.")
parser.add_argument("--save", action="store_true", default=False, help="Save the obtained data to disk.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import numpy as np
import os
import random
import torch
import traceback

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.debug_draw._debug_draw as omni_debug_draw
import omni.replicator.core as rep
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.prims.soft.deformable_prim import DeformablePrim
from pxr import Gf, UsdGeom

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.sensors.camera import Camera, CameraCfg
from omni.isaac.orbit.utils import convert_dict_to_backend
from omni.isaac.orbit.utils.math import project_points, transform_points, unproject_depth
from omni.isaac.orbit_assets import LOCAL_ASSETS_DIR

# Acquire draw interface
draw_interface = omni_debug_draw.acquire_debug_draw_interface()


def define_sensor() -> Camera:
    prim_utils.create_prim("/World/Origin_00", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/Origin_.*/CameraSensor",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane", 
                    "normals", "semantic_segmentation",
                    "pointcloud"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        # semantic_types=[{"class": "cube"}]
        semantic_labels=["oring"],
    )
    # Create camera
    camera = Camera(cfg=camera_cfg)

    return camera


def design_scene():
    """Design the scene."""
    # Populate scene
    # -- Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # -- Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    prim_utils.create_prim(
        # usd_path = f"{LOCAL_ASSETS_DIR}/disentangle/test_two_ring_scene.usd", # two O-Ring
        # usd_path = f"{LOCAL_ASSETS_DIR}/disentangle/test_ring_scene.usd", # 005_14 O-Ring
        # usd_path = f"{LOCAL_ASSETS_DIR}/disentangle/test_twist_scene.usd", # invisible nail and hook
        usd_path = f"{LOCAL_ASSETS_DIR}/disentangle/test_twist.usd", # invisible nail and hook
        prim_path = f"/World/Objects",
        translation=np.asarray([0., 0., 0.0]),
        semantic_label="oring",
    )

    prim_utils.create_prim(
        usd_path = f"{LOCAL_ASSETS_DIR}/disentangle/test_twist_nail.usd", # invisible nail and hook
        prim_path = f"/World/nails",
        translation=np.asarray([0., 0., 0.0]),
    )
    
    # Sensors
    camera = define_sensor()

    # return the scene information
    scene_entities = {"camera": camera}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    # extract entities for simplified notation
    camera: Camera = scene_entities["camera"]

    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)

    # Set pose: There are two ways to set the pose of the camera.
    eyes = torch.tensor([[2.5, 2.5, 2.5]], device=sim.device)
    targets = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)

    camera.set_world_poses_from_view(eyes, targets)

    for _ in range(200):
        sim.step()

    # Simulate physics
    while simulation_app.is_running():
        # Step simulation
        sim.step()
        # Update camera data
        camera.update(dt=sim.get_physics_dt())
    
        # Print camera info
        print(camera)
        print("Received shape of rgb   image: ", camera.data.output["rgb"].shape)
        print("Received shape of depth image: ", camera.data.output["distance_to_image_plane"].shape)
        print("-------------------------------")

        # Extract camera data
        if args_cli.save:
            # Save images from camera 1
            camera_index = 1
            # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
            if sim.backend == "torch":
                # tensordict allows easy indexing of tensors in the dictionary
                single_cam_data = convert_dict_to_backend(camera.data.output[camera_index], backend="numpy")
            else:
                # for numpy, we need to manually index the data
                single_cam_data = dict()
                for key, value in camera.data.output.items():
                    single_cam_data[key] = value[camera_index]
            # Extract the other information
            single_cam_info = camera.data.info[camera_index]

            # Pack data back into replicator format to save them using its writer
            rep_output = dict()
            for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                if info is not None:
                    rep_output[key] = {"data": data, "info": info}
                else:
                    rep_output[key] = data
            # Save images
            # Note: We need to provide On-time data for Replicator to save the images.
            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            rep_writer.write(rep_output)

        # Draw pointcloud
        if args_cli.draw:

            # partially point cloud 
            points_3d_world = camera.data.output["pointcloud"]
            # Convert to numpy for visualization
            if not isinstance(points_3d_world, np.ndarray):
                points_3d_world = points_3d_world.cpu().numpy()
            # Clear any existing points
            draw_interface.clear_points()
            # Obtain drawing settings
            num_batch = points_3d_world.shape[0]
            num_points = points_3d_world.shape[1]
            
            
            points_size = [5.25] * num_points
            # Fix random seed
            random.seed(0)
            
            # We can get partially point cloud (No segmentation!)
            # points_3d_world
            # Visualize the points
            for index in range(num_batch):
                # generate random color
                color = [random.random() for _ in range(3)]
                color += [1.0]
                # plain color for points
                points_color = [color] * num_points
                draw_interface.draw_points(points_3d_world[index].tolist(), points_color, points_size)
                
                np.savetxt(os.path.join(LOCAL_ASSETS_DIR, "disentangle", "twist_1_collision.xyz"), points_3d_world[index])
                print("save it !!")
                # simulation_app.close()
                

def main():
    """Main function."""
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(device="cpu" if args_cli.cpu else "cuda")
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # design the scene
    scene_entities = design_scene()
    # Play simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run simulator
    run_simulator(sim, scene_entities)


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
