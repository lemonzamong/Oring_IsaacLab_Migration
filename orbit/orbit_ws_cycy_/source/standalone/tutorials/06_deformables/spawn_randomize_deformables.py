# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 24.01.17
# Chanyoung Ahn

""" Spawn and randomization deformable objects 
    Testbed for Isaac-Disentangle-franka-v0

.. code-block:: bash

    # Usage
    ./orbit.sh -p source/standalone/tutorials/06_deformables/spawn_randomize_deformables.py --object 07_01 --vis

"""


import argparse
from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--object", type=str, default="07_01", help="Object types: 07_01, 07_005, 07_015")
parser.add_argument("--vis", action="store_true", default=True, help="visualize")

args_cli = parser.parse_args()

# launch omniverse app
simulation_app = SimulationApp({"headless": False})

import numpy as np
import os
import torch
from omni.isaac.core import World
from omni.isaac.core.utils.viewports import set_camera_view

from omni.isaac.core.physics_context.physics_context import PhysicsContext
from omni.isaac.core.prims import RigidPrimView
from pxr import UsdGeom, PhysxSchema, Gf, Usd
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.transformations import get_relative_transform, tf_matrix_from_pose, pose_from_tf_matrix

from omni.isaac.core.utils.prims import get_prim_at_path
import omni.isaac.core.utils.prims as prim_utils
from omni.usd.commands import DeletePrimsCommand
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims.soft.deformable_prim_view import DeformablePrimView
from omni.isaac.orbit_assets import LOCAL_ASSETS_DIR
from omni.isaac.orbit.sensors.camera import Camera, CameraCfg
import omni.isaac.orbit.sim as sim_utils

import omni.usd

class Test():
    def __init__(self):
        self._device = "cuda:0"        

    def init_simulation(self):
        self._scene = PhysicsContext(sim_params={"use_gpu_pipeline": True},
                                     device='cuda:0', backend='torch')
        self._scene.set_broadphase_type("GPU")
        self._scene.enable_gpu_dynamics(flag=True)
        self._scene.enable_ccd(flag=False)
        self._scene.set_gravity(value=0.0)

    def create_object(self):
        if args_cli.object == "07_01":
            prim_utils.create_prim("/World/Object",
                        usd_path=os.path.join(LOCAL_ASSETS_DIR, 'disentangle','test', 'oring_07_01_test.usd'), 
                        translation=(0.0, 0.0, 1))
        elif args_cli.object == "07_005":
            prim_utils.create_prim("/World/Object",
                        usd_path=os.path.join(LOCAL_ASSETS_DIR, 'disentangle', 'train', 'oring_07_005.usd'), 
                        translation=(0.0, 0.0, 0.5))
        elif args_cli.object == "07_015":
            prim_utils.create_prim("/World/Object",
                        usd_path=os.path.join(LOCAL_ASSETS_DIR, 'disentangle', 'train', 'oring_07_015.usd'), 
                        translation=(0.0, 0.0, 0.5))
        else:
            print("Add valid object name: 07_01, 07_005, 07_015")

        self.deformable_body = PhysxSchema.PhysxDeformableBodyAPI(get_prim_at_path("/World/Object/mesh"))
        self.deformable = DeformablePrimView(prim_paths_expr="/World/Object/mesh")
        
    def define_camera(self) -> Camera:
        # sensors; segmentation partially pointcloud
        camera_cfg = CameraCfg(
            prim_path="World/camera",
            # (-x, w, z, -y)
            offset=CameraCfg.OffsetCfg(pos=(2,-2.3,2), rot=(-0.45986, 0.80202, 0.34647, -0.15893)),
            update_period=0,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane", 
                        "normals", "semantic_segmentation",
                        "pointcloud"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 6.0)
            ),
            semantic_labels=["oring"],
        )
        camera = Camera(cfg=camera_cfg)
        
        return camera

    """
    Visualizer functions 
    """
    
    def visualizer_setup(self, color=(1, 0, 0), size=0.05):
        N = 1
        point_list = np.zeros([N, 3])
        sizes = size * np.ones(N)
        stage = omni.usd.get_context().get_stage()
        point = UsdGeom.Points.Define(stage, "/World/pcd")
        point.CreatePointsAttr().Set(point_list)
        point.CreateWidthsAttr().Set(sizes)
        point.CreateDisplayColorPrimvar("constant").Set([color])
        self.points = point



    def main(self):        
        # setting init stage 
        prim_path = '/World/Ground'
        prim_utils.create_prim(prim_path, 
                        usd_path=os.path.join(LOCAL_ASSETS_DIR, 'grid_ground.usd'), 
                        translation=(0.0, 0.0, -2.0))
        prim_utils.create_prim("/World/env", 
                    usd_path=os.path.join(LOCAL_ASSETS_DIR, 'disentangle', 'test_spawn.usd'), 
                    translation=(0.0, 0.0, 0.0))
        
        world = World(stage_units_in_meters=1, backend='torch', device=self._device)
        
        world._physics_context.enable_gpu_dynamics(flag=True)
        self.stage = world.stage
        self.init_simulation()
        self.create_object()
        # self.camera = self.define_camera()
        
        self.init_pole = ArticulationView(prim_paths_expr="/World/env/init_nail_mover")
        init_twist = np.loadtxt(os.path.join(LOCAL_ASSETS_DIR, "disentangle", "oring_07_01.txt"))
        world.scene.add(self.init_pole)
        # world.scene.add(self.camera)
        
        world.reset()
        # self.deformable.initialize()

        pnt = torch.tensor(init_twist, device="cuda:0").reshape(1,-1,3)
        pnt[:,:,-1] -= 1
        world.stop()
        self.deformable.set_simulation_mesh_nodal_positions(pnt)
        world.play()
        if args_cli.vis:
            self.visualizer_setup()
            self.points.GetPointsAttr().Set(init_twist) 
        self.init_pole.initialize()
        
            
        i=0
        while simulation_app.is_running():
            world.step(render=True)
            if world.is_playing():
                if world.current_time_step_index == 0:
                    world.reset()
            i+=1
            if 1 < i < 11: 
                test_joint = self.init_pole.get_joint_positions()
                test_joint[:, 1] -= 0.5 # x
                self.init_pole.set_joint_position_targets(positions=test_joint)
                print("test_joint:", test_joint)
            # self.deformable.set_simulation_mesh_nodal_positions(pnt)

                
if __name__ == "__main__":
    try:
        test = Test()
        test.main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()