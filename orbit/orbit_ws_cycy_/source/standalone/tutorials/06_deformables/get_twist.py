# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 24.01.17
# Chanyoung Ahn

"""This script demonstrates provide get/spawn mesh point of deformables for reset / initialization.
    If you want to add new shape of deformables, save nodal_mesh_position.
    FOR Disentangle task.
.. code-block:: bash

    # Usage
    ./orbit.sh -p source/standalone/tutorials/06_deformables/get_deformable_point.py --object 07_01 --vis

"""


import argparse
from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--object", type=str, default="07_01", help="Object types: 07_01, 07_005, 07_015")
parser.add_argument("--vis", action="store_true", default=False, help="visualize")

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
from omni.isaac.core.utils.stage import open_stage

from omni.usd.commands import DeletePrimsCommand
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims.soft.deformable_prim_view import DeformablePrimView
from omni.isaac.orbit_assets import LOCAL_ASSETS_DIR

import omni.usd

class Test():
    def __init__(self):
        self._device = "cuda:0"        

    def init_simulation(self):
        self._scene = PhysicsContext(sim_params={"use_gpu_pipeline": True},
                                     device='cuda', backend='torch')
        self._scene.set_broadphase_type("GPU")
        self._scene.enable_gpu_dynamics(flag=True)
        self._scene.enable_ccd(flag=False)

    def create_object(self):
        if args_cli.object == "07_01":
            prim_utils.create_prim("/World/Object",
                        usd_path=os.path.join(LOCAL_ASSETS_DIR, 'disentangle','test', 'oring_07_01_test.usd'), 
                        translation=(0.0, 0.0, 0.5))
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
        
    def get_position_array(self):
        local_collision_point = (np.array(self.deformable_body.GetCollisionPointsAttr().Get())) 
        vertices = np.array(local_collision_point)
        vertices_tf_row_major = np.pad(vertices, ((0, 0), (0, 1)), constant_values=1.0)
        relative_tf_column_major = get_relative_transform(get_prim_at_path("/World/Object/mesh"), get_prim_at_path("/World"))
        relative_tf_row_major = np.transpose(relative_tf_column_major)
        points_in_relative_coord = vertices_tf_row_major @ relative_tf_row_major
        pcd = points_in_relative_coord[:, :-1]
        return pcd

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
        path = os.path.join(LOCAL_ASSETS_DIR, 'disentangle', 'test.usd')
        open_stage(usd_path=path)
        
        world = World(stage_units_in_meters=1, backend='torch')
        
        world._physics_context.enable_gpu_dynamics(flag=True)
        self.stage = world.stage
        self.init_simulation()
        world.reset()
        # self.create_object()
        self.deformable = DeformablePrimView(prim_paths_expr="/World/oring_07_01/mesh")
        world.scene.add(self.deformable)
        
        # world.reset()
        self.deformable.initialize()
        
        self.visualizer_setup()
        if args_cli.vis:
            self.visualizer_setup()
        i=0
        while simulation_app.is_running():
            world.step(render=True)
            if world.is_playing():
                if world.current_time_step_index == 0:
                    world.reset()
            i += 1
            
            if i == 200:
                pnts = self.deformable.get_simulation_mesh_nodal_positions()
                self.points.GetPointsAttr().Set(pnts[0].cpu().numpy()) 
                np.savetxt(os.path.join(LOCAL_ASSETS_DIR, "disentangle", "spawn","oring_07_01_2.txt"), pnts[0].cpu())

                print("SAVE") 
                simulation_app.close()

            
                
if __name__ == "__main__":
    try:
        test = Test()
        test.main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()