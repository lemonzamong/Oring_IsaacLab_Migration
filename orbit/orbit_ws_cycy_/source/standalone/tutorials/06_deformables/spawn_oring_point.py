# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 24.02.19
# Chanyoung Ahn

"""This script demonstrates provide get/spawn mesh point of deformables for reset / initialization.

.. code-block:: bash

    # Usage
    ./orbit.sh -p source/standalone/tutorials/06_deformables/spawn_oring_point.py 

"""


import argparse
from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--object", type=str, default="015_07", help="Object types: 015_07, 015_05, 010_07, 010_05, 005_07, 005_05")
parser.add_argument("--vis", action="store_true", default=True, help="visualize")

args_cli = parser.parse_args()

# launch omniverse app
simulation_app = SimulationApp({"headless": False})

import numpy as np
import os
import torch
from omni.isaac.core import World
import omni.isaac.orbit.sim as sim_utils

from omni.isaac.core.physics_context.physics_context import PhysicsContext
from pxr import UsdGeom, PhysxSchema, Gf, Usd
import omni.isaac.core.utils.prims as prim_utils
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
        # self._scene.set_gravity(value=0.0)

    def visualizer_setup(self, color=(1, 0, 0), size=0.001):
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
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/defaultGroundPlane", cfg)
        
        # -- Lights
        cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        cfg.func("/World/Light", cfg)
        
        prim_utils.create_prim(
        usd_path=f"{LOCAL_ASSETS_DIR}/reshape/train/oring_{args_cli.object}.usd",
        prim_path="/World/Object",
        translation=np.asarray([0., 0., 0.0]),
        semantic_label="oring",
        )
        world = World(stage_units_in_meters=1, backend='torch')
        
        world._physics_context.enable_gpu_dynamics(flag=True)
        self.stage = world.stage
        self.init_simulation()
        
        deformable_body = PhysxSchema.PhysxDeformableBodyAPI(prim_utils.get_prim_at_path("/World/Object/mesh"))
            # np.savetxt(os.path.join(LOCAL_ASSETS_DIR, "reshape", "spawn", f"oring_{args_cli.object}.txt"), pnts)
        init_twist = np.loadtxt(os.path.join(LOCAL_ASSETS_DIR, "reshape", "spawn",  f"oring_{args_cli.object}.txt"))

        world.reset()
        # self.deformable.initialize()

        # pnt = torch.tensor(init_twist, device="cuda:0").reshape(1,-1,3)
        # pnt[:,:,-1] -= 1
        world.stop()
        deformable_body.GetSimulationPointsAttr().Set(init_twist)
        # self.deformable.set_simulation_mesh_nodal_positions(init_twist)
        world.play()

        if args_cli.vis:
            self.visualizer_setup()
            self.points.GetPointsAttr().Set(init_twist/1000) 
        i=0
        while simulation_app.is_running():
            world.step(render=True)
            if world.is_playing():
                if world.current_time_step_index == 0:
                    world.reset()
            i += 1
            if i == 200:     
                i=0   
                world.stop()
                deformable_body.GetSimulationPointsAttr().Set(init_twist)
                world.play()
                print('reset')
            

                
if __name__ == "__main__":
    try:
        test = Test()
        test.main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()