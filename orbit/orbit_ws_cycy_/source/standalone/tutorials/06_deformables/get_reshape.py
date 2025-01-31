# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 24.02.19
# Chanyoung Ahn

"""This script demonstrates provide get/spawn mesh point of deformables for reset / initialization.
    If you want to add new shape of deformables, save nodal_mesh_position.
    
    Initialization D.O. FOR RESHAPE Task
    
.. code-block:: bash

    # Usage
    ./orbit.sh -p source/standalone/tutorials/06_deformables/get_reshape.py --vis

"""


import argparse
from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--object", type=str, default="005_07", help="Object types: 015_07, 015_05, 010_07, 010_05, 005_07, 005_05")
parser.add_argument("--vis", action="store_true", default=True, help="visualize")

args_cli = parser.parse_args()

# launch omniverse app
simulation_app = SimulationApp({"headless": False})

import numpy as np
import os
import torch
import carb
from omni.isaac.core import World
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.core.physics_context.physics_context import PhysicsContext
import omni.isaac.core.utils.prims as prim_utils
from pxr import UsdGeom, PhysxSchema, Gf, Usd
from omni.isaac.core.prims.soft.deformable_prim_view import DeformablePrimView
from omni.isaac.orbit_assets import LOCAL_ASSETS_DIR
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.utils.transformations import get_relative_transform
import omni.usd

def init_simulation():
    _scene = PhysicsContext(sim_params={"use_gpu_pipeline": False, 
                                                 "use_gpu": True, 
                                                 "device": "cuda", 
                                                 "use_flatcache": False})
    _scene.set_broadphase_type("GPU")
    _scene.enable_gpu_dynamics(flag=True)


def visualizer_setup(color=(1, 0, 0), size=0.001):
    N = 1
    point_list = np.zeros([N, 3])
    sizes = size * np.ones(N)
    stage = omni.usd.get_context().get_stage()
    point = UsdGeom.Points.Define(stage, "/World/pcd")
    point.CreatePointsAttr().Set(point_list)
    point.CreateWidthsAttr().Set(sizes)
    point.CreateDisplayColorPrimvar("constant").Set([color])
    return point


def main():        
    # setting init stage 
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    
    # -- Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    
    prim_utils.create_prim(
        usd_path=f"{LOCAL_ASSETS_DIR}/reshape/train/oring_{args_cli.object}.usd",
        prim_path="/World/Object",
        translation=np.asarray([0., 0., 0.3]),
        semantic_label="oring",
    )
    
    world = World(stage_units_in_meters=1, backend='torch')
    world._physics_context.enable_gpu_dynamics(flag=True)
    stage = world.stage
    init_simulation()
    
    deformable_body = PhysxSchema.PhysxDeformableBodyAPI(prim_utils.get_prim_at_path("/World/Object/mesh"))
    # deformable
    # deformable_body = PhysxSchema.PhysxDeformableBodyAPI(get_prim_at_path("/World/Object/mesh"))_view = DeformablePrimView(prim_paths_expr="/World/Object/mesh")
    # world.scene.add(deformable_view)
    
    world.reset()
    print("[INFO]: Setup complete...")
    
    if args_cli.vis:
        point = visualizer_setup()
    i=0
    
    while simulation_app.is_running():
        world.step()
        i+=1
        

        
        pnts = np.array(deformable_body.GetSimulationPointsAttr().Get())
        
        if args_cli.vis:
            vertices_tf_row_major = np.pad(pnts, ((0, 0), (0, 1)), constant_values=1.0)
            relative_tf_column_major = get_relative_transform(prim_utils.get_prim_at_path("/World/Object/mesh"), 
                                                            prim_utils.get_prim_at_path("/World"))
            relative_tf_row_major = np.transpose(relative_tf_column_major)
            points_in_relative_coord = vertices_tf_row_major @ relative_tf_row_major
            pcd = points_in_relative_coord[:, :-1]
            
            point.GetPointsAttr().Set(pcd) 
            
            
        if i == 50:
            i=0
            pnts[:,-1] += 10
            # # change this name
            np.savetxt(os.path.join(LOCAL_ASSETS_DIR, "reshape", "spawn", f"oring_{args_cli.object}.txt"), pnts)
            print("SAVE THIS STATE") 

            
                
if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        import traceback
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()