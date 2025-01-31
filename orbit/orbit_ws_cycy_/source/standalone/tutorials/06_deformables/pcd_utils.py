# Pointcloud utils
# 23.10.12

import numpy as np
import math
import trimesh as t
import open3d as o3d
import torch
from pxr import UsdGeom, Usd, Gf, PhysicsSchemaTools, Sdf, PhysxSchema
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import get_relative_transform

class PointCloudUtil():
    def __init__(
        self,
        mesh_path):
        self.mesh_path = mesh_path
        self.initalize_points(mesh_path)
        return
    
    def initalize_points(self, mesh_path):
        """
        Get Deformable Mesh's points.
        """
        mesh_prim = get_prim_at_path(mesh_path)
        self.deformable = PhysxSchema.PhysxDeformableBodyAPI(mesh_prim)

    def get_number_of_points(self):
        positions = self.get_position_array()
        return len(positions)
    
    def get_position_array(self):
        local_collision_point = (np.array(self.deformable.GetCollisionPointsAttr().Get())) 
        vertices = np.array(local_collision_point)
        vertices_tf_row_major = np.pad(vertices, ((0, 0), (0, 1)), constant_values=1.0)
        relative_tf_column_major = get_relative_transform(get_prim_at_path(self.mesh_path), get_prim_at_path("/World"))
        relative_tf_row_major = np.transpose(relative_tf_column_major)
        points_in_relative_coord = vertices_tf_row_major @ relative_tf_row_major
        pcd = points_in_relative_coord[:, :-1]
        # pcds.append(pcd)

        return pcd

    def get_number_of_inside_points(self):
        trigger = UsdGeom.Mesh(get_current_stage().GetPrimAtPath("/World/trigger"))
        trig_vert = np.array(trigger.GetPointsAttr().Get())
        ###
        vertices = trig_vert
        vertices_tf_row_major = np.pad(vertices, ((0, 0), (0, 1)), constant_values=1.0)
        relative_tf_column_major = get_relative_transform(get_prim_at_path("/World/trigger"), 
                                                        get_prim_at_path("/World"))
        relative_tf_row_major = np.transpose(relative_tf_column_major)

        points_in_relative_coord = vertices_tf_row_major @ relative_tf_row_major
        points_in_meters = points_in_relative_coord[:, :-1]
        ###
        trig_bbox = t.PointCloud(points_in_meters).bounding_box
        # contact_check = np.array([item for item in trig_bbox.contains(self.get_position_array())])
        contact_check = trig_bbox.contains(self.get_position_array())
        
        return len(contact_check[contact_check == True])

    def get_chamfer_distance(self, raw_pcds, target_pcds):
        """
        Check distance(chamfer distanse) between object node and target_node. 
        Args:

        INPUT
            object_node [num_envs*N*3](npy, np.array): 
            target_nodes [num_envs*N*3](npy, np.array): 
            
        OUTPUT
            chamfer_dist [num_envs*1](npy, np.array): each node's chamfer distances.    
        """
        chamfer_dists = []
        for i, raw_pcd in enumerate(raw_pcds):    
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(raw_pcd)
            
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_pcds[i])
            # o3d.visualization.draw_geometries([object_pcd, target_pcd])

            cham_dist = np.asarray(object_pcd.compute_point_cloud_distance(target_pcd)).sum() # chamfer distance
            chamfer_dists.append(cham_dist)
 
        return chamfer_dists