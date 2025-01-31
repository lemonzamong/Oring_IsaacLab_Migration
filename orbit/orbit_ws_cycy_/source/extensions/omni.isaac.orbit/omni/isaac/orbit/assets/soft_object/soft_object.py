# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
from typing import List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import carb
import omni.physics.tensors.impl.api as physx
# from pxr import UsdPhysics
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade, Vt

import omni.kit.app
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
import omni.isaac.orbit.utils.string as string_utils
from omni.isaac.core.utils.transformations import get_relative_transform, tf_matrix_from_pose, pose_from_tf_matrix
from omni.isaac.core.utils.prims import (
    find_matching_prim_paths,
    get_prim_at_path,
    get_prim_parent,
    is_prim_non_root_articulation_link,
    is_prim_path_valid,
)
from omni.isaac.core.utils.xforms import get_local_pose, get_world_pose

from ..asset_base import AssetBase
from .soft_object_data import SoftObjectData

if TYPE_CHECKING:
    from .soft_object_cfg import SoftObjectCfg
import random 

# 2024.01.17
# Chanyoung Ahn

# TODO: 
# Add complete pcd / update each step # 30518 vs (651)... [x]
# Add success checker / call final step                 [x]
# Add randmization object / call reset part             [o]
# USE /home/scorpius/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1/extsPhysics/omni.physics.tensors-105.1.10-5.1/omni/physics/tensors/impl/api.py
# 

class SoftObject(AssetBase):
    """A soft object asset class.

    Soft objects are assets comprising of deformable bodies. They can be used to represent dynamic objects
    such as any mesh type deformables 

    For an asset to be considered a soft object, the root prim of the asset must have the `PhysxSchema PhysxDeformableBodyAPI`_
    applied to it. This API is used to define the simulation properties of the soft body. 

    .. note::
        If you want to set deformable node when reset environment, install isaac-sim 2023.1.0 > version.
    """

    cfg: SoftObjectCfg
    """Configuration instance for the rigid object."""

    def __init__(self, cfg: SoftObjectCfg):
        """Initialize the rigid object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        # container for data access
        self._data = SoftObjectData()
      
        

    """
    Properties
    """

    @property
    def data(self) -> SoftObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self.root_physx_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset."""
        return 1

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in articulation."""
        prim_paths = self.body_physx_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def root_physx_view(self) -> physx.SoftBodyView:
        """Rigid body view for the asset (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_physx_view

    @property
    def physics_view(self) -> physx.SoftBodyView:
        """View for the bodies in the asset (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._physics_view

    # verified properies (from deformable_prim_view.py)

    @property
    def max_simulation_mesh_elements_per_body(self) -> int:
        """
        Returns:
            int: maximum number of simulation mesh elements per deformable body.
        """
        return self._max_simulation_mesh_elements_per_body

    @property
    def max_simulation_mesh_vertices_per_body(self) -> int:
        """
        Returns:
            int: maximum number of simulation mesh vertices per deformable body.
        """
        return self._max_simulation_mesh_vertices_per_body

    @property
    def max_collision_mesh_elements_per_body(self) -> int:
        """
        Returns:
            int: maximum number of collision mesh elements per deformable body.
        """
        return self._max_collision_mesh_elements_per_body

    @property
    def max_collision_mesh_vertices_per_body(self) -> int:
        """
        Returns:
            int: maximum number of collision mesh vertices per deformable body.
        """
        return self._max_collision_mesh_vertices_per_body

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # resolve all indices
        if env_ids is None:
            env_ids = slice(None)
        # # reset external wrench
        # self._external_force_b[env_ids] = 0.0
        # self._external_torque_b[env_ids] = 0.0
        # # reset last body vel
        # self._last_body_vel_w[env_ids] = 0.0
        
    def write_data_to_sim(self):
        """Write external wrench to the simulation.

        Note:
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        pass
        # write external wrench
        # if self.has_external_wrench:
        #     self.body_physx_view.apply_forces_and_torques_at_position(
        #         force_data=self._external_force_b.view(-1, 3),
        #         torque_data=self._external_torque_b.view(-1, 3),
        #         position_data=None,
        #         indices=self._ALL_BODY_INDICES,
        #         is_global=False,
        #     )
        
    def update(self, dt: float):
        # -- root-state (note: we roll the quaternion to match the convention used in Isaac Sim -- wxyz)
        # self._data.root_state_w[:, :7] = self.root_physx_view.get_transforms()
        # self._data.root_state_w[:, 3:7] = math_utils.convert_quat(self._data.root_state_w[:, 3:7], to="wxyz")
        # self._data.root_state_w[:, 7:] = self.root_physx_view.get_velocities()
        # -- update common data
        # self._update_common_data(dt)
        pass

    def find_bodies(self, name_keys: str | Sequence[str]) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Please check the :meth:`omni.isaac.orbit.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names)


    """
    Operations - Setters.
    """
    # def set_initialization_oring_points()
    
    def set_simulation_mesh_nodal_positions(
        self,
        positions: Optional[Union[np.ndarray, torch.Tensor]],
        env_ids: Sequence[int] | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = len(self._env_paths) 
            
        rand_max = len(positions)
        position = positions[0].repeat(env_ids, 1, 1).clone()
        rand_ints = torch.randint(0, rand_max, (env_ids,)) 
        
        for i in range(env_ids):
            env_position, _ = get_world_pose(self._env_paths[i])
            position[i] = positions[rand_ints[i]].clone() + torch.tensor(env_position, device=self._device)
            # print(f"env_{i} reset {rand_ints[i]}")
        self.root_physx_view.set_sim_nodal_positions(position, self._ALL_INDICES)
        return rand_ints
    
    def get_simulation_mesh_nodal_positions(
        self, env_ids: Sequence[int] | None = None, clone: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """Gets the nodal positions of the simulation mesh for the deformable bodies indicated by the indices.

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor]], optional): indices to specify which deformable prims to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            clone (bool, optional): True to return a clone of the internal buffer. Otherwise False. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]: position tensor with shape (M, max_simulation_mesh_vertices_per_body, 3)
        """
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
            
        positions = torch.zeros_like(
            [env_ids.shape[0], self.max_simulation_mesh_vertices_per_body, 3], dtype="float32", device=self._device
        )
        write_idx = 0
        for i in env_ids:
            self._apply_deformable_body_api(i.tolist())
            points = self._deformable_body_apis[i.tolist()].GetSimulationPointsAttr().Get()
            if points is None:
                raise Exception(f"The prim {self.name} does not have points attribute.")
            
            positions[write_idx] = torch.tensor(
                points, dtype="float32", device=self._device
            ).view(self.max_simulation_mesh_vertices_per_body, 3)
            write_idx += 1
        return positions
    
    def _get_collision_mesh_nodal_position(
        self, index: int) -> Union[np.ndarray, torch.Tensor]:
        # GetCollisionPointsAttr
        # FIX 271
        self._apply_deformable_body_api(index.tolist())
        
        local_collision_point = self._deformable_body_apis[index.tolist()].GetCollisionPointsAttr().Get()
        if local_collision_point is None:
            raise Exception(f"The prim deformable does not have points attribute.")
            
        vertices = torch.tensor(local_collision_point, dtype="float32")
        vertices_tf_row_major = torch.nn.functional.pad(vertices, (0, 1), "constant", 1.0)
        relative_tf_column_major = get_relative_transform(get_prim_at_path("/World/Object/mesh"), get_prim_at_path("/World"))
        relative_tf_row_major = torch.tensor(relative_tf_column_major, dtype="float32").t()
        points_in_relative_coord = torch.matmul(vertices_tf_row_major, relative_tf_row_major)

        position = torch.tensor(points_in_relative_coord[:, :-1], dtype="float32",
                                            device=self._device).view(self.max_collision_mesh_vertices_per_body, 3)

        return position
    
    def get_success_checker(self, env_ids: Sequence[int] | None = None, 
                            clone: bool = True ) -> Union[np.ndarray, torch.Tensor]:
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
            
        # Success checker
        
        # select points env, 32 from _get_collision_mesh_nodal_position
        # success checker
        # return [env, 1]
    
    """
    Internal helper.
    """
    def _apply_deformable_body_api(self, index):
        # Will remove
        if self._deformable_body_apis[index] is None:
            
            if self._prims[index].HasAPI(PhysxSchema.PhysxDeformableBodyAPI):
                api = PhysxSchema.PhysxDeformableBodyAPI(self._prims[index])
            else:
                api = PhysxSchema.PhysxDeformableBodyAPI.Apply(self._prims[index])
            self._deformable_body_apis[index] = api

    def _initialize_impl(self):
        # create simulation view
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        carb.log_info("initializing view for {}".format("deformable")) # Will Fix
        if not carb.settings.get_settings().get_as_bool("/physics/suppressReadback"):
            carb.log_error(
                "Using Deformable body requires the gpu pipeline or (a World initialized with a cuda device)"
            )
        
        # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
        # template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        root_prim_path_expr = template_prim.GetPath().pathString
        
        self._prim_paths = find_matching_prim_paths(self.cfg.prim_path)
        self._prims = []
        
        self._env_paths = find_matching_prim_paths("/World/envs/env_.*")
        for prim_path in self._prim_paths:
            self._prims.append(get_prim_at_path(prim_path+"/mesh"))

        # -- object view
        self._root_physx_view = self._physics_sim_view.create_soft_body_view(self.cfg.prim_path.replace(".*", "*")+"/mesh")
        # self._root_physx_view = self._physics_sim_view.create_soft_body_view(root_prim_path_expr.replace(".*", "*")+"/mesh")
        self._physics_view = self._root_physx_view
        # self._physcis_vies = sle
        # log information about the articulation
        # self._count = self._physics_view.count
        self._max_collision_mesh_elements_per_body = self._physics_view.max_elements_per_body
        self._max_collision_mesh_vertices_per_body = self._physics_view.max_vertices_per_body
        self._max_simulation_mesh_elements_per_body = self._physics_view.max_sim_elements_per_body
        self._max_simulation_mesh_vertices_per_body = self._physics_view.max_sim_vertices_per_body
        carb.log_info("Soft Object with Device: {}".format(self._device))
        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)
        self._deformable_body_apis = [None] * self.num_instances
        self._deformable_apis = [None] * self.num_instances
        # asset data
        # TODO: X Asset simulation data ? only use reset... not step
        # Asset Simulation position
        # TODO: GET partially pcd from carmera? > from sensor data....
        
        # self._ALL_BODY_INDICES = torch.arange(self.body_physx_view.count, dtype=torch.long, device=self.device)
        # self.GRAVITY_VEC_W = torch.tensor((0.0, 0.0, -1.0), device=self.device).repeat(self.num_instances, 1)
        # self.FORWARD_VEC_B = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self.num_instances, 1)
        # # external forces and torques
        # self.has_external_wrench = False
        # self._external_force_b = torch.zeros((self.num_instances, self.num_bodies, 3), device=self.device)
        # self._external_torque_b = torch.zeros_like(self._external_force_b)

        # # asset data
        # # -- properties
        # self._data.body_names = self.body_names
        # # -- root states
        # self._data.root_state_w = torch.zeros(self.num_instances, 13, device=self.device)

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        # default state
        # -- root state
        # note: we cast to tuple to avoid torch/numpy type mismatch.
        # default_root_state = (
        #     tuple(self.cfg.init_state.pos)
        #     + tuple(self.cfg.init_state.rot)
        #     + tuple(self.cfg.init_state.lin_vel)
        #     + tuple(self.cfg.init_state.ang_vel)
        # )
        # default_root_state = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
        # self._data.default_root_state = default_root_state.repeat(self.num_instances, 1)
        pass
    
    # def _update_common_data(self, dt: float):
    #     """Update common quantities related to rigid objects.

    #     Note:
    #         This has been separated from the update function to allow for the child classes to
    #         override the update function without having to worry about updating the common data.
    #     """
    #     # -- body-state (note: we roll the quaternion to match the convention used in Isaac Sim -- wxyz)
    #     self._data.body_state_w[..., :7] = self.body_physx_view.get_transforms().view(-1, self.num_bodies, 7)
    #     self._data.body_state_w[..., 3:7] = math_utils.convert_quat(self._data.body_state_w[..., 3:7], to="wxyz")
    #     self._data.body_state_w[..., 7:] = self.body_physx_view.get_velocities().view(-1, self.num_bodies, 6)
    #     # -- body acceleration
    #     self._data.body_acc_w[:] = (self._data.body_state_w[..., 7:] - self._last_body_vel_w) / dt
    #     self._last_body_vel_w[:] = self._data.body_state_w[..., 7:]
    #     # -- root state in body frame
    #     self._data.root_vel_b[:, 0:3] = math_utils.quat_rotate_inverse(
    #         self._data.root_quat_w, self._data.root_lin_vel_w
    #     )
    #     self._data.root_vel_b[:, 3:6] = math_utils.quat_rotate_inverse(
    #         self._data.root_quat_w, self._data.root_ang_vel_w
    #     )
    #     self._data.projected_gravity_b[:] = math_utils.quat_rotate_inverse(self._data.root_quat_w, self.GRAVITY_VEC_W)
    #     # -- heading direction of root
    #     forward_w = math_utils.quat_apply(self._data.root_quat_w, self.FORWARD_VEC_B)
    #     self._data.heading_w[:] = torch.atan2(forward_w[:, 1], forward_w[:, 0])

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._physics_sim_view = None
        self._root_physx_view = None
        self._body_physx_view = None
