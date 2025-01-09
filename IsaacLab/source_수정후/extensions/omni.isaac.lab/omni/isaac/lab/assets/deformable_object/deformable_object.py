# isaaclab/deformableobject.py
# 25.01.09
# Hyeokjun Kwon


from __future__ import annotations
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import omni.log
import omni.physics.tensors.impl.api as physx
from pxr import PhysxSchema, UsdShade
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.markers import VisualizationMarkers
from ..asset_base import AssetBase
from .deformable_object_data import DeformableObjectData
import random 
from omni.isaac.core.utils.transformations import get_relative_transform
from omni.isaac.core.utils.prims import get_prim_at_path


if TYPE_CHECKING:
    from .deformable_object_cfg import DeformableObjectCfg
class DeformableObject(AssetBase):
    """A deformable object asset class."""
    cfg: DeformableObjectCfg
    """Configuration instance for the deformable object."""
    def __init__(self, cfg: DeformableObjectCfg):
        """Initialize the deformable object."""
        super().__init__(cfg)
        self._env_paths = sim_utils.find_matching_prim_paths("/World/envs/env_.*")
        self._env_count = len(self._env_paths)
    """
    Properties
    """
    @property
    def data(self) -> DeformableObjectData:
        return self._data
    @property
    def num_instances(self) -> int:
        return self.root_physx_view.count
    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset."""
        return 1
    @property
    def root_physx_view(self) -> physx.SoftBodyView:
        """Deformable body view for the asset (PhysX)."""
        return self._root_physx_view
    @property
    def material_physx_view(self) -> physx.SoftBodyMaterialView | None:
        """Deformable material view for the asset (PhysX)."""
        return self._material_physx_view
    @property
    def max_sim_elements_per_body(self) -> int:
        """The maximum number of simulation mesh elements per deformable body."""
        return self.root_physx_view.max_sim_elements_per_body
    @property
    def max_collision_elements_per_body(self) -> int:
        """The maximum number of collision mesh elements per deformable body."""
        return self.root_physx_view.max_elements_per_body
    @property
    def max_sim_vertices_per_body(self) -> int:
        """The maximum number of simulation mesh vertices per deformable body."""
        return self.root_physx_view.max_sim_vertices_per_body
    @property
    def max_collision_vertices_per_body(self) -> int:
        """The maximum number of collision mesh vertices per deformable body."""
        return self.root_physx_view.max_vertices_per_body
    """
    Operations.
    """
    def reset(self, env_ids: Sequence[int] | None = None):
        # resolve all indices
        if env_ids is None:
            env_ids = slice(None)
        # randomly assign the initial configuration
        rand_ints = self._set_simulation_mesh_nodal_positions(self._default_positions, env_ids)
        # Reset all internal buffers to default
        self._data.reset(env_ids)

    def write_data_to_sim(self):
        pass
    def update(self, dt: float):
        self._data.update(dt)
    """
    Operations - Write to simulation.
    """
    def write_nodal_state_to_sim(self, nodal_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the nodal state over selected environment indices into the simulation."""
        # set into simulation
        self.write_nodal_pos_to_sim(nodal_state[..., :3], env_ids=env_ids)
        self.write_nodal_velocity_to_sim(nodal_state[..., 3:], env_ids=env_ids)
    def write_nodal_pos_to_sim(self, nodal_pos: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the nodal positions over selected environment indices into the simulation."""
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.nodal_pos_w[env_ids] = nodal_pos.clone()
        # set into simulation
        self.root_physx_view.set_sim_nodal_positions(self._data.nodal_pos_w, indices=physx_env_ids)
    def write_nodal_velocity_to_sim(self, nodal_vel: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the nodal velocity over selected environment indices into the simulation."""
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.nodal_vel_w[env_ids] = nodal_vel.clone()
        # set into simulation
        self.root_physx_view.set_sim_nodal_velocities(self._data.nodal_vel_w, indices=physx_env_ids)
    def write_nodal_kinematic_target_to_sim(self, targets: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the kinematic targets of the simulation mesh for the deformable bodies indicated by the indices."""
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # store into internal buffers
        self._data.nodal_kinematic_target[env_ids] = targets.clone()
        # set into simulation
        self.root_physx_view.set_sim_kinematic_targets(self._data.nodal_kinematic_target, indices=physx_env_ids)
    """
    Operations - Helper.
    """
    def transform_nodal_pos(
        self, nodal_pos: torch.tensor, pos: torch.Tensor | None = None, quat: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Transform the nodal positions based on the pose transformation."""
        # offset the nodal positions to center them around the origin
        mean_nodal_pos = nodal_pos.mean(dim=1, keepdim=True)
        nodal_pos = nodal_pos - mean_nodal_pos
        # transform the nodal positions based on the pose around the origin
        return math_utils.transform_points(nodal_pos, pos, quat) + mean_nodal_pos
    """
    Internal helper.
    """
    def _set_simulation_mesh_nodal_positions(
        self,
        positions: torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids = self._env_count
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
    ) -> torch.Tensor:
        """Gets the nodal positions of the simulation mesh for the deformable bodies indicated by the indices."""
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        positions = torch.zeros_like(
            [env_ids.shape[0], self.max_sim_vertices_per_body, 3], dtype="float32", device=self._device
        )
        write_idx = 0
        for i in env_ids:
            points = self._deformable_body_apis[i].GetSimulationPointsAttr().Get()
            if points is None:
                raise Exception(f"The prim {self.name} does not have points attribute.")
            positions[write_idx] = torch.tensor(
                points, dtype="float32", device=self._device
            ).view(self.max_sim_vertices_per_body, 3)
            write_idx += 1
        return positions
    def _get_collision_mesh_nodal_position(
        self, index: int) -> torch.Tensor:
        # GetCollisionPointsAttr
        local_collision_point = self._deformable_body_apis[index].GetCollisionPointsAttr().Get()
        if local_collision_point is None:
            raise Exception(f"The prim deformable does not have points attribute.")
        vertices = torch.tensor(local_collision_point, dtype="float32")
        vertices_tf_row_major = torch.nn.functional.pad(vertices, (0, 1), "constant", 1.0)
        relative_tf_column_major = get_relative_transform(get_prim_at_path("/World/Object/mesh"), get_prim_at_path("/World"))
        relative_tf_row_major = torch.tensor(relative_tf_column_major, dtype="float32").t()
        points_in_relative_coord = torch.matmul(vertices_tf_row_major, relative_tf_row_major)
        position = torch.tensor(points_in_relative_coord[:, :-1], dtype="float32",
                                            device=self._device).view(self.max_collision_vertices_per_body, 3)
        return position
    def get_success_checker(self, env_ids: Sequence[int] | None = None,
                            clone: bool = True ) -> torch.Tensor:
        # -- env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        # Success checker
        success_status = []
        for i in env_ids:
            # select points env, 32 from _get_collision_mesh_nodal_position
            collision_position = self._get_collision_mesh_nodal_position(i)
            # success checker
            # return [env, 1]
            if collision_position.min() > 0.1 :
                success_status.append(1.0)
            else :
                success_status.append(0.0)
        success_tensor = torch.tensor(success_status, dtype=torch.float32, device=self.device).view(-1, 1)
        return success_tensor
    def _initialize_impl(self):
        # create simulation view
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString
        # find deformable root prims
        root_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path, predicate=lambda prim: prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI)
        )
        if len(root_prims) == 0:
            raise RuntimeError(
                f"Failed to find a deformable body when resolving '{self.cfg.prim_path}'."
                " Please ensure that the prim has 'PhysxSchema.PhysxDeformableBodyAPI' applied."
            )
        if len(root_prims) > 1:
            raise RuntimeError(
                f"Failed to find a single deformable body when resolving '{self.cfg.prim_path}'."
                f" Found multiple '{root_prims}' under '{template_prim_path}'."
                " Please ensure that there is only one deformable body in the prim path tree."
            )
        # we only need the first one from the list
        root_prim = root_prims[0]
        # find deformable material prims
        material_prim = None
        # obtain material prim from the root prim
        # note: here we assume that all the root prims have their material prims at similar paths
        #   and we only need to find the first one. This may not be the case for all scenarios.
        #   However, the checks in that case get cumbersome and are not included here.
        if root_prim.HasAPI(UsdShade.MaterialBindingAPI):
            # check the materials that are bound with the purpose 'physics'
            material_paths = UsdShade.MaterialBindingAPI(root_prim).GetDirectBindingRel("physics").GetTargets()
            # iterate through targets and find the deformable body material
            if len(material_paths) > 0:
                for mat_path in material_paths:
                    mat_prim = root_prim.GetStage().GetPrimAtPath(mat_path)
                    if mat_prim.HasAPI(PhysxSchema.PhysxDeformableBodyMaterialAPI):
                        material_prim = mat_prim
                        break
        if material_prim is None:
            omni.log.info(
                f"Failed to find a deformable material binding for '{root_prim.GetPath().pathString}'."
                " The material properties will be set to default values and are not modifiable at runtime."
                " If you want to modify the material properties, please ensure that the material is bound"
                " to the deformable body."
            )
        # resolve root path back into regex expression
        # -- root prim expression
        root_prim_path = root_prim.GetPath().pathString
        root_prim_path_expr = self.cfg.prim_path + root_prim_path[len(template_prim_path) :]
        # -- object view
        self._root_physx_view = self._physics_sim_view.create_soft_body_view(root_prim_path_expr.replace(".*", "*"))
        # Return if the asset is not found
        if self._root_physx_view._backend is None:
            raise RuntimeError(f"Failed to create deformable body at: {self.cfg.prim_path}. Please check PhysX logs.")
        # resolve material path back into regex expression
        if material_prim is not None:
            # -- material prim expression
            material_prim_path = material_prim.GetPath().pathString
            # check if the material prim is under the template prim
            # if not then we are assuming that the single material prim is used for all the deformable bodies
            if template_prim_path in material_prim_path:
                material_prim_path_expr = self.cfg.prim_path + material_prim_path[len(template_prim_path) :]
            else:
                material_prim_path_expr = material_prim_path
            # -- material view
            self._material_physx_view = self._physics_sim_view.create_soft_body_material_view(
                material_prim_path_expr.replace(".*", "*")
            )
        else:
            self._material_physx_view = None
        # log information about the deformable body
        omni.log.info(f"Deformable body initialized at: {root_prim_path_expr}")
        omni.log.info(f"Number of instances: {self.num_instances}")
        omni.log.info(f"Number of bodies: {self.num_bodies}")
        if self._material_physx_view is not None:
            omni.log.info(f"Deformable material initialized at: {material_prim_path_expr}")
            omni.log.info(f"Number of instances: {self._material_physx_view.count}")
        else:
            omni.log.info("No deformable material found. Material properties will be set to default values.")
        # container for data access
        self._data = DeformableObjectData(self.root_physx_view, self.device)
        # create buffers
        self._create_buffers()
        # update the deformable body data
        self.update(0.0)
    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)
        # default state
        # we use the initial nodal positions at spawn time as the default state
        # note: these are all in the simulation frame
        nodal_positions = self.root_physx_view.get_sim_nodal_positions()
        nodal_velocities = torch.zeros_like(nodal_positions)
        self._data.default_nodal_state_w = torch.cat((nodal_positions, nodal_velocities), dim=-1)
        self._default_positions = self.get_simulation_mesh_nodal_positions()
        # kinematic targets
        self._data.nodal_kinematic_target = self.root_physx_view.get_sim_kinematic_targets()
        # set all nodes as non-kinematic targets by default
        self._data.nodal_kinematic_target[..., -1] = 1.0
        self._deformable_body_apis = [None] * self.num_instances
        for index in range(self.num_instances):
            prim = self._root_physx_view.get_prim_at_index(index)
            if prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI):
                api = PhysxSchema.PhysxDeformableBodyAPI(prim)
            else:
                api = PhysxSchema.PhysxDeformableBodyAPI.Apply(prim)
            self._deformable_body_apis[index] = api
    """
    Internal simulation callbacks.
    """
    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)
    def _debug_vis_callback(self, event):
        # check where to visualize
        targets_enabled = self.data.nodal_kinematic_target[:, :, 3] == 0.0
        num_enabled = int(torch.sum(targets_enabled).item())
        # get positions if any targets are enabled
        if num_enabled == 0:
            # create a marker below the ground
            positions = torch.tensor([[0.0, 0.0, -10.0]], device=self.device)
        else:
            positions = self.data.nodal_kinematic_target[targets_enabled][..., :3]
        # show target visualizer
        self.target_visualizer.visualize(positions)
    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._physics_sim_view = None
        self._root_physx_view = None
