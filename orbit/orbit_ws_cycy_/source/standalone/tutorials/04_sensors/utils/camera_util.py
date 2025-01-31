# Chanyoung Ahn
# 24.01.26
# Camera utils

import omni.kit.commands
import omni.usd
import random
import math
import numpy as np
import re
import torch
from tensordict import TensorDict
from typing import Any, Sequence


from omni.isaac.core.prims import XFormPrimView
import omni.replicator.core as rep
from pxr import UsdGeom
# if you add other functions for camera, use "UsdGeom.__init__.pyi "Camera" Class."

from .camera_cfg import CameraCfg
from .camera_data import CameraData
from .utils import convert_orientation_convention, create_rotation_matrix_from_view

class Camera():
    def __init__(self,
                 cfg: CameraCfg
                #  prim_path,
                #  device,
                 ):
        self.cfg = cfg 
        self._data = CameraData()

    @property
    def data(self) -> CameraData:
        # update sensors if needed
        # self._update_outdated_buffers()
        # return the data
        return self._data
 
     @property
    def image_shape(self) -> tuple[int, int]:
        """A tuple containing (height, width) of the camera sensor."""
        return (self.cfg.height, self.cfg.width)
        
    # initialize
    def _initialize_impl(self):
        
        self._view = XFormPrimView(self.cfg.prim_path, reset_xform_properties=False)
        self._view.initialize()

        self._ALL_INDICES = torch.arange(self._view.count, device=self.cfg.device, dtype=torch.long)

        # Create frame count buffer
        self._frame = torch.zeros(self._view.count, device=self.cfg.device, dtype=torch.long)
        self._rep_registry: dict[str, list[rep.annotators.Annotator]] = {name: list() for name in self.cfg.data_types}
        
        self._sensor_prims: list[UsdGeom.Camera] = list()
        
        stage = omni.usd.get_context().get_stage()
        
        for cam_prim_path in self._view.prim_paths:
            # Get camera prim
            cam_prim = stage.GetPrimAtPath(cam_prim_path)
            # Check if prim is a camera
            if not cam_prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"Prim at path '{cam_prim_path}' is not a Camera.")
            # Add to list
            sensor_prim = UsdGeom.Camera(cam_prim)
            self._sensor_prims.append(sensor_prim)
            # Get render product
            # From Isaac Sim 2023.1 onwards, render product is a HydraTexture so we need to extract the path
            render_prod_path = rep.create.render_product(cam_prim_path, resolution=(self.cfg.width, self.cfg.height))
            if not isinstance(render_prod_path, str):
                render_prod_path = render_prod_path.path
            # self._render_product_paths.append(render_prod_path)
            
            # Iterate over each data type and create annotator
            # create annotator node
            for name in self.cfg.data_types:
                # This code set camera sensor!
                rep_annotator = rep.AnnotatorRegistry.get_annotator(name, device=self.cfg.device)
                rep_annotator.attach(render_prod_path)
                # add to registry
                self._rep_registry[name].append(rep_annotator)
        # Create internal buffers
        self._create_buffers()


    def _create_buffers(self):
        """Create buffers for storing data."""
        # create the data object
        # -- pose of the cameras
        self._data.pos_w = torch.zeros((self._view.count, 3), device=self.cfg.device)
        self._data.quat_w_world = torch.zeros((self._view.count, 4), device=self.cfg.device)
        # -- intrinsic matrix
        self._data.intrinsic_matrices = torch.zeros((self._view.count, 3, 3), device=self.cfg.device)
        self._data.image_shape = (self.cfg.height, self.cfg.width)
        # -- output data
        # lazy allocation of data dictionary
        # since the size of the output data is not known in advance, we leave it as None
        # the memory will be allocated when the buffer() function is called for the first time.
        self._data.output = TensorDict({}, batch_size=self._view.count, device=self.cfg.device)
        self._data.info = [{name: None for name in self.cfg.data_types} for _ in range(self._view.count)]


    # def _update_intrinsic_matrices(self, idxs=[0]):
    #     """
    #     If you want to use multiple cameras, add idx=[0, 1 ....]
    #     Compute camera's matrix of intrinsic parameters.
        
    #     Also called calibration matrix. This matrix works for linear depth images. We assume square pixels.

    #     Note:
    #         The calibration matrix projects points in the 3D scene onto an imaginary screen of the camera.
    #         The coordinates of points on the image plane are in the homogeneous representation.
    #     """
    #     # iterate over all cameras
    #     for idx in idxs:
    #         # Get corresponding sensor prim
    #         sensor_prim = self._sensor_prims[idx]
    #         # get camera parameters
    #         focal_length = sensor_prim.GetFocalLengthAttr().Get()
    #         horiz_aperture = sensor_prim.GetHorizontalApertureAttr().Get()
    #         # get viewport parameters
    #         height, width = self.image_shape
    #         # calculate the field of view
    #         fov = 2 * math.atan(horiz_aperture / (2 * focal_length))
    #         # calculate the focal length in pixels
    #         focal_px = width * 0.5 / math.tan(fov / 2)
    #         # create intrinsic matrix for depth linear
    #         self._data.intrinsic_matrices[idx, 0, 0] = focal_px
    #         self._data.intrinsic_matrices[idx, 0, 2] = width * 0.5
    #         self._data.intrinsic_matrices[idx, 1, 1] = focal_px
    #         self._data.intrinsic_matrices[idx, 1, 2] = height * 0.5
    #         self._data.intrinsic_matrices[idx, 2, 2] = 1
    
        """
    Configuration
    """

    def set_intrinsic_matrices(
        self, matrices: torch.Tensor, focal_length: float = 1.0, env_ids: Sequence[int] | None = None
    ):
        """Set parameters of the USD camera from its intrinsic matrix.

        The intrinsic matrix and focal length are used to set the following parameters to the USD camera:

        - ``focal_length``: The focal length of the camera.
        - ``horizontal_aperture``: The horizontal aperture of the camera.
        - ``vertical_aperture``: The vertical aperture of the camera.
        - ``horizontal_aperture_offset``: The horizontal offset of the camera.
        - ``vertical_aperture_offset``: The vertical offset of the camera.

        .. warning::

            Due to limitations of Omniverse camera, we need to assume that the camera is a spherical lens,
            i.e. has square pixels, and the optical center is centered at the camera eye. If this assumption
            is not true in the input intrinsic matrix, then the camera will not set up correctly.

        Args:
            matrices: The intrinsic matrices for the camera. Shape is (N, 3, 3).
            focal_length: Focal length to use when computing aperture values. Defaults to 1.0.
            env_ids: A sensor ids to manipulate. Defaults to None, which means all sensor indices.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # iterate over env_ids
        for i, matrix in zip(env_ids, matrices):
            # convert to numpy for sanity
            intrinsic_matrix = np.asarray(matrix, dtype=float)
            # extract parameters from matrix
            f_x = intrinsic_matrix[0, 0]
            c_x = intrinsic_matrix[0, 2]
            f_y = intrinsic_matrix[1, 1]
            c_y = intrinsic_matrix[1, 2]
            # get viewport parameters
            height, width = self.image_shape
            height, width = float(height), float(width)
            # resolve parameters for usd camera
            params = {
                "focal_length": focal_length,
                "horizontal_aperture": width * focal_length / f_x,
                "vertical_aperture": height * focal_length / f_y,
                "horizontal_aperture_offset": (c_x - width / 2) / f_x,
                "vertical_aperture_offset": (c_y - height / 2) / f_y,
            }
            # change data for corresponding camera index
            sensor_prim = self._sensor_prims[i]
            # set parameters for camera
            for param_name, param_value in params.items():
                # convert to camel case (CC)
                param_name = self.to_camel_case(param_name, to="CC")
                # get attribute from the class
                param_attr = getattr(sensor_prim, f"Get{param_name}Attr")
                # set value
                # note: We have to do it this way because the camera might be on a different
                #   layer (default cameras are on session layer), and this is the simplest
                #   way to set the property on the right layer.
                omni.usd.set_prop_val(param_attr(), param_value)




    """
    String formatting
    
    """

    def to_camel_case(self, snake_str: str, to: str = "cC") -> str:
        """Converts a string from snake case to camel case.

        Args:
            snake_str: A string in snake case (i.e. with '_')
            to: Convention to convert string to. Defaults to "cC".

        Raises:
            ValueError: Invalid input argument `to`, i.e. not "cC" or "CC".

        Returns:
            A string in camel-case format.
        """
        # check input is correct
        if to not in ["cC", "CC"]:
            msg = "to_camel_case(): Choose a valid `to` argument (CC or cC)"
            raise ValueError(msg)
        # convert string to lower case and split
        components = snake_str.lower().split("_")
        if to == "cC":
            # We capitalize the first letter of each component except the first one
            # with the 'title' method and join them together.
            return components[0] + "".join(x.title() for x in components[1:])
        else:
            # Capitalize first letter in all the components
            return "".join(x.title() for x in components)