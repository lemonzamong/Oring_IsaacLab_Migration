# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawners that spawn lights in the simulation.

There are various different kinds of lights that can be spawned into the USD stage.
Please check the Omniverse documentation for `lighting overview
<https://docs.omniverse.nvidia.com/materials-and-rendering/latest/103/lighting.html>`_.
"""

from .lights import spawn_light
from .lights_cfg import CylinderLightCfg, DiskLightCfg, DistantLightCfg, DomeLightCfg, LightCfg, SphereLightCfg
