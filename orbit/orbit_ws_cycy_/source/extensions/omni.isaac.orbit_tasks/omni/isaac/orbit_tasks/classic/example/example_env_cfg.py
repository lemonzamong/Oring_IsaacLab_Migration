# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass

import omni.isaac.orbit_tasks.classic.example.mdp as mdp

##
# Pre-defined configs
##
from omni.isaac.orbit_assets.cartpole import CARTPOLE_CFG  # isort:skip
from omni.isaac.orbit.sim.spawners.from_files.from_files_cfg import UsdFileCfg, GroundPlaneCfg
from omni.isaac.orbit.utils import configclass
# from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit_assets import LOCAL_ASSETS_DIR
from omni.isaac.orbit_assets.franka_gripper import PANDA_CFG  # isort: skip
from omni.isaac.orbit.sensors.camera import Camera, CameraCfg

##
# Scene definition
##


@configclass
class ExampleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.]), 
        spawn=GroundPlaneCfg(),
    )

    # cartpole
    # robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot: ArticulationCfg = PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )
    
    # Add assets
    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0, 0], rot=[1., 0, 0, 0]), 
        spawn=UsdFileCfg(usd_path=f"{LOCAL_ASSETS_DIR}/disentangle/table.usd"),
    )

    # poles 
    goal_pole = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalPole",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0], rot=[1, 0, 0, 0]), 
        spawn=UsdFileCfg(usd_path=f"{LOCAL_ASSETS_DIR}/disentangle/goal_nail.usd") 
    )

    # Change Articulation
    init_pole = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/InitPole",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, -0.8, 0], rot=[1, 0, 0, 0]), 
        spawn=UsdFileCfg(usd_path=f"{LOCAL_ASSETS_DIR}/disentangle/init_nail.usd"),
    )

    # Add oring
    aux_pole = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Oring",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 2], rot=[1, 0, 0, 0]), 
        spawn=UsdFileCfg(usd_path=f"{LOCAL_ASSETS_DIR}/disentangle/train/oring_07_015.usd"),
    )

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/camera",
        # offset=CameraCfg.OffsetCfg(pos=(5,-3,3.5), rot=(0.73151, 0.45945, 0.24828, 0.43836)),
        # (-x, w, z, -y)
        offset=CameraCfg.OffsetCfg(pos=(5,-3,3.5), rot=(-0.45945, 0.73151, 0.43836, -0.24828)),
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane", 
                    "normals", "semantic_segmentation",
                    "pointcloud"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        
        # semantic_labels=["sphere"],
    )

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
    # joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
    joint_position = mdp.JointPositionActionCfg(asset_name="robot", 
                                                joint_names=['joint_x', 'joint_y', 'joint_z', 'rev_z'],
                                                scale=0.5,)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RandomizationCfg:
    """Configuration for randomization."""

    # # reset
    # reset_cart_position = RandTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
    #         "position_range": (-1.0, 1.0),
    #         "velocity_range": (-0.5, 0.5),
    #     },
    # )

    # reset_pole_position = RandTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
    #         "position_range": (-0.25 * math.pi, 0.25 * math.pi),
    #         "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
    #     },
    # )
    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    # pole_pos = RewTerm(
    #     func=mdp.joint_pos_target_l2,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    # )
    # # (4) Shaping tasks: lower cart velocity
    # cart_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.01,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    # )
    # # (5) Shaping tasks: lower pole angular velocity
    # pole_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.005,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    # cart_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    # )


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass


##
# Environment configuration
##


@configclass
class ExampleEnvCfg(RLTaskEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: ExampleSceneCfg = ExampleSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=False) #if use deformables 
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    randomization: RandomizationCfg = RandomizationCfg()
    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # No command generator
    commands: CommandsCfg = CommandsCfg()


    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 2
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 100
