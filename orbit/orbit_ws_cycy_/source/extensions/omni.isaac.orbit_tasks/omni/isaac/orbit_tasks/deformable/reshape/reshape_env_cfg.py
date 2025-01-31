# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import math

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.managers import CurriculumTermCfg as CurrTerm

from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg, SoftObjectCfg, GeometryObjectCfg
from omni.isaac.orbit.envs import RLDOTaskEnvCfg
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass

import omni.isaac.orbit_tasks.deformable.reshape.mdp as mdp
from pxr import Usd

##
# Pre-defined configs
##
from omni.isaac.orbit.sim.spawners.spawner_cfg import SpawnerCfg
from omni.isaac.orbit.sim.spawners.from_files.from_files import spawn_from_usd

from omni.isaac.orbit.sim.spawners.from_files.from_files_cfg import UsdFileCfg, GroundPlaneCfg
from omni.isaac.orbit.utils import configclass

from omni.isaac.orbit_assets import LOCAL_ASSETS_DIR
from omni.isaac.orbit_assets.franka_reshape import FRANKA_PANDA_CFG  # isort: skip
from omni.isaac.orbit.sensors.camera import Camera, CameraCfg

from omni.isaac.orbit.compat.markers import PointMarker

##
# Scene definition
##


@configclass
class ReshapeSceneCfg(InteractiveSceneCfg):

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.6]), 
        spawn=GroundPlaneCfg(),
    )

    # Add oring
    oring = SoftObjectCfg(
        prim_path="{ENV_REGEX_NS}/Oring",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.6, 0, 0.3], rot=[1, 0, 0, 0]), 
        spawn=UsdFileCfg(usd_path=f"{LOCAL_ASSETS_DIR}/reshape/train/oring_005_07.usd",
                         semantic_tags=[("class", "oring")])
    )
    
    # franka gripper
    robot: ArticulationCfg = MISSING
    
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

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.6, 0, -0.6], rot=[1., 0, 0, 0]), 
        spawn=UsdFileCfg(usd_path=f"{LOCAL_ASSETS_DIR}/reshape/table.usd"),
    )

    # franka_table
    franka_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Franka_table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-0.05, 0, -0.6], rot=[1., 0, 0, 0]), 
        spawn=UsdFileCfg(usd_path=f"{LOCAL_ASSETS_DIR}/reshape/franka_table.usd"),
    )
    
    # sensors; segmentation partially pointcloud
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/camera",
        # (-x, w, z, -y)
        # offset=CameraCfg.OffsetCfg(pos=(2,-2.8,2.3), rot=(-0.45986, 0.80202, 0.34647, -0.15893)),
        # offset=CameraCfg.OffsetCfg(pos=(1,-1.4,1.3), rot=(-0.45986, 0.80202, 0.34647, -0.15893)),
        
        offset=CameraCfg.OffsetCfg(pos=(1,-0.5,0.4), rot=(-0.45986, 0.80202, 0.34647, -0.15893)),
        update_period=0,
        # 2048*1536 
        height=768,
        width=1024,
        # height=480,
        # width=640,
        data_types=["rgb", "distance_to_image_plane", 
                    "normals", "semantic_segmentation",
                    "pointcloud"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.5)
        ),
        semantic_labels=["oring"],
    )

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()

    # object_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name=MISSING,  # will be set by agent env cfg
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
    #     ),
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # joint_velocity = mdp.JointVelocityActionCfg(asset_name="robot", 
    #                                         joint_names=['joint_x', 'joint_y', 'joint_z', "rev_z"],
    #                                         scale=10.,)
    # will be set by agent env cfg
    body_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RandomizationCfg:
    """Configuration for randomization."""

    # reset robot init pose
    reset_all = RandTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Oring peroperty
    # Randomization shape
    # 
    # reset_oring_shape = RandTerm(
    #     func=mdp.randomize_init_oring_shape,
    #     mode='reset',
    #     params={
    #         "asset_cfg": SceneEntityCfg("oring"),
    #         "oring_type" : [0, "07_01"]
    #     },
    # )

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


    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    # )
    pass

##
# Environment configuration
##


@configclass
class ReshapeEnvCfg(RLDOTaskEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: ReshapeSceneCfg = ReshapeSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=False) #if use deformables 
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
        
        # # self.sim.physx.bounce_threshold_velocity = 0.2 #FIXME
        # # self.sim.physx.bounce_threshold_velocity = 0.01 #FIXME
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 8 #FIXME
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2 * 16 * 1024 #FIXME

        self.sim.physx.gpu_max_rigid_contact_count: 524288
        self.sim.physx.gpu_max_rigid_patch_count: 33554432
        self.sim.physx.gpu_found_lost_pairs_capacity: 524288 #20965884
        # self.sim.physx.gpu_found_lost_aggregate_pairs_capacity: 262144
        # self.sim.physx.gpu_total_aggregate_pairs_capacity: 1048576
        self.sim.physx.gpu_max_soft_body_contacts: 4194304 #2097152 #16777216 #8388608 #2097152 #1048576
        self.sim.physx.gpu_max_particle_contacts: 1048576 #2097152 #1048576
        self.sim.physx.gpu_heap_capacity: 33554432
        self.sim.physx.gpu_temp_buffer_capacity: 16777216
        self.sim.physx.gpu_max_num_partitions: 8