# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, GeometryObjectCfg
from omni.isaac.lab.envs import RLDOTaskEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RandomizationTermCfg as RandTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.deformable.disentangle.mdp as mdp
from pxr import Usd

##
# Pre-defined configs
##
from omni.isaac.lab.sim.spawners.spawner_cfg import SpawnerCfg
from omni.isaac.lab.sim.spawners.from_files.from_files import spawn_from_usd

from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg, GroundPlaneCfg
from omni.isaac.lab.utils import configclass
# from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab_assets import LOCAL_ASSETS_DIR
from omni.isaac.lab_assets.franka_gripper_with_hook import PANDA_CFG  # isort: skip
# from omni.isaac.lab_assets.franka_gripper import PANDA_CFG  # isort: skip
from omni.isaac.lab_assets.disentangle_pole import POLE_CFG  # isort: skip
from omni.isaac.lab.sensors.camera import Camera, CameraCfg

from omni.isaac.lab.compat.markers import PointMarker

##
# Scene definition
##


@configclass
class DisentangleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.]), 
        spawn=GroundPlaneCfg(),
    )
    # Add assets
    # Add oring
    oring = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Oring",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 1], rot=[1, 0, 0, 0]), 
        # spawn=UsdFileCfg(usd_path=f"{LOCAL_ASSETS_DIR}/disentangle/train/oring_07_005.usd",
        spawn=UsdFileCfg(usd_path=f"{LOCAL_ASSETS_DIR}/disentangle/test/oring_07_01_test.usd",
                         semantic_tags=[("class", "oring")])
    )
    # franka gripper
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

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0, 0], rot=[1., 0, 0, 0]), 
        spawn=UsdFileCfg(usd_path=f"{LOCAL_ASSETS_DIR}/disentangle/table.usd"),
    )
    # goal pole
    goal_pole = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalPole",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0.71, 0], rot=[1, 0, 0, 0]), 
        spawn=UsdFileCfg(usd_path=f"{LOCAL_ASSETS_DIR}/disentangle/goal_nail.usd") 
    )
    
    # init pole
    init_pole : ArticulationCfg = POLE_CFG.replace(prim_path="{ENV_REGEX_NS}/InitPole")
    
    # sensors; segmentation partially pointcloud
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/camera",
        # (-x, w, z, -y)
        offset=CameraCfg.OffsetCfg(pos=(2,-2.8,2.3), rot=(-0.45986, 0.80202, 0.34647, -0.15893)),
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane", 
                    "normals", "semantic_segmentation",
                    "pointcloud"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 6.0)
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


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_velocity = mdp.JointVelocityActionCfg(asset_name="robot", 
                                            joint_names=['joint_x', 'joint_y', 'joint_z', "rev_z"],
                                            scale=10.,)

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

    # reset robot init pose
    # need to add randomization
    reset_franka_position = RandTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    # set_init_pole_default_pose
    reset_init_pole = RandTerm(
        func=mdp.set_init_pole_position,
        mode='reset',
        params={
            "asset_cfg": SceneEntityCfg("init_pole"),
        },
    )

    
    # oring
    reset_oring_shape = RandTerm(
        func=mdp.randomize_init_oring_shape,
        mode='reset',
        params={
            "asset_cfg": SceneEntityCfg("oring"),
            "oring_type" : [0, "07_01"]
        },
    )
    
    # manual grip using franka gripper
    reset_gripper_pose = RandTerm(
        func=mdp.set_manual_grip,
        mode='reset',
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )
    
    # init_pole
    # ADD AFTER RESET PART...
    reset_randomize_init_pole = RandTerm(
        func=mdp.randomize_init_pole_position,
        mode='reset',
        params={
            "asset_cfg": SceneEntityCfg("init_pole", joint_names=["joint_y"]),
            "position_range": (-0.7, -0.4),
            
            
        },
    )



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
class DisentangleEnvCfg(RLDOTaskEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: DisentangleSceneCfg = DisentangleSceneCfg(num_envs=4, env_spacing=5.0, replicate_physics=False) #if use deformables 
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
        # # self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4 #FIXME
        # # self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 #FIXME
        # # self.sim.physx.friction_correlation_distance = 0.00625 #FIXME
        # self.sim.physx.bounce_threshold_velocity: 0.2
        # self.sim.physx.enable_stabilization: True
        # self.sim.physx.gpu_max_rigid_contact_count: 2**24
        # self.sim.physx.gpu_max_rigid_patch_count: 33554432
        # # self.sim.physx.gpu_found_lost_pairs_capacity: 524288 #20965884
        # # self.sim.physx.gpu_found_lost_aggregate_pairs_capacity: 262144
        # # self.sim.physx.gpu_total_aggregate_pairs_capacity: 1048576
        # self.sim.physx.gpu_max_soft_body_contacts: 4194304 #2097152 #16777216 #8388608 #2097152 #1048576
        # self.sim.physx.gpu_max_particle_contacts: 1048576 #2097152 #1048576
        # self.sim.physx.enable_ccd: True # 
        # self.sim.physx.gpu_temp_buffer_capacity: 2**26
        # # self.sim.physx.gpu_heap_capacity: 33554432
        # # self.sim.physx.gpu_temp_buffer_capacity: 16777216
        # # self.sim.physx.gpu_max_num_partitions: 8