# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from dataclasses import MISSING

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, SoftObjectCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.orbit.sensors.camera import Camera, CameraCfg
from omni.isaac.orbit.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.orbit.utils import configclass
# from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit_assets import LOCAL_ASSETS_DIR

from . import mdp

##
# Scene definition
##


@configclass
class OringDisentangleSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0, 0], rot=[1., 0, 0, 0]), 
        spawn=UsdFileCfg(usd_path=f"{LOCAL_ASSETS_DIR}/disentangle/table.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.]), 
        spawn=GroundPlaneCfg(),
    )

    # poles 
    goal_pole = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalPole",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0], rot=[1, 0, 0, 0]), 
        spawn=UsdFileCfg(usd_path=f"{LOCAL_ASSETS_DIR}/disentangle/goal_nail.usd") 
    )

    aux_pole = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/AuxPole",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, -0.8, 0], rot=[1, 0, 0, 0]), 
        spawn=UsdFileCfg(usd_path=f"{LOCAL_ASSETS_DIR}/disentangle/init_nail.usd"),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/camera",
        update_period=0,
        height=2048,
        width=1536,
        data_types=["rgb", "distance_to_image_plane", 
                    "normals", "semantic_segmentation",
                    "pointcloud"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        semantic_labels=["sphere"],
    )

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # object_pose = mdp.UniformPoseCommandCfg(
        # asset_name="robot",
        # body_name=MISSING,  # will be set by agent env cfg
        # resampling_time_range=(5.0, 5.0),
        # debug_vis=True,
        # ranges=mdp.UniformPoseCommandCfg.Ranges(
            # pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        # ),
    # )
    # no commands for this MDP
    null = mdp.NullCommandCfg()



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    body_joint_pos: mdp.JointPositionActionCfg = MISSING
    # finger_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        #FIXME: Scales, func args
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel, scale=1.0) #FIXME
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=1.0) #FIXME
        # # object_embedding = ObsTerm(funct=mdp.object_embedding, scale=1.0)
        # object_transformation = ObsTerm(func=mdp.object_transformation, scale=1.0)
        # aux_pole_position = ObsTerm(func=mdp.aux_pole_position, scale=1.0)
        # z_move_states = ObsTerm(func=mdp.z_move_states, scale=1.0)
        # tool_positions = ObsTerm(func=mdp.tool_positions, scale=1.0)
        # tool_orientations_cos = ObsTerm(func=mdp.tool_orientations_cos, scale=1.0)
        # tool_orientations_sin = ObsTerm(func=mdp.tool_orientations_sin, scale=1.0)
        actions = ObsTerm(func=mdp.last_action, scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RandomizationCfg:
    """Configuration for randomization."""

    reset_all = RandTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = RandTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    # lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.06}, weight=15.0)

    # object_goal_tracking = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.3, "minimal_height": 0.06, "command_name": "object_pose"},
    #     weight=16.0,
    # )

    # object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.05, "minimal_height": 0.06, "command_name": "object_pose"},
    #     weight=5.0,
    # )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
# 
    # object_dropping = DoneTerm( #FIXME: Remove?
        # func=mdp.base_height, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # action_rate = CurrTerm(
        # func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    # )
# 
    # joint_vel = CurrTerm(
        # func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    # )
# 
    pass
##
# Environment configuration
##


@configclass
class DisentangleEnvCfg(RLTaskEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: OringDisentangleSceneCfg = OringDisentangleSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    randomization: RandomizationCfg = RandomizationCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 5.0 
        
        # simulation settings
        self.sim.dt = 0.01  # 100Hz 

        # self.sim.physx.bounce_threshold_velocity = 0.2 #FIXME
        # self.sim.physx.bounce_threshold_velocity = 0.01 #FIXME
        # self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4 #FIXME
        # self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 #FIXME
        # self.sim.physx.friction_correlation_distance = 0.00625 #FIXME
        # FIX THIS