# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from dataclasses import MISSING

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, DeformObjectCfg
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
from omni.isaac.orbit.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: DeformObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]), #FIXME
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]), #FIXME
        spawn=GroundPlaneCfg(),
    )

    # poles #FIXME: implement here or at randomazations?
    goal_pole = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalPole",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]), #FIXME
        spawn=UsdFileCfg() #FIXME,
    )

    aux_pole = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/AuxPole",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]), #FIXME
        spawn=UsdFileCfg() #FIXME,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # camera #FIXME
    camera = AssetBaseCfg(
        prim_path="/World/camera",
        spawn=sim_utils.Sensors.camera() #FIXME
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    #FIXME: WHAT IS THIS?

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    body_joint_pos: mdp.JointPositionActionCfg = MISSING
    finger_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        #FIXME: Scales, func args
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, scale=1.0) #FIXME
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=1.0) #FIXME
        object_embedding = ObsTerm(funct=mdp.object_embedding, scale=1.0)
        object_transformation = ObsTerm(func=mdp.object_transformation, scale=1.0)
        aux_pole_position = ObsTerm(func=mdp.aux_pole_position, scale=1.0)
        z_move_states = ObsTerm(func=mdp.z_move_states, scale=1.0)
        tool_positions = ObsTerm(func=mdp.tool_positions, scale=1.0)
        tool_orientations_cos = ObsTerm(func=mdp.tool_orientations_cos, scale=1.0)
        tool_orientations_sin = ObsTerm(func=mdp.tool_orientations_sin, scale=1.0)
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

    randomize_deform_body_parameter = RandTerm(
        func=mdp.randomize_deform_body_parameter,
        mode="reset",
        params={
            "stiffness": {min: 0.1, max: 0.1}, #FIXME
            "youngs": {min: 0.1, max: 0.1}, #FIXME
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )
    
    randomize_aux_pole_position = RandTerm(
        func=mdp.randomize_aux_pole_position,
        mode="reset",
        params={
            "position": {min: 0.1, max: 0.1}, #FIXME
            "asset_cfg": SceneEntityCfg("aux_pole"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.06}, weight=15.0)

    ee_goal_distance = RewTerm(
        func=mdp.ee_goal_distance,
        params={}, #FIXME
        weight=5.0,
    )

    object_chamfer_distance = RewTerm(
        func=mdp.object_chamfer_distance,
        params={}, #FIXME
        weight=1.0,
    )

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

    time_out = DoneTerm(func=mdp.time_out, time_out=True) #FIXME: mdp.time_out

    object_dropping = DoneTerm( #FIXME: Remove?
        func=mdp.base_height, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    #FIXME:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class DisentangleEnvCfg(RLTaskEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
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
        self.decimation = 2
        self.episode_length_s = 5.0 #FIXME: 50, why float?
        # simulation settings
        self.sim.dt = 0.01  # 100Hz #FIXME

        self.sim.physx.bounce_threshold_velocity = 0.2 #FIXME
        self.sim.physx.bounce_threshold_velocity = 0.01 #FIXME
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4 #FIXME
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 #FIXME
        self.sim.physx.friction_correlation_distance = 0.00625 #FIXME
