"""
Utilities to instantiate and configure a Duckietown environment, including the addition of wrappers.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import logging
import gym
import gym_duckietown
import numpy as np
from pathlib import Path
from gym_duckietown.simulator import Simulator, DEFAULT_ROBOT_SPEED, DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT

from duckietown_utils.wrappers.observation_wrappers import *
from duckietown_utils.wrappers.action_wrappers import *
from duckietown_utils.wrappers.reward_wrappers import *
from duckietown_utils.wrappers.simulator_mod_wrappers import *
from duckietown_utils.wrappers.aido_wrapper import AIDOWrapper
from duckietown_utils.wrappers.yolo_detection_wrapper import YOLOObjectDetectionWrapper
from duckietown_utils.wrappers.enhanced_observation_wrapper import EnhancedObservationWrapper
from duckietown_utils.wrappers.object_avoidance_action_wrapper import ObjectAvoidanceActionWrapper
from duckietown_utils.wrappers.lane_changing_action_wrapper import LaneChangingActionWrapper
from duckietown_utils.wrappers.multi_objective_reward_wrapper import MultiObjectiveRewardWrapper
from config.config import load_config
from config.enhanced_config import EnhancedRLConfig, load_enhanced_config

logger = logging.getLogger(__name__)

MAPSETS = {'multimap1': ['_custom_technical_floor', '_huge_C_floor', '_huge_V_floor', '_plus_floor',
                         'small_loop', 'small_loop_cw', 'loop_empty'],
           'multimap2': ['_custom_technical_floor', '_custom_technical_grass', 'udem1', 'zigzag_dists',
                         'loop_dyn_duckiebots'],
           'multimap_lfv': ['_custom_technical_floor_lfv', 'loop_dyn_duckiebots', 'loop_obstacles', 'loop_pedestrians'],
           'multimap_lfv_dyn_duckiebots': ['_loop_dyn_duckiebots_inner', '_loop_dyn_duckiebots_outer'],
           'multimap_lfv_duckiebots': ['_loop_duckiebots', '_loop_dyn_duckiebots_inner', '_loop_dyn_duckiebots_outer'],
           'multimap_aido5': ['LF-norm-loop', 'LF-norm-small_loop', 'LF-norm-zigzag', 'LF-norm-techtrack',
                              '_custom_technical_floor', '_huge_C_floor', '_huge_V_floor', '_plus_floor',
                              'small_loop', 'small_loop_cw', 'loop_empty'
                              ],
           }

CAMERA_WIDTH = DEFAULT_CAMERA_WIDTH
CAMERA_HEIGHT = DEFAULT_CAMERA_HEIGHT


def launch_and_wrap_env(env_config, default_env_id=0):
    try:
        env_id = env_config.worker_index  # config is passed by rllib
    except AttributeError as err:
        logger.warning(err)
        env_id = default_env_id

    robot_speed = env_config.get('robot_speed', DEFAULT_ROBOT_SPEED)
    # If random robot speed is specified, the robot speed key holds a dictionary
    if type(robot_speed) is dict or robot_speed == 'default':
        robot_speed = DEFAULT_ROBOT_SPEED  # The initial robot speed won't be random

    # The while loop and try block are necessary to prevent instant training crash from the
    # "Exception: Could not find a valid starting pose after 5000 attempts" in duckietown-gym-daffy 5.0.13
    spawn_successful = False
    seed = 1234 + env_id
    while not spawn_successful:
        try:
            env = Simulator(
                seed=seed,  # random seed
                map_name=resolve_multimap_name(env_config["training_map"], env_id),
                max_steps=env_config.get("episode_max_steps", 500),
                domain_rand=env_config["domain_rand"],
                dynamics_rand=env_config["dynamics_rand"],
                camera_rand=env_config["camera_rand"],
                camera_width=CAMERA_WIDTH,
                camera_height=CAMERA_HEIGHT,
                accept_start_angle_deg=env_config["accepted_start_angle_deg"],
                full_transparency=True,
                distortion=env_config["distortion"],
                frame_rate=env_config["simulation_framerate"],
                frame_skip=env_config["frame_skip"],
                robot_speed=robot_speed
            )
            spawn_successful = True
        except Exception as e:
            seed += 1  # Otherwise it selects the same tile in the next attempt
            logger.error("{}; Retrying with new seed: {}".format(e, seed))
    logger.debug("Env init successful")
    env = wrap_env(env_config, env)
    return env


def resolve_multimap_name(training_map_conf, env_id):
    if 'multimap' in training_map_conf:
        mapset = MAPSETS[training_map_conf]
        map_name_single_env = mapset[env_id % len(mapset)]
    else:
        map_name_single_env = training_map_conf
    return map_name_single_env


def wrap_env(env_config: dict, env=None):
    if env is None:
        # Create a dummy Duckietown-like env if None was passed. This is mainly necessary to easily run
        # dts challenges evaluate
        env = DummyDuckietownGymLikeEnv()

    # Simulation mod wrappers
    if env_config["mode"] in ['train', 'debug'] and env_config['aido_wrapper']:
        env = AIDOWrapper(env)
    env = InconvenientSpawnFixingWrapper(env)
    if env_config.get('spawn_obstacles', False):
        env = ObstacleSpawningWrapper(env, env_config)
    if env_config.get('spawn_forward_obstacle', False):
        env = ForwardObstacleSpawnnigWrapper(env, env_config)
    if env_config['mode'] in ['train', 'debug']:
        if type(env_config.get('frame_skip')) is dict or type(env_config.get('robot_speed')) is dict:
            # Randomize frame skip or robot speed
            env = ParamRandWrapper(env, env_config)

        if isinstance(env_config.get('action_delay_ratio', 0.), float):
            if env_config.get('action_delay_ratio', 0.) > 0.:
                env = ActionDelayWrapper(env, env_config)
        if env_config.get('action_delay_ratio', 0.) == 'random':
            env = ActionDelayWrapper(env, env_config)

    # Observation wrappers
    if env_config["crop_image_top"]:
        env = ClipImageWrapper(env, top_margin_divider=env_config["top_crop_divider"])
    if env_config.get("grayscale_image", False):
        env = RGB2GrayscaleWrapper(env)
    env = ResizeWrapper(env, shape=env_config["resized_input_shape"])
    if env_config['mode'] in ['train', 'debug'] and env_config.get('frame_repeating', 0.0) > 0:
        env = RandomFrameRepeatingWrapper(env, env_config)
    if env_config["frame_stacking"]:
        env = ObservationBufferWrapper(env, obs_buffer_depth=env_config["frame_stacking_depth"])
    if env_config["mode"] in ['train', 'debug'] and env_config['motion_blur']:
        env = MotionBlurWrapper(env)
    env = NormalizeWrapper(env)

    # Action wrappers
    if env_config["action_type"] == 'discrete':
        env = DiscreteWrapper(env)
    elif 'heading' in env_config["action_type"]:
        env = Heading2WheelVelsWrapper(env, env_config["action_type"])
    elif env_config["action_type"] == 'leftright_braking':
        env = LeftRightBraking2WheelVelsWrapper(env)
    elif env_config["action_type"] == 'leftright_clipped':
        env = LeftRightClipped2WheelVelsWrapper(env)
    elif env_config["action_type"] == 'steering_braking':
        env = SteeringBraking2WheelVelsWrapper(env)

    # Reward wrappers
    if env_config['mode'] in ['train', 'debug', 'inference']:
        if env_config["reward_function"] in ['Posangle', 'posangle']:
            env = DtRewardPosAngle(env)
            env = DtRewardVelocity(env)
        elif env_config["reward_function"] == 'target_orientation':
            env = DtRewardTargetOrientation(env)
            env = DtRewardVelocity(env)
        elif env_config["reward_function"] == 'lane_distance':
            env = DtRewardWrapperDistanceTravelled(env)
        elif env_config["reward_function"] == 'default_clipped':
            env = DtRewardClipperWrapper(env, 2, -2)
        else:  # Also env_config['mode'] == 'default'
            logger.warning("Default Gym Duckietown reward used")
        env = DtRewardCollisionAvoidance(env)
        # env = DtRewardProximityPenalty(env)
    return env


class DummyDuckietownGymLikeEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3),
            dtype=np.uint8
        )
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32
        )
        self.road_tile_size = 0.585

    def reset(self):
        logger.warning("Dummy Duckietown Gym reset() called!")
        return np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3))

    def step(self, action):
        logger.warning("Dummy Duckietown Gym step() called!")
        obs = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3))
        reward = 0.0
        done = True
        info = {}
        return obs, reward, done, info


def launch_and_wrap_enhanced_env(env_config, enhanced_config=None, default_env_id=0):
    """
    Launch and wrap Duckietown environment with enhanced capabilities.
    
    This function extends the existing launch_and_wrap_env function to include
    enhanced wrappers for object detection, avoidance, and lane changing.
    
    Args:
        env_config: Standard environment configuration dictionary
        enhanced_config: EnhancedRLConfig instance or path to enhanced config file
        default_env_id: Default environment ID if not specified in config
        
    Returns:
        Wrapped environment with enhanced capabilities
        
    Raises:
        ValueError: If wrapper configuration is invalid
        ImportError: If required dependencies are missing
    """
    # Load enhanced configuration
    if enhanced_config is None:
        enhanced_config = load_enhanced_config()
    elif isinstance(enhanced_config, (str, Path)):
        enhanced_config = load_enhanced_config(enhanced_config)
    elif not isinstance(enhanced_config, EnhancedRLConfig):
        raise ValueError("enhanced_config must be EnhancedRLConfig instance, path string, or None")
    
    # Start with standard environment setup
    env = launch_and_wrap_env(env_config, default_env_id)
    
    # Apply enhanced wrappers in proper order
    env = _apply_enhanced_wrappers(env, env_config, enhanced_config)
    
    logger.info("Enhanced Duckietown environment created successfully")
    return env


def _apply_enhanced_wrappers(env, env_config, enhanced_config):
    """
    Apply enhanced wrappers to environment with proper ordering and compatibility checks.
    
    Args:
        env: Base environment to wrap
        env_config: Standard environment configuration
        enhanced_config: Enhanced configuration
        
    Returns:
        Environment with enhanced wrappers applied
        
    Raises:
        ValueError: If wrapper configuration is incompatible
    """
    # Validate wrapper compatibility
    _validate_wrapper_compatibility(env_config, enhanced_config)
    
    # Apply observation wrappers first (YOLO detection and enhanced observation)
    if enhanced_config.is_feature_enabled('yolo'):
        try:
            env = YOLOObjectDetectionWrapper(
                env,
                model_path=enhanced_config.yolo.model_path,
                confidence_threshold=enhanced_config.yolo.confidence_threshold,
                device=enhanced_config.yolo.device,
                input_size=enhanced_config.yolo.input_size,
                max_detections=enhanced_config.yolo.max_detections
            )
            logger.info("YOLO Object Detection Wrapper applied")
        except Exception as e:
            logger.error(f"Failed to apply YOLO wrapper: {e}")
            if enhanced_config.debug_mode:
                raise
            logger.warning("Continuing without YOLO detection")
    
    # Apply enhanced observation wrapper to combine detection data
    if enhanced_config.is_feature_enabled('yolo'):
        try:
            env = EnhancedObservationWrapper(
                env,
                include_detection_features=True,
                flatten_observations=True
            )
            logger.info("Enhanced Observation Wrapper applied")
        except Exception as e:
            logger.error(f"Failed to apply Enhanced Observation wrapper: {e}")
            if enhanced_config.debug_mode:
                raise
            logger.warning("Continuing without enhanced observations")
    
    # Apply action wrappers (object avoidance and lane changing)
    if enhanced_config.is_feature_enabled('object_avoidance'):
        try:
            env = ObjectAvoidanceActionWrapper(
                env,
                safety_distance=enhanced_config.object_avoidance.safety_distance,
                avoidance_strength=enhanced_config.object_avoidance.avoidance_strength,
                min_clearance=enhanced_config.object_avoidance.min_clearance,
                max_avoidance_angle=enhanced_config.object_avoidance.max_avoidance_angle,
                smoothing_factor=enhanced_config.object_avoidance.smoothing_factor
            )
            logger.info("Object Avoidance Action Wrapper applied")
        except Exception as e:
            logger.error(f"Failed to apply Object Avoidance wrapper: {e}")
            if enhanced_config.debug_mode:
                raise
            logger.warning("Continuing without object avoidance")
    
    if enhanced_config.is_feature_enabled('lane_changing'):
        try:
            env = LaneChangingActionWrapper(
                env,
                lane_change_threshold=enhanced_config.lane_changing.lane_change_threshold,
                safety_margin=enhanced_config.lane_changing.safety_margin,
                max_lane_change_time=enhanced_config.lane_changing.max_lane_change_time,
                min_lane_width=enhanced_config.lane_changing.min_lane_width,
                evaluation_distance=enhanced_config.lane_changing.evaluation_distance
            )
            logger.info("Lane Changing Action Wrapper applied")
        except Exception as e:
            logger.error(f"Failed to apply Lane Changing wrapper: {e}")
            if enhanced_config.debug_mode:
                raise
            logger.warning("Continuing without lane changing")
    
    # Apply multi-objective reward wrapper last
    if enhanced_config.is_feature_enabled('multi_objective_reward') or any(
        enhanced_config.is_feature_enabled(f) for f in ['object_avoidance', 'lane_changing']
    ):
        try:
            reward_weights = {
                'lane_following': enhanced_config.reward.lane_following_weight,
                'object_avoidance': enhanced_config.reward.object_avoidance_weight,
                'lane_change': enhanced_config.reward.lane_change_weight,
                'efficiency': enhanced_config.reward.efficiency_weight,
                'safety_penalty': enhanced_config.reward.safety_penalty_weight,
                'collision_penalty': enhanced_config.reward.collision_penalty
            }
            
            env = MultiObjectiveRewardWrapper(
                env,
                reward_weights=reward_weights,
                log_rewards=enhanced_config.logging.log_rewards
            )
            logger.info("Multi-Objective Reward Wrapper applied")
        except Exception as e:
            logger.error(f"Failed to apply Multi-Objective Reward wrapper: {e}")
            if enhanced_config.debug_mode:
                raise
            logger.warning("Continuing without multi-objective rewards")
    
    return env


def _validate_wrapper_compatibility(env_config, enhanced_config):
    """
    Validate compatibility between standard and enhanced wrapper configurations.
    
    Args:
        env_config: Standard environment configuration
        enhanced_config: Enhanced configuration
        
    Raises:
        ValueError: If configurations are incompatible
    """
    # Check observation space compatibility
    if enhanced_config.is_feature_enabled('yolo'):
        if env_config.get('frame_stacking', False):
            logger.warning("Frame stacking with YOLO detection may impact performance")
        
        if env_config.get('grayscale_image', False):
            raise ValueError("YOLO detection requires RGB images, but grayscale_image is enabled")
    
    # Check action space compatibility
    if enhanced_config.is_feature_enabled('object_avoidance') or enhanced_config.is_feature_enabled('lane_changing'):
        if env_config.get('action_type') == 'discrete':
            logger.warning("Enhanced action wrappers work best with continuous action spaces")
    
    # Check reward function compatibility
    if enhanced_config.is_feature_enabled('multi_objective_reward'):
        if env_config.get('reward_function') not in ['Posangle', 'posangle', 'default']:
            logger.warning(f"Multi-objective rewards may conflict with reward_function: {env_config.get('reward_function')}")
    
    # Validate feature dependencies
    if enhanced_config.is_feature_enabled('object_avoidance') and not enhanced_config.is_feature_enabled('yolo'):
        logger.warning("Object avoidance without YOLO detection will use basic obstacle detection")
    
    if enhanced_config.is_feature_enabled('lane_changing') and not enhanced_config.is_feature_enabled('yolo'):
        logger.warning("Lane changing without YOLO detection may have limited effectiveness")


def get_enhanced_wrappers(wrapped_env):
    """
    Get lists of all wrappers applied to an enhanced environment.
    
    Args:
        wrapped_env: Wrapped environment
        
    Returns:
        Tuple of (obs_wrappers, action_wrappers, reward_wrappers, enhanced_wrappers)
    """
    obs_wrappers = []
    action_wrappers = []
    reward_wrappers = []
    enhanced_wrappers = []
    
    orig_env = wrapped_env
    while not (isinstance(orig_env, gym_duckietown.simulator.Simulator) or
               isinstance(orig_env, DummyDuckietownGymLikeEnv)):
        
        # Check for enhanced wrappers
        if isinstance(orig_env, (YOLOObjectDetectionWrapper, EnhancedObservationWrapper,
                                ObjectAvoidanceActionWrapper, LaneChangingActionWrapper,
                                MultiObjectiveRewardWrapper)):
            enhanced_wrappers.append(orig_env)
        
        # Standard wrapper categorization
        if isinstance(orig_env, gym.ObservationWrapper):
            obs_wrappers.append(orig_env)
        elif isinstance(orig_env, gym.ActionWrapper):
            action_wrappers.append(orig_env)
        elif isinstance(orig_env, gym.RewardWrapper):
            reward_wrappers.append(orig_env)
        elif isinstance(orig_env, gym.Wrapper):
            None
        else:
            assert False, ("[duckietown_utils.env.get_enhanced_wrappers] - {} Wrapper type is none of these:"
                           " gym.ObservationWrapper, gym.ActionWrapper, gym.RewardWrapper".format(orig_env))
        orig_env = orig_env.env

    return obs_wrappers[::-1], action_wrappers[::-1], reward_wrappers[::-1], enhanced_wrappers[::-1]


def get_wrappers(wrapped_env):
    obs_wrappers = []
    action_wrappers = []
    reward_wrappers = []
    orig_env = wrapped_env
    while not (isinstance(orig_env, gym_duckietown.simulator.Simulator) or
               isinstance(orig_env, DummyDuckietownGymLikeEnv)):
        if isinstance(orig_env, gym.ObservationWrapper):
            obs_wrappers.append(orig_env)
        elif isinstance(orig_env, gym.ActionWrapper):
            action_wrappers.append(orig_env)
        elif isinstance(orig_env, gym.RewardWrapper):
            reward_wrappers.append(orig_env)
        elif isinstance(orig_env, gym.Wrapper):
            None
        else:
            assert False, ("[duckietown_utils.env.get_wrappers] - {} Wrapper type is none of these:"
                           " gym.ObservationWrapper, gym.ActionWrapper, gym.ActionWrapper".format(orig_env))
        orig_env = orig_env.env

    return obs_wrappers[::-1], action_wrappers[::-1], reward_wrappers[::-1]


if __name__ == "__main__":
    # execute only if run as a script to test some functionality
    config = load_config('./config/config.yml')
    
    # Test standard environment
    print("=== Standard Environment ===")
    dummy_env = wrap_env(config['env_config'])
    obs_wrappers, action_wrappers, reward_wrappers = get_wrappers(dummy_env)
    print("Observation wrappers")
    print(*obs_wrappers, sep="\n")
    print("\nAction wrappers")
    print(*action_wrappers, sep="\n")
    print("\nReward wrappers")
    print(*reward_wrappers, sep="\n")
    
    # Test enhanced environment
    print("\n=== Enhanced Environment ===")
    try:
        enhanced_env = launch_and_wrap_enhanced_env(config['env_config'])
        obs_wrappers, action_wrappers, reward_wrappers, enhanced_wrappers = get_enhanced_wrappers(enhanced_env)
        print("Enhanced wrappers")
        print(*enhanced_wrappers, sep="\n")
        print("\nAll observation wrappers")
        print(*obs_wrappers, sep="\n")
        print("\nAll action wrappers")
        print(*action_wrappers, sep="\n")
        print("\nAll reward wrappers")
        print(*reward_wrappers, sep="\n")
    except Exception as e:
        print(f"Enhanced environment test failed: {e}")
        import traceback
        traceback.print_exc()
