import math
from typing import TYPE_CHECKING

import numpy as np

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle
from gymnasium.envs.box2d.bipedal_walker import FPS
from collections import deque


POSE_H = 6
IMITATION_H = 2
PROGRESS_H = 3
ALIVE_H = 2

MAX_FORWARD_ANGLE_DEG = 40.0
MIN_FORWARD_ANGLE_DEG = -40.0
MAX_FORWARD_ANGLE_RAD = MAX_FORWARD_ANGLE_DEG * math.pi / 180.0
MIN_FORWARD_ANGLE_RAD = MIN_FORWARD_ANGLE_DEG * math.pi / 180.0
HIP_AMPLITUDE = (MAX_FORWARD_ANGLE_RAD - MIN_FORWARD_ANGLE_RAD) / 2.0
HIP_OFFSET = (MAX_FORWARD_ANGLE_RAD + MIN_FORWARD_ANGLE_RAD) / 2.0
HIP_DEVIATION_PENALTY_WEIGHT = 2.5

MIN_KNEE_ANGLE_RAD = 45 * math.pi / 180.0
MAX_KNEE_ANGLE_RAD = 90 * math.pi / 180.0
KNEE_AMPLITUDE = (MAX_KNEE_ANGLE_RAD - MIN_KNEE_ANGLE_RAD) / 2.0
KNEE_OFFSET = (MAX_KNEE_ANGLE_RAD + MIN_KNEE_ANGLE_RAD) / 2.0
KNEE_PENALTY_CONSTANT = 2

ALTERNATE_SWING_REWARD = 1.0

AIRBORNE_PENALTY_CONSTANT = -0.1

SYMMETRY_PENALTY_WEIGHT = 1.8

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e

def get_target_hip_angles(phase):
    """
    Calculate target hip angles for both legs based on gait phase.

    Input:
        phase: float, current gait phase (0 to 2*pi)

    Output:
        target_hip1: float, target angle for first hip joint
        target_hip2: float, target angle for second hip joint

    Function: Computes sinusoidal hip angles ensuring legs swing only in front
    of body within [MIN_FORWARD_ANGLE, MAX_FORWARD_ANGLE] range with opposite phases.
    """
    target_hip1 = HIP_OFFSET + HIP_AMPLITUDE * math.cos(phase)
    target_hip2 = HIP_OFFSET + HIP_AMPLITUDE * math.cos(phase + math.pi)
    return target_hip1, target_hip2

def calculate_hip_deviation_penalty(current_hip_angle, target_hip_angle):
    """
    Calculate penalty for hip joint deviation from target angle.

    Input:
        current_hip_angle: float, current hip joint angle
        target_hip_angle: float, target hip joint angle

    Output:
        penalty: float, negative penalty value (squared deviation)

    Function: Computes quadratic penalty for hip angle deviation to encourage
    following the target gait pattern.
    """
    deviation = current_hip_angle - target_hip_angle
    penalty = -HIP_DEVIATION_PENALTY_WEIGHT * (deviation ** 2)
    return penalty

def get_target_knee_angles(phase):
    """
    Calculate target knee angles for both legs based on gait phase.

    Input:
        phase: float, current gait phase (0 to 2*pi)

    Output:
        target_knee1: float, target angle for first knee joint
        target_knee2: float, target angle for second knee joint

    Function: Computes sinusoidal knee angles synchronized with hip movement,
    range [45, 90] degrees, with opposite phases for alternating leg motion.
    """
    target_knee1 = KNEE_OFFSET + KNEE_AMPLITUDE * math.cos(phase)
    target_knee2 = KNEE_OFFSET + KNEE_AMPLITUDE * math.cos(phase + math.pi)
    return target_knee1, target_knee2

def calculate_knee_deviation_penalty(current_knee_angle, target_knee_angle):
    """
    Calculate penalty for knee joint deviation from target angle.

    Input:
        current_knee_angle: float, current knee joint angle
        target_knee_angle: float, target knee joint angle

    Output:
        penalty: float, negative penalty value (squared deviation)

    Function: Computes quadratic penalty for knee angle deviation to encourage
    following the target gait pattern.
    """
    deviation = current_knee_angle - target_knee_angle
    penalty = -KNEE_PENALTY_CONSTANT * (deviation ** 2)
    return penalty

def calculate_airborne_penalty(leg1_contact, leg2_contact):
    """
    Calculate penalty when both legs are off the ground simultaneously.

    Input:
        leg1_contact: float, ground contact value for first leg
        leg2_contact: float, ground contact value for second leg

    Output:
        penalty: float, penalty value if both legs airborne, 0 otherwise

    Function: Penalizes jumping/hopping by detecting when neither leg contacts ground.
    """
    if leg1_contact < 1.0 and leg2_contact < 1.0:
        return AIRBORNE_PENALTY_CONSTANT
    else:
        return 0.0

def calculate_alternate_swing_reward(hip1_angle, hip2_angle, hip1_vel, hip2_vel):
    """
    Calculate reward for alternating leg swing motion.

    Input:
        hip1_angle: float, current angle of first hip
        hip2_angle: float, current angle of second hip
        hip1_vel: float, angular velocity of first hip
        hip2_vel: float, angular velocity of second hip

    Output:
        reward: float, sum of rewards for both legs' swing motion

    Function: Rewards legs swinging in correct direction (forward or backward)
    within valid range, scaled by remaining distance to angle limits.
    """
    alternate_swing_reward1, alternate_swing_reward2 = 0.0, 0.0
    if (hip1_angle > 0 and hip1_vel > 0 and hip1_angle < MAX_FORWARD_ANGLE_RAD):
        alternate_swing_reward1 = ALTERNATE_SWING_REWARD * (MAX_FORWARD_ANGLE_RAD - hip1_angle) / MAX_FORWARD_ANGLE_RAD
    elif (hip1_angle < 0 and hip1_vel < 0 and hip1_angle > MIN_FORWARD_ANGLE_RAD):
        alternate_swing_reward1 = ALTERNATE_SWING_REWARD * (hip1_angle - MIN_FORWARD_ANGLE_RAD) / abs(MIN_FORWARD_ANGLE_RAD)

    if(hip2_angle > 0 and hip2_vel > 0 and hip2_angle < MAX_FORWARD_ANGLE_RAD):
        alternate_swing_reward2 = ALTERNATE_SWING_REWARD * (MAX_FORWARD_ANGLE_RAD - hip2_angle) / MAX_FORWARD_ANGLE_RAD
    elif (hip2_angle < 0 and hip2_vel < 0 and hip2_angle > MIN_FORWARD_ANGLE_RAD):
        alternate_swing_reward2 = ALTERNATE_SWING_REWARD * (hip2_angle - MIN_FORWARD_ANGLE_RAD) / abs(MIN_FORWARD_ANGLE_RAD)

    return alternate_swing_reward1 + alternate_swing_reward2

def calculate_symmetry_penalty(avg_hip1, avg_hip2):
    """
    Calculate penalty for asymmetric leg motion patterns.

    Input:
        avg_hip1: float, moving average of first hip angle
        avg_hip2: float, moving average of second hip angle

    Output:
        penalty: float, negative penalty for deviation from zero center

    Function: Penalizes legs having non-zero average angle (bias forward/backward),
    enforcing symmetric swing around body centerline for both legs.
    """
    penalty1 = -SYMMETRY_PENALTY_WEIGHT * (avg_hip1 ** 2)
    penalty2 = -SYMMETRY_PENALTY_WEIGHT * (avg_hip2 ** 2)
    return penalty1 + penalty2

def calculate_custom_reward(obs, original_reward, terminated, phase, prev_front, front, prev_contact, contact, imitation_weight: float, symmetry_weight: float, avg_hip1: float, avg_hip2: float):
    """
    Calculate custom reward combining imitation, progress, stability, and symmetry.

    Input:
        obs: array, current observation state
        original_reward: float, original environment reward
        terminated: bool, episode termination flag
        phase: float, current gait phase
        prev_front: int, previous front leg indicator
        front: int, current front leg indicator
        prev_contact: tuple, previous leg contact states
        contact: tuple, current leg contact states
        imitation_weight: float, curriculum weight for imitation reward
        symmetry_weight: float, curriculum weight for symmetry penalty
        avg_hip1: float, moving average angle of hip1
        avg_hip2: float, moving average angle of hip2

    Output:
        total_reward: float, combined reward value

    Function: Computes weighted sum of multiple reward components including gait imitation,
    forward progress, pose stability, survival bonus, airborne penalty, and symmetry penalty.
    Uses curriculum learning weights that increase during training.
    """
    if terminated and original_reward <= -95:
        return -1.0
    elif terminated:
        return 3

    hull_angle, hull_ang_vel, vel_x, vel_y = obs[0:4]
    hip1_angle = obs[4]
    knee1_angle = obs[6]
    hip2_angle = obs[9]
    knee2_angle = obs[11]
    hip1_vel = obs[5]
    hip2_vel = obs[10]

    target_hip1, target_hip2 = get_target_hip_angles(phase)
    hip_deviation_penalty1 = calculate_hip_deviation_penalty(hip1_angle, target_hip1)
    hip_deviation_penalty2 = calculate_hip_deviation_penalty(hip2_angle, target_hip2)
    imitation_panalty = 0.0
    imitation_panalty += hip_deviation_penalty1 + hip_deviation_penalty2

    target_knee1, target_knee2 = get_target_knee_angles(phase)
    knee_deviation_penalty1 = calculate_knee_deviation_penalty(knee1_angle, target_knee1)
    knee_deviation_penalty2 = calculate_knee_deviation_penalty(knee2_angle, target_knee2)
    imitation_panalty += knee_deviation_penalty1 + knee_deviation_penalty2

    alternate_swing_reward = calculate_alternate_swing_reward(hip1_angle, hip2_angle, hip1_vel, hip2_vel)

    symmetry_penalty = calculate_symmetry_penalty(avg_hip1, avg_hip2)

    progress_reward = 5.0 * vel_x * PROGRESS_H
    pose_penalty = (-0.2 * abs(hull_angle) - 0.1 * abs(vel_y)) * POSE_H
    alive_bonus = 0.1 * ALIVE_H
    airborne_penalty = calculate_airborne_penalty(obs[8], obs[13])

    total_reward = (
        imitation_panalty * imitation_weight +
        alternate_swing_reward * imitation_weight +
        progress_reward +
        pose_penalty +
        alive_bonus +
        airborne_penalty +
        symmetry_penalty * symmetry_weight
    )
    return total_reward

class new_custom_reward_step(gym.Wrapper):
    """
    Custom environment wrapper implementing curriculum learning with gait-based rewards.

    Input:
        env: gym.Env, base environment to wrap
        total_training_steps: int, total training steps for curriculum scheduling

    Function: Wraps BipedalWalker environment to provide custom reward shaping based on
    biomechanical gait patterns. Implements curriculum learning by gradually increasing
    imitation and symmetry penalty weights during training. Tracks gait phase, hip angle
    history for symmetry analysis, and contact states for alternating leg motion.
    """
    def __init__(self, env, total_training_steps: int):
        super().__init__(env)
        self.phase = 0.0
        self.gait_frequency = 2.5
        self.prev_front = 1
        self.prev_contact = (1, 1)
        self.current_step = 0
        self.total_training_steps = total_training_steps
        self.history_len = 20
        self.hip1_angle_history = deque(maxlen=self.history_len)
        self.hip2_angle_history = deque(maxlen=self.history_len)

    def reset(self, **kwargs):
        """
        Reset environment and all wrapper state variables.

        Input:
            **kwargs: additional arguments passed to base environment reset

        Output:
            obs: array, initial observation
            info: dict, initial info dictionary

        Function: Resets gait phase, contact tracking, and initializes hip angle
        history buffers with starting angles from the environment.
        """
        obs, info = self.env.reset(**kwargs)
        self.phase = 0.0
        self.gait_frequency = 2.5
        self.prev_front = 1
        self.prev_contact = (1, 1)
        initial_hip1_angle = obs[4]
        initial_hip2_angle = obs[9]
        self.hip1_angle_history.clear()
        self.hip2_angle_history.clear()
        for _ in range(self.history_len):
            self.hip1_angle_history.append(initial_hip1_angle)
            self.hip2_angle_history.append(initial_hip2_angle)
        return obs, info

    def step(self, action):
        """
        Execute one environment step with custom reward calculation.

        Input:
            action: array, action to take in environment

        Output:
            state: array, next observation
            total_reward: float, custom calculated reward
            terminated: bool, episode termination flag
            truncated: bool, episode truncation flag
            info: dict, additional information

        Function: Executes base environment step, updates gait phase and hip angle history,
        computes curriculum learning weights based on training progress, calculates custom
        reward using biomechanical gait patterns, and tracks training step count.
        """
        state, original_reward, terminated, truncated, info = self.env.step(action)

        self.hip1_angle_history.append(state[4])
        self.hip2_angle_history.append(state[9])

        self.phase += 2 * math.pi * self.gait_frequency * (1.0 / FPS)
        if self.phase > 2 * math.pi:
            self.phase -= 2 * math.pi

        front = 0 if state[4] > state[9] else 1
        contact = (state[8]>0, state[13]>0)

        progress_ratio = self.current_step / self.total_training_steps
        if progress_ratio < 0.25:
            imitation_weight = progress_ratio / 0.25
            symmetry_weight = 0.0
        elif progress_ratio < 0.5:
            imitation_weight = 1.0
            symmetry_weight = (progress_ratio - 0.25) / 0.25
        else:
            imitation_weight = 1.0
            symmetry_weight = 1.0

        imitation_weight = min(1.0, imitation_weight)
        symmetry_weight = min(1.0, symmetry_weight)

        avg_hip1_angle = np.mean(self.hip1_angle_history)
        avg_hip2_angle = np.mean(self.hip2_angle_history)

        total_reward = calculate_custom_reward(
            state,
            original_reward,
            terminated,
            self.phase,
            self.prev_front,
            front,
            self.prev_contact,
            contact,
            imitation_weight=imitation_weight,
            symmetry_weight=symmetry_weight,
            avg_hip1=avg_hip1_angle,
            avg_hip2=avg_hip2_angle
        )
        self.prev_front = front
        self.current_step += 1
        return state, total_reward, terminated, truncated, info
