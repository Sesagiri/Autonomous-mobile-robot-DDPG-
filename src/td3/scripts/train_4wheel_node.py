#!/usr/bin/env python3

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer

import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
import threading

import math
import random

import point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
GOAL_REACHED_DIST = 0.3          # metres (real-world distance from odometry)
COLLISION_DIST    = 0.25         # metres (real-world; lidar is de-normalised before comparison)
TIME_DELTA        = 0.2          # seconds per step

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

last_odom       = None
environment_dim = 20
# Start with all-10 so the "waiting" loop works correctly
lidar_data = np.ones(environment_dim) * 10.0   # raw metres (NOT normalised globally)


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for ep in range(eval_episodes):
        env.get_logger().info(f"Evaluating episode {ep}")
        count = 0
        state = env.reset()
        done  = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
            a_in   = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _ = env.step(a_in)
            avg_reward += reward
            count      += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col     = col / eval_episodes
    env.get_logger().info("=" * 50)
    env.get_logger().info(
        f"Eval | Epoch {epoch} | Avg Reward {avg_reward:.3f} | Avg Collisions {avg_col:.3f}"
    )
    env.get_logger().info("=" * 50)
    return avg_reward


# ─────────────────────────────────────────────
# ACTOR
# ─────────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh    = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        return self.tanh(self.layer_3(s))


# ─────────────────────────────────────────────
# CRITIC  (twin Q-networks)
# ─────────────────────────────────────────────
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1
        self.layer_1   = nn.Linear(state_dim,  800)
        self.layer_2_s = nn.Linear(800,         600)
        self.layer_2_a = nn.Linear(action_dim,  600)
        self.layer_3   = nn.Linear(600,           1)
        # Q2
        self.layer_4   = nn.Linear(state_dim,  800)
        self.layer_5_s = nn.Linear(800,         600)
        self.layer_5_a = nn.Linear(action_dim,  600)
        self.layer_6   = nn.Linear(600,           1)

    def forward(self, s, a):
        # Q1
        s1  = F.relu(self.layer_1(s))
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a,  self.layer_2_a.weight.data.t())
        s1  = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1  = self.layer_3(s1)
        # Q2
        s2  = F.relu(self.layer_4(s))
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a,  self.layer_5_a.weight.data.t())
        s2  = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2  = self.layer_6(s2)
        return q1, q2


# ─────────────────────────────────────────────
# TD3 AGENT
# ─────────────────────────────────────────────
class td3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor        = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic        = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action  = max_action
        self.writer      = SummaryWriter(log_dir=runs_path)
        self.iter_count  = 0

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size  = 100,
        discount    = 0.99,   # FIX: was 0.99999 (≈1.0) → unstable Q-values
        tau         = 0.005,
        policy_noise= 0.2,
        noise_clip  = 0.5,
        policy_freq = 2,
    ):
        av_Q   = 0
        max_Q  = -inf
        av_loss = 0

        for it in range(iterations):
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)

            state      = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action     = torch.Tensor(batch_actions).to(device)
            reward     = torch.Tensor(batch_rewards).to(device)
            done       = torch.Tensor(batch_dones).to(device)

            # Target policy smoothing
            next_action = self.actor_target(next_state)
            noise       = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise       = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Clipped double Q-learning target
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q  = torch.min(target_Q1, target_Q2)
            av_Q     += torch.mean(target_Q)
            max_Q     = max(max_Q, torch.max(target_Q))
            target_Q  = reward + ((1 - done) * discount * target_Q).detach()

            current_Q1, current_Q2 = self.critic(state, action)
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            # Delayed policy update
            if it % policy_freq == 0:
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad    = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            av_loss += loss

        self.iter_count += 1
        env.get_logger().info(
            f"Train | Loss: {av_loss/iterations:.4f} | Av.Q: {av_Q/iterations:.4f} | Max.Q: {max_Q:.4f}"
        )
        self.writer.add_scalar("loss",   av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q",  av_Q    / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q,                self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(),  f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict( torch.load(f"{directory}/{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))


# ─────────────────────────────────────────────
# OBSTACLE / BOUNDS CHECK
# ─────────────────────────────────────────────
def check_pos(x, y):
    """Return True if (x, y) is a valid free position."""
    if -3.8 > x > -6.2 and 6.2 > y > 3.8:   return False
    if -1.3 > x > -2.7 and 4.7 > y > -0.2:  return False
    if -0.3 > x > -4.2 and 2.7 > y > 1.3:   return False
    if -0.8 > x > -4.2 and -2.3 > y > -4.2: return False
    if -1.3 > x > -3.7 and -0.8 > y > -2.7: return False
    if  4.2 > x >  0.8 and -1.8 > y > -3.2: return False
    if  4.0 > x >  2.5 and  0.7 > y > -3.2: return False
    if  6.2 > x >  3.8 and -3.3 > y > -4.2: return False
    if  4.2 > x >  1.3 and  3.7 > y >  1.5: return False
    if -3.0 > x > -7.2 and  0.5 > y > -1.5: return False
    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5: return False
    return True


# ─────────────────────────────────────────────
# GAZEBO ENVIRONMENT
# ─────────────────────────────────────────────
class GazeboEnv(Node):

    def __init__(self):
        super().__init__('env')
        self.environment_dim = 20
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.last_distance = 0.0

        self.goal_x = 1.0
        self.goal_y = 0.0

        self.upper = 5.0
        self.lower = -5.0

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        # Publishers / subscribers / service clients
        self.vel_pub   = self.create_publisher(Twist,      "/cmd_vel",                 1)
        self.set_state = self.create_publisher(ModelState, "gazebo/set_model_state",  10)

        self.unpause     = self.create_client(Empty, "/unpause_physics")
        self.pause       = self.create_client(Empty, "/pause_physics")
        self.reset_proxy = self.create_client(Empty, "/reset_world")

        self.publisher  = self.create_publisher(MarkerArray, "goal_point",       3)
        self.publisher2 = self.create_publisher(MarkerArray, "linear_velocity",  1)
        self.publisher3 = self.create_publisher(MarkerArray, "angular_velocity", 1)

    # ── STEP ──────────────────────────────────
    def step(self, action):
        """
        action : [linear_vel (0-1), angular_vel (-1 to 1)]
        Returns (state, reward, done, target)
        """
        global lidar_data, last_odom
        if last_odom is None:
            return np.zeros(self.environment_dim + 4), 0.0, False, False

        target = False

        # Publish velocity command
        vel_cmd           = Twist()
        vel_cmd.linear.x  = float(action[0])
        vel_cmd.angular.z = float(action[1])
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        # Un-pause physics
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /unpause_physics ...")
        try:
            self.unpause.call_async(Empty.Request())
        except Exception:
            self.get_logger().warn("/unpause_physics call failed")

        time.sleep(TIME_DELTA)

        # Pause physics
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /pause_physics ...")
        try:
            self.pause.call_async(Empty.Request())
        except Exception:
            self.get_logger().warn("/pause_physics call failed")

        # ── READ SENSOR DATA ──────────────────
        # lidar_data is in RAW METRES (set by Lidar_subscriber)
        laser_state = [list(lidar_data)]                            # 20 raw-metre readings
        done, collision, min_laser = self.observe_collision(lidar_data)

        # Odometry
        self.odom_x = last_odom.pose.pose.position.x
        self.odom_y = last_odom.pose.pose.position.y
        quaternion  = Quaternion(
            last_odom.pose.pose.orientation.w,
            last_odom.pose.pose.orientation.x,
            last_odom.pose.pose.orientation.y,
            last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Distance & heading to goal
        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        skew_x   = self.goal_x - self.odom_x
        skew_y   = self.goal_y - self.odom_y
        dot      = skew_x * 1 + skew_y * 0
        mag1     = math.sqrt(skew_x ** 2 + skew_y ** 2)
        beta     = math.acos(dot / (mag1 * 1.0 + 1e-8))
        if skew_y < 0:
            beta = -beta if skew_x < 0 else -beta
        theta = beta - angle
        if theta >  np.pi: theta = theta - 2 * np.pi
        if theta < -np.pi: theta = theta + 2 * np.pi

        # Goal check
        if distance < GOAL_REACHED_DIST:
            self.get_logger().info("GOAL REACHED!")
            target = True
            done   = True

        # Normalise lidar for the neural network (0–1 range)
        lidar_norm = lidar_data / 10.0
        robot_state = [distance, theta, action[0], action[1]]
        state       = np.append(lidar_norm, robot_state)

        reward = self.get_reward(target, collision, action, min_laser, theta)
        return state, reward, done, target

    # ── RESET ─────────────────────────────────
    def reset(self):
        global lidar_data

        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /reset_world ...")
        try:
            self.reset_proxy.call_async(Empty.Request())
        except rclpy.ServiceException as e:
            self.get_logger().warn(f"/reset_world failed: {e}")

        angle      = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        obj        = self.set_self_state

        x, y = 0.0, 0.0
        while not check_pos(x, y):
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)

        obj.pose.position.x    = x
        obj.pose.position.y    = y
        obj.pose.orientation.x = quaternion.x
        obj.pose.orientation.y = quaternion.y
        obj.pose.orientation.z = quaternion.z
        obj.pose.orientation.w = quaternion.w
        self.set_state.publish(obj)

        self.odom_x = x
        self.odom_y = y

        self.change_goal()
        self.random_box()
        self.publish_markers([0.0, 0.0])

        # Un-pause
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /unpause_physics ...")
        try:
            self.unpause.call_async(Empty.Request())
        except Exception:
            self.get_logger().warn("/unpause_physics call failed")

        time.sleep(TIME_DELTA)

        # Pause
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /pause_physics ...")
        try:
            self.pause.call_async(Empty.Request())
        except Exception:
            self.get_logger().warn("/pause_physics call failed")

        # Build initial state with normalised lidar
        lidar_norm = lidar_data / 10.0

        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        skew_x   = self.goal_x - self.odom_x
        skew_y   = self.goal_y - self.odom_y
        dot      = skew_x * 1 + skew_y * 0
        mag1     = math.sqrt(skew_x ** 2 + skew_y ** 2)
        beta     = math.acos(dot / (mag1 + 1e-8))
        if skew_y < 0:
            beta = -beta
        theta = beta - angle
        if theta >  np.pi: theta = theta - 2 * np.pi
        if theta < -np.pi: theta = theta + 2 * np.pi

        self.last_distance = distance

        robot_state = [distance, theta, 0.0, 0.0]
        state       = np.append(lidar_norm, robot_state)
        return state

    # ── GOAL PLACEMENT ────────────────────────
    def change_goal(self):
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False
        while not goal_ok:
            # Use absolute random position instead of robot-relative offset
            # to avoid placing goals outside valid bounds
            self.goal_x = np.random.uniform(-4.5, 4.5)
            self.goal_y = np.random.uniform(-4.5, 4.5)
            goal_ok     = check_pos(self.goal_x, self.goal_y)
            # Ensure goal is reachable (not too close to robot)
            if np.linalg.norm([self.goal_x - self.odom_x, self.goal_y - self.odom_y]) < 1.0:
                goal_ok = False

    # ── BOX RANDOMISATION ─────────────────────
    def random_box(self):
        for i in range(4):
            name   = f"cardboard_box_{i}"
            x, y   = 0.0, 0.0
            box_ok = False
            while not box_ok:
                x      = np.random.uniform(-6, 6)
                y      = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                if np.linalg.norm([x - self.odom_x,  y - self.odom_y])  < 1.5: box_ok = False
                if np.linalg.norm([x - self.goal_x,  y - self.goal_y])  < 1.5: box_ok = False

            box_state                   = ModelState()
            box_state.model_name        = name
            box_state.pose.position.x   = x
            box_state.pose.position.y   = y
            box_state.pose.position.z   = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    # ── MARKERS ───────────────────────────────
    def publish_markers(self, action):
        # Goal marker (green cylinder)
        ma1    = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type            = Marker.CYLINDER
        marker.action          = Marker.ADD
        marker.scale.x         = 0.1
        marker.scale.y         = 0.1
        marker.scale.z         = 0.01
        marker.color.a         = 1.0
        marker.color.g         = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        ma1.markers.append(marker)
        self.publisher.publish(ma1)

        # Linear velocity marker
        ma2     = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id  = "odom"
        marker2.type             = Marker.CUBE
        marker2.action           = Marker.ADD
        marker2.scale.x          = float(abs(action[0]))
        marker2.scale.y          = 0.1
        marker2.scale.z          = 0.01
        marker2.color.a          = 1.0
        marker2.color.r          = 1.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x  = 5.0
        ma2.markers.append(marker2)
        self.publisher2.publish(ma2)

        # Angular velocity marker
        ma3     = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id  = "odom"
        marker3.type             = Marker.CUBE
        marker3.action           = Marker.ADD
        marker3.scale.x          = float(abs(action[1]))
        marker3.scale.y          = 0.1
        marker3.scale.z          = 0.01
        marker3.color.a          = 1.0
        marker3.color.r          = 1.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x  = 5.0
        marker3.pose.position.y  = 0.2
        ma3.markers.append(marker3)
        self.publisher3.publish(ma3)

    # ── COLLISION DETECTION ───────────────────
    @staticmethod
    def observe_collision(laser_data):
        """
        laser_data : raw metres (NOT normalised).
        COLLISION_DIST is also in metres → comparison is correct.
        """
        min_laser = float(np.min(laser_data))
        if min_laser < COLLISION_DIST:
            env.get_logger().info(f"COLLISION detected! min_laser={min_laser:.3f}m")
            return True, True, min_laser
        return False, False, min_laser

    # ── REWARD ────────────────────────────────
    def get_reward(self, target, collision, action, min_laser, theta):
        """
        min_laser : raw metres.
        All reward components are in meaningful physical units.
        """
        if target:
            self.get_logger().info("Reward: +100 (Goal Reached)")
            return 100.0
        if collision:
            self.get_logger().info("Reward: -100 (Collision)")
            return -100.0

        # Survival bonus (small, keeps agent alive)
        r_survival = 0.1

        # Forward speed reward — encourages movement
        r_v = action[0] * 1.5

        # Heading reward — penalise facing away from goal
        r_theta = -abs(theta) / np.pi                       # range [-1, 0]

        # Distance-progress reward — most important signal
        current_dist = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        r_distance       = (self.last_distance - current_dist) * 10.0  # amplified
        self.last_distance = current_dist

        # Stagnation penalty — discourages near-zero linear speed
        r_stagnant = -0.5 if action[0] < 0.05 else 0.0

        # Obstacle proximity penalty (min_laser is in real metres)
        if min_laser < 1.0:
            r_obstacle = -0.5 / (min_laser + 0.1)          # sharper gradient near wall
        else:
            r_obstacle = 0.0

        total = r_survival + r_v + r_theta + r_distance + r_stagnant + r_obstacle
        return float(total)


# ─────────────────────────────────────────────
# ROS 2 SUBSCRIBER NODES
# ─────────────────────────────────────────────
class Odom_subscriber(Node):
    def __init__(self):
        super().__init__('odom_subscriber')
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

    def odom_callback(self, od_data):
        global last_odom
        last_odom = od_data


class Lidar_subscriber(Node):
    def __init__(self):
        super().__init__('lidar_subscriber')
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.environment_dim = 20

    def lidar_callback(self, msg):
        global lidar_data
        raw = np.array(msg.ranges, dtype=np.float32)

        # 1. Replace Inf / NaN with max range (10 m)
        raw[np.isinf(raw)] = 10.0
        raw[np.isnan(raw)] = 10.0
        raw = np.clip(raw, 0.0, 10.0)

        # 2. Sample 20 evenly-spaced indices from the front 160°
        #    Index 100-260 spans the front sector of a 360-point scan
        indices    = np.linspace(100, 260, self.environment_dim, dtype=int)
        # 3. Store RAW metres — normalisation happens in step() / reset()
        lidar_data = raw[indices]

        self.get_logger().info(
            f"LiDAR | Front-centre: {lidar_data[10]:.2f} m | Min: {np.min(lidar_data):.2f} m"
        )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':

    rclpy.init(args=None)

    # ── Hyper-parameters ──────────────────────
    seed                = 0
    eval_freq           = 5_000        # steps between evaluations
    max_ep              = 500          # max steps per episode
    eval_ep             = 10           # evaluation episodes
    max_timesteps       = 5_000_000
    expl_noise          = 1.0          # initial exploration noise
    expl_decay_steps    = 50_000
    expl_min            = 0.1
    batch_size          = 128
    discount            = 0.99         # FIX: was 0.99999 → now stable
    tau                 = 0.005
    policy_noise        = 0.2
    noise_clip          = 0.5
    policy_freq         = 2
    buffer_size         = 1_000_000
    file_name           = "td3_green_4wheel"
    save_model          = True
    load_model          = False
    random_near_obstacle= True

    # ── Paths ─────────────────────────────────
    script_dir   = os.path.dirname(os.path.realpath(__file__))
    models_path  = os.path.join(script_dir, "pytorch_models")
    results_path = os.path.join(script_dir, "results")
    runs_path    = os.path.join(script_dir, "runs")

    for p in [results_path, runs_path]:
        os.makedirs(p, exist_ok=True)
    if save_model:
        os.makedirs(models_path, exist_ok=True)

    # ── Network setup ─────────────────────────
    environment_dim = 20
    robot_dim       = 4
    state_dim       = environment_dim + robot_dim
    action_dim      = 2
    max_action      = 1

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create nodes BEFORE td3 (td3 uses runs_path which is now defined)
    env              = GazeboEnv()
    odom_subscriber  = Odom_subscriber()
    lidar_subscriber = Lidar_subscriber()

    network       = td3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(buffer_size, seed)

    if load_model:
        try:
            network.load(file_name, models_path)
            print("Loaded existing model.")
        except Exception as e:
            print(f"Could not load model ({e}). Starting fresh.")

    # ── Executor (multi-threaded for parallel spin) ──
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(lidar_subscriber)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # ── Wait for first sensor readings ────────
    env.get_logger().info("Waiting for Gazebo sensors (Odom & LiDAR) ...")
    while last_odom is None or np.all(lidar_data == 10.0):
        time.sleep(0.1)
    env.get_logger().info("Sensors ready. Starting training loop.")

    # ── Training state ────────────────────────
    evaluations          = []
    timestep             = 0
    timesteps_since_eval = 0
    episode_num          = 0
    done                 = True
    epoch                = 1
    episode_timesteps    = 0
    episode_reward       = 0.0
    state                = None

    count_rand_actions = 0
    random_action      = []

    try:
        while rclpy.ok() and timestep < max_timesteps:

            # ── Episode boundary ──
            if done:
                env.get_logger().info(
                    f"Episode {episode_num} ended | Steps: {timestep} | "
                    f"Ep.Reward: {episode_reward:.2f}"
                )

                if timestep != 0:
                    network.train(
                        replay_buffer,
                        episode_timesteps,
                        batch_size,
                        discount,
                        tau,
                        policy_noise,
                        noise_clip,
                        policy_freq,
                    )

                if timesteps_since_eval >= eval_freq:
                    env.get_logger().info("Running evaluation ...")
                    timesteps_since_eval %= eval_freq
                    evaluations.append(evaluate(network, epoch, eval_ep))
                    network.save(file_name, directory=models_path)
                    np.save(os.path.join(results_path, file_name), evaluations)
                    epoch += 1

                state          = env.reset()
                done           = False
                episode_reward = 0.0
                episode_timesteps = 0
                episode_num   += 1

            # ── Exploration noise decay ──
            if expl_noise > expl_min:
                expl_noise -= (1.0 - expl_min) / expl_decay_steps

            # ── Action selection ──
            action = network.get_action(np.array(state))
            action = np.clip(
                action + np.random.normal(0, expl_noise, size=action_dim),
                -max_action, max_action
            )

            # ── Random actions near obstacles ──
            # FIX: state[0:20] are normalised lidar (0-1); threshold 0.06 ≈ 0.6 m
            if random_near_obstacle:
                if (
                    np.random.uniform(0, 1) > 0.85
                    and min(state[0:20]) < 0.06          # 0.6 m in normalised units
                    and count_rand_actions < 1
                ):
                    count_rand_actions = np.random.randint(8, 15)
                    random_action      = np.random.uniform(-1, 1, 2)

                if count_rand_actions > 0:
                    count_rand_actions -= 1
                    action    = random_action.copy()
                    action[0] = -1                       # reverse to escape obstacle

            # ── Map action to physical range ──
            # Linear: [-1,1] → [0,1]   Angular: [-1,1] unchanged
            a_in = [(action[0] + 1) / 2, action[1]]

            next_state, reward, done, target = env.step(a_in)

            # Terminal flag for Bellman: don't bootstrap at true terminal states
            done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
            done      = 1 if episode_timesteps + 1 == max_ep else int(done)

            episode_reward += reward
            replay_buffer.add(state, action, reward, done_bool, next_state)

            state              = next_state
            episode_timesteps += 1
            timestep          += 1
            timesteps_since_eval += 1

    except KeyboardInterrupt:
        env.get_logger().info("Training interrupted by user.")

    finally:
        if save_model:
            network.save(file_name, directory=models_path)
        rclpy.shutdown()
