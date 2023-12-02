import gymnasium as gym
from gymnasium import spaces
import pybullet as p
from qibullet import SimulationManager
from qibullet.robot_posture import NaoPosture
import numpy as np
from humanoid_envs.envs.control_functions.fase2 import fase2
from humanoid_envs.envs.control_functions.fase3 import fase3
import pandas as pd
import time
from humanoid_envs.envs.control_functions.transformacao import transformacao
from humanoid_envs.envs.control_functions.dualQuatMult import dualQuatMult
from humanoid_envs.envs.control_functions.getPositionDualQuat import getPositionDualQuat
from humanoid_envs.envs.control_functions.kinematicRobo import kinematicRobo


class NaoEnv(gym.Env):
    def __init__(self, gui=False, action_space_size=2, obs_type="xyz", reward_type="delta"):
        super(NaoEnv, self).__init__()
        self.simulation_manager = SimulationManager()
        self.client = self.simulation_manager.launchSimulation(gui=gui, use_shared_memory=False)
        self.robot = self.simulation_manager.spawnNao(self.client, spawn_ground_plane=True)
        self.simulation_manager.setLightPosition(self.client, [0, 0, 100])
        self.left = True
        self.step_counter = 0
        time.sleep(2)

        # stand pose parameters
        pose = NaoPosture('StandInit')
        pose_dict = {}
        for joint_name, joint_value in zip(pose.joint_names, pose.joint_values):
            pose_dict[joint_name] = joint_value

        # joint parameters
        self.joint_names = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'RShoulderPitch',
                            'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw',
                            'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll',
                            'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll']

        self.init_angles = []
        for joint_name in self.joint_names:
            self.init_angles.append(pose_dict[joint_name])
        self.link_names = []
        for joint_name in self.joint_names:
            linkName = p.getJointInfo(self.robot.getRobotModel(), self.robot.joint_dict[joint_name].getIndex())[
                12].decode("utf-8")
            self.link_names.append(linkName)
        for joint_name in self.joint_names:
            linkName = p.getJointInfo(self.robot.getRobotModel(), self.robot.joint_dict[joint_name].getIndex())[
                12].decode("utf-8")
            self.link_names.append(linkName)

        self.action_space_size = action_space_size
        if action_space_size == 2:
            upper_limits = np.array([[1, 1]]).T
            lower_limits = np.array([[0, 0]]).T
            self.action_space = spaces.Box(low=lower_limits, high=upper_limits, shape=(2, 1), dtype=np.float64)
        elif action_space_size == 7:
            lower_limits = np.array([[0, 0, -1, -1, -1, -1, 0]]).T
            self.action_space = spaces.Box(low=lower_limits, high=1, shape=(7, 1), dtype=np.float64)

        self.obs_type = obs_type
        if obs_type == "xyz":
            upper_limits_obs = np.array([[1, 1, 1, 1, 1, 1, 10]]).T
            self.observation_space = spaces.Box(low=-1, high=upper_limits_obs, shape=(7, 1), dtype=np.float64)
        elif obs_type == "angles":
            self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(12, 1), dtype=np.float64)

        self.reward_type = reward_type

        self.L1 = 0.05
        L2 = 0.085
        L3 = 0.100
        L4 = 0.10274
        self.z_max = L4 + L3 + L2
        self.step_size_max = 0.08  # http://doc.aldebaran.com/2-1/naoqi/motion/control-walk.html

        # LQR and proportional gains
        S1 = 0
        Q1 = 10  # New gains: 12
        R1 = 0.000012  # New gains: 0.000015
        K1 = 1100  # New gains: 900
        self.Gain = np.array([S1, Q1, R1, K1]).reshape((4, 1))
        self.right_theta = pd.read_csv('humanoid-envs/humanoid_envs/envs/control_functions/thetaRight1.csv')
        self.left_theta = pd.read_csv('humanoid-envs/humanoid_envs/envs/control_functions/thetaLeft1.csv')
        self.theta = np.zeros((6, 2))
        self.init_pos()

        self.init_state = p.saveState()
        self.previous_robot_x = 0
        self.first_observation = self.get_observation()
        time.sleep(1)

    def get_observation(self):
        angles = self.robot.getAnglesPosition(self.joint_names)

        # Inverts 'LHipYawPitch' e 'RHipYawPitch'
        angles[10] = -angles[10]
        angles[16] = -angles[16]
        theta = np.zeros((6, 2))
        theta[:, 1] = angles[10:16]
        theta[:, 0] = angles[16:22]
        n = np.array([0, 1, 0])
        thetab = np.pi / 2
        realRb = np.array([np.cos(thetab / 2)])
        hOrg = np.array([1, 0, 0, 0, 0, 0, 0, 0]).reshape((8, 1))
        rb = np.concatenate((realRb, np.sin(thetab / 2) * n), axis=0).reshape((4, 1))
        pb = np.array([[0], [0], [0], [0]])

        # base B para a base O6 (perna em movimento)
        hB_O6 = transformacao(pb, rb)
        hP = dualQuatMult(hOrg, hB_O6)
        ha2 = kinematicRobo(theta, hOrg, hP, 1, 0)
        ha = kinematicRobo(theta, hOrg, hP, 1, 1)

        pos_CoM = getPositionDualQuat(ha)
        pos_foot = getPositionDualQuat(ha2)
        x_robot_value = self.get_robot_x()

        if self.obs_type == "xyz":
            observation = np.concatenate((
                pos_CoM,
                pos_foot,
                np.reshape(x_robot_value, (1, 1))
            ), axis=0)
        elif self.obs_type == "angles":
            observation = np.concatenate((theta[:, 0], theta[:, 1]), axis=None).reshape(-1, 1)
        return observation

    def init_pos(self):
        for k in range(0, len(self.right_theta), 10):
            Vr = self.right_theta.iloc[k].to_numpy()
            Vl = self.left_theta.iloc[k].to_numpy()

            joints = self.init_angles
            joints[10:16] = Vl[:]
            joints[16:22] = Vr[:]

            # Inverts 'LHipYawPitch' e 'RHipYawPitch'
            joints[10] = -joints[10]
            joints[16] = -joints[16]

            self.robot.setAngles(self.joint_names, joints, 1)
            self.simulation_manager.stepSimulation(self.client)

        angles = self.robot.getAnglesPosition(self.joint_names)
        invertidos = ['LHipYawPitch', 'RHipYawPitch']
        for invertido in invertidos:
            indice = self.joint_names.index(invertido)
            angles[indice] = -angles[indice]

        self.theta[:, 1] = angles[10:16]
        self.theta[:, 0] = angles[16:22]
        z = self.robot.getLinkPosition("Head")[0][2]

    def CoM_trajectory(self, sin_params, n_pts):
        a1 = sin_params[0]
        v1 = sin_params[1]
        c1 = sin_params[2]
        a2 = sin_params[3]
        c2 = sin_params[4]

        if v1 == 0:
            v1 = 1e-8
        x_max = v1  # * self.step_size_max
        x_max_divided_by_n_pts = x_max / n_pts
        x = np.arange(0, x_max + x_max_divided_by_n_pts, x_max_divided_by_n_pts)

        inner_sin = (2 * np.pi / v1) * x + c1
        z = a1 * np.sin(inner_sin) - a1 + self.z_max
        inner_sin = (np.pi / v1) * x + c2
        y = a2 * np.sin(inner_sin)  # + a2
        CoM_traj = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)

        return CoM_traj

    def foot_trajectory(self, sin_params, n_pts):
        a3 = sin_params[0]
        v3 = sin_params[1]
        if v3 == 0:
            v3 = 1e-8
        x_max = v3
        x_max_divided_by_ts = x_max / n_pts
        x = np.arange(0, x_max, x_max_divided_by_ts)
        if self.left:
            y = -0.003 * np.sin((np.pi / v3) * x)
        else:
            y = 0.003 * np.sin((np.pi / v3) * x)
        z = a3 * np.sin((np.pi / v3) * x)
        y = np.zeros_like(x)
        foot_traj = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1).reshape(-1, 3)
        return foot_traj

    def denormalize(self, sin_params):
        v1 = sin_params[1]
        v3 = sin_params[6]
        if self.action_space_size == 7:
            sin_params[0] = sin_params[0] * self.z_max / 20
            sin_params[3] = sin_params[3] * self.L1
            sin_params[5] = sin_params[5] * self.z_max / 4
            sin_params[2] = np.pi * (sin_params[2])

        sin_params[6] = (v3 * self.step_size_max)
        sin_params[1] = (v1 * self.step_size_max)

        if self.left:
            sin_params[4] = np.pi + (sin_params[4]) * np.pi / 8
        else:
            sin_params[4] = 0 + (sin_params[4]) * np.pi / 8

        return sin_params

    def get_robot_x(self):
        l_ankle = self.robot.getLinkPosition("l_ankle")[0][0]
        r_ankle = self.robot.getLinkPosition("r_ankle")[0][0]
        return (l_ankle + r_ankle) / 2

    def step(self, actions, joints=None):
        if self.action_space_size == 7:
            acoes = actions
        elif self.action_space_size == 2:
            acoes = np.array([[-0.0001],
                              [actions[0][0]],
                              [0.],
                              [0.015],
                              [0],
                              [0.015],
                              [actions[1][0]]])

        self.step_counter += 1

        if isinstance(acoes, np.ndarray):
            acoes = [ac[0] for ac in acoes.tolist()]
        acoes = self.denormalize(acoes)

        # set joint angles
        params_CoM = [acoes[0], acoes[1], acoes[2], acoes[3], acoes[4]]
        params_foot = [acoes[5], acoes[6]]

        # calculate the trajectories
        CoM_traj = self.CoM_trajectory(sin_params=params_CoM, n_pts=300)
        foot_traj = self.foot_trajectory(sin_params=params_foot, n_pts=300)

        if self.left:
            # CoM_traj[:, 1] = CoM_traj[:, 1] + 2 * self.L1
            [ha, ha2, self.theta, CoM_pos, feet_pos, t] = fase3(CoM_traj, np.size(foot_traj, 0), foot_traj,
                                                                self.theta, self.Gain, self.robot,
                                                                self.simulation_manager,
                                                                self.client, self.joint_names, self.init_angles)
            self.left = False
        else:
            # CoM_traj[:, 1] = CoM_traj[:, 1] + self.L1
            [ha, ha2, self.theta, CoM_pos, feet_pos, t] = fase2(CoM_traj, np.size(foot_traj, 0), foot_traj,
                                                                self.theta, self.Gain, self.robot,
                                                                self.simulation_manager,
                                                                self.client, self.joint_names, self.init_angles)
            self.left = True

        x_robot_value = self.get_robot_x()
        delta_x = x_robot_value - self.previous_robot_x
        self.previous_robot_x = x_robot_value

        if self.obs_type == "xyz":
            observation = np.concatenate((
                CoM_pos,
                feet_pos,
                np.reshape(x_robot_value, (1, 1))
            ), axis=0)
        elif self.obs_type == "angles":
            observation = np.concatenate((self.theta[:, 0], self.theta[:, 1]), axis=None).reshape(-1, 1)

        done = False
        info = {"x_robot_value": x_robot_value}

        if t == -1:
            done = True
            print("")
            return observation, -1.0, done, False, info

        if self.reward_type == "delta":
            reward = delta_x / 0.06
        elif self.reward_type == "delta_inv_timestep":
            reward = (delta_x / 0.06) + (1 / self.step_counter)

        if self.step_counter % 10 == 0:
            print(f"Step ({self.step_counter}): x_robot_value: {round(x_robot_value, 3)}, reward: {round(reward, 3)}")

        if self.step_counter == 100:  # Limited to 100 steps
            print("")
            done = True

        return observation, reward, done, False, info

    def reset(self, seed=None, options=None):
        p.restoreState(self.init_state)
        self.init_pos()
        self.step_counter = 0
        self.left = True
        self.previous_robot_x = self.get_robot_x()
        info = {"x_robot_value": self.previous_robot_x}

        return self.get_observation(), info

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()
