from .agent import Agent
from .humanoid import Humanoid
from gym.spaces import Box
import numpy as np


def mass_center(mass, xpos):
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class HumanoidBlocker(Humanoid):

    def __init__(self, agent_id, xml_path=None, team='blocker'):
        super(HumanoidBlocker, self).__init__(agent_id, xml_path)
        self.team = team

    def before_step(self):
        pass

    def after_step(self, action):
        forward_reward = 0.
        ctrl_cost = .1 * np.square(action).sum()
        cfrc_ext = self.get_cfrc_ext()
        contact_cost = .5e-6 * np.square(cfrc_ext).sum()
        contact_cost = min(contact_cost, 10)
        qpos = self.get_qpos()
        agent_standing = qpos[2] >= 1.0
        survive = 5.0 if agent_standing else -5.
        reward = forward_reward - ctrl_cost - contact_cost + survive
        # reward = survive

        reward_info = dict()
        reward_info['reward_forward'] = forward_reward
        reward_info['reward_ctrl'] = ctrl_cost
        reward_info['reward_contact'] = contact_cost
        reward_info['reward_survive'] = survive
        reward_info['reward_move'] = reward

        done = bool(qpos[2] <= 0.5)

        return reward, done, reward_info

    def reached_goal(self):
        return False
