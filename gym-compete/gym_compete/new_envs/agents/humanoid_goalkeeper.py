from .agent import Agent
from .humanoid_kicker import HumanoidKicker
from gym.spaces import Box
import numpy as np


def mass_center(mass, xpos):
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class HumanoidGoalKeeper(HumanoidKicker):

    def __init__(self, agent_id, xml_path=None, team='blocker', goal_x=4, goal_y=3):
        super(HumanoidGoalKeeper, self).__init__(agent_id, xml_path)
        self.team = team
        self.GOAL_X = goal_x
        self.GOAL_Y = goal_y

    def before_step(self):
        pass

    def after_step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        forward_reward = 0.
        ctrl_cost = .1 * np.square(action).sum()
        cfrc_ext = self.get_cfrc_ext()
        contact_cost = .5e-6 * np.square(cfrc_ext).sum()
        contact_cost = min(contact_cost, 10)
        qpos = self.get_qpos()
        agent_standing = qpos[2] >= 1.0
        at_goal = (qpos[0] <= self.GOAL_X + 0.05) and (abs(qpos[1]) <= self.GOAL_Y)
        survive = 5.0 if agent_standing and at_goal else -5.
        reward = forward_reward - ctrl_cost - contact_cost + survive

        reward_info = dict()
        reward_info['reward_forward'] = forward_reward
        reward_info['reward_ctrl'] = ctrl_cost
        reward_info['reward_contact'] = contact_cost
        reward_info['reward_survive'] = survive
        reward_info['reward_move'] = reward
        # print("keeper", reward_info)

        done = bool(qpos[2] <= 0.5)

        return reward, done, reward_info

    def _get_obs(self):
        obs = super(HumanoidGoalKeeper, self)._get_obs()
        assert np.isfinite(obs).all(), "Humanoid Keeper observation is not finite!!"
        return obs

    def reached_goal(self):
        return False

    def reset_agent(self):
        xpos = self.get_qpos()[0]
        # NOTE: for keeper Target is on same side as the agent
        if xpos * self.TARGET < 0 :
            self.set_goal(-self.TARGET)
