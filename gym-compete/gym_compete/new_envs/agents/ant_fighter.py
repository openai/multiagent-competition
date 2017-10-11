from .agent import Agent
from .ant import Ant
from gym.spaces import Box
import numpy as np
import six


def mass_center(mass, xpos):
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class AntFighter(Ant):

    def __init__(self, agent_id, xml_path=None, team='ant'):
        super(AntFighter, self).__init__(agent_id, xml_path)
        self.team = team

    def before_step(self):
        pass

    def set_env(self, env):
        super(AntFighter, self).set_env(env)
        self.arena_id = self.env.model.geom_names.index(six.b('arena'))
        self.arena_height = self.env.model.geom_size[self.arena_id][1] * 2

    def after_step(self, action):
        ctrl_cost = .1 * np.square(action).sum()
        cfrc_ext = self.get_cfrc_ext()
        contact_cost = .5e-6 * np.square(cfrc_ext).sum()
        contact_cost = min(contact_cost, 10)
        qpos = self.get_qpos()
        center_reward = - np.sqrt(np.sum((0. - qpos[:2])**2))
        agent_standing = qpos[2] - self.arena_height >=  0.28
        survive = 5.0 if agent_standing else -5.
        reward = center_reward - ctrl_cost - contact_cost + survive
        # reward = survive

        reward_info = dict()
        # reward_info['reward_forward'] = forward_reward
        reward_info['reward_center'] = center_reward
        reward_info['reward_ctrl'] = ctrl_cost
        reward_info['reward_contact'] = contact_cost
        reward_info['reward_survive'] = survive
        reward_info['reward_move'] = reward

        done = bool(qpos[2] - self.arena_height <= 0.28)

        return reward, done, reward_info

    def _get_obs(self):
        '''
        Return agent's observations
        '''
        qpos = self.get_qpos()
        my_pos = qpos
        other_qpos = self.get_other_qpos()
        other_pos = other_qpos
        other_relative_xy = other_qpos[:2] - qpos[:2]

        vel = self.get_qvel()
        cfrc_ext = np.clip(self.get_cfrc_ext(), -1, 1)

        torso_xmat = self.get_torso_xmat()

        obs = np.concatenate(
            [my_pos.flat, vel.flat, cfrc_ext.flat,
             other_pos.flat, other_relative_xy.flat,
             torso_xmat.flat
            ]
        )
        assert np.isfinite(obs).all(), "Ant observation is not finite!!"
        return obs

    def reached_goal(self):
        return False
