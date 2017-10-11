from .agent import Agent
from gym.spaces import Box
import numpy as np


def mass_center(mass, xpos):
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class Humanoid(Agent):

    def __init__(self, agent_id, xml_path=None, **kwargs):
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid_body.xml")
        super(Humanoid, self).__init__(agent_id, xml_path, **kwargs)
        self.team = 'walker'

    def set_goal(self, goal):
        self.GOAL = goal
        self.move_left = False
        if self.get_qpos()[0] > 0:
            self.move_left = True

    def before_step(self):
        self._pos_before = mass_center(self.get_body_mass(), self.get_xipos())

    def after_step(self, action):
        pos_after = mass_center(self.get_body_mass(), self.get_xipos())
        forward_reward = 0.25 * (pos_after - self._pos_before) / self.env.model.opt.timestep
        if self.move_left:
            forward_reward *= -1
        ctrl_cost = .1 * np.square(action).sum()
        cfrc_ext = self.get_cfrc_ext()
        contact_cost = .5e-6 * np.square(cfrc_ext).sum()
        contact_cost = min(contact_cost, 10)
        qpos = self.get_qpos()
        agent_standing = qpos[2] >= 1.0
        survive = 5.0 if agent_standing else -5.
        reward = forward_reward - ctrl_cost - contact_cost + survive
        reward_goal = - np.abs(np.asscalar(qpos[0]) - self.GOAL)
        reward += reward_goal

        reward_info = dict()
        reward_info['reward_forward'] = forward_reward
        reward_info['reward_ctrl'] = ctrl_cost
        reward_info['reward_contact'] = contact_cost
        reward_info['reward_survive'] = survive
        if self.team == 'walker':
            reward_info['reward_goal_dist'] = reward_goal
        reward_info['reward_move'] = reward

        # done = not agent_standing
        done = qpos[2] < 0.8

        return reward, done, reward_info


    def _get_obs(self):
        '''
        Return agent's observations
        '''
        my_pos = self.get_qpos()
        other_pos = self.get_other_qpos()
        vel = self.get_qvel()
        cfrc_ext = np.clip(self.get_cfrc_ext(), -1, 1)
        cvel = self.get_cvel()
        cinert = self.get_cinert()
        qfrc_actuator = self.get_qfrc_actuator()

        obs = np.concatenate(
            [my_pos.flat, vel.flat,
             cinert.flat, cvel.flat,
             qfrc_actuator.flat, cfrc_ext.flat,
             other_pos.flat]
        )
        assert np.isfinite(obs).all(), "Humanoid observation is not finite!!"
        return obs

    def _get_obs_relative(self):
        '''
        Return agent's observations, positions are relative
        '''
        qpos = self.get_qpos()
        my_pos = qpos[2:]
        other_agents_qpos = self.get_other_agent_qpos()
        all_other_qpos = []
        for i in range(self.n_agents):
            if i == self.id: continue
            other_qpos = other_agents_qpos[i]
            other_relative_xy = other_qpos[:2] - qpos[:2]
            other_qpos = np.concatenate([other_relative_xy.flat, other_qpos[2:].flat], axis=0)
            all_other_qpos.append(other_qpos)
        all_other_qpos = np.concatenate(all_other_qpos)

        vel = self.get_qvel()
        cfrc_ext = np.clip(self.get_cfrc_ext(), -1, 1)
        cvel = self.get_cvel()
        cinert = self.get_cinert()
        qfrc_actuator = self.get_qfrc_actuator()

        obs = np.concatenate(
            [my_pos.flat, vel.flat,
             cinert.flat, cvel.flat,
             qfrc_actuator.flat, cfrc_ext.flat,
             all_other_qpos.flat]
        )
        assert np.isfinite(obs).all(), "Humanoid observation is not finite!!"
        return obs

    def set_observation_space(self):
        obs = self._get_obs()
        self.obs_dim = obs.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = Box(low, high)

    def reached_goal(self):
        xpos = self.get_body_com('torso')[0]
        if self.GOAL > 0 and xpos > self.GOAL:
            return True
        elif self.GOAL < 0 and xpos < self.GOAL:
            return True
        return False

    def reset_agent(self):
        xpos = self.get_qpos()[0]
        if xpos * self.GOAL > 0 :
            self.set_goal(-self.GOAL)
        if xpos > 0:
            self.move_left = True
        else:
            self.move_left = False
