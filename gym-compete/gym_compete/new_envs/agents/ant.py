from .agent import Agent
from gym.spaces import Box
import numpy as np

class Ant(Agent):

    def __init__(self, agent_id, xml_path=None):
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "assets", "ant_body.xml")
        super(Ant, self).__init__(agent_id, xml_path)

    def set_goal(self, goal):
        self.GOAL = goal
        self.move_left = False
        if self.get_qpos()[0] > 0:
            self.move_left = True

    def before_step(self):
        self._xposbefore = self.get_body_com("torso")[0]

    def after_step(self, action):
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - self._xposbefore) / self.env.dt
        if self.move_left:
            forward_reward *= -1
        ctrl_cost = .5 * np.square(action).sum()
        cfrc_ext = self.get_cfrc_ext()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(cfrc_ext, -1, 1))
        )
        qpos = self.get_qpos()
        agent_standing = qpos[2] >= 0.28
        survive = 1.0 if agent_standing else -1.
        reward = forward_reward - ctrl_cost - contact_cost + survive

        reward_info = dict()
        reward_info['reward_forward'] = forward_reward
        reward_info['reward_ctrl'] = ctrl_cost
        reward_info['reward_contact'] = contact_cost
        reward_info['reward_survive'] = survive
        reward_info['reward_move'] = reward

        done = not agent_standing

        return reward, done, reward_info


    def _get_obs(self):
        '''
        Return agent's observations
        '''
        my_pos = self.get_qpos()
        other_pos = self.get_other_qpos()
        my_vel = self.get_qvel()
        cfrc_ext = np.clip(self.get_cfrc_ext(), -1, 1)

        obs = np.concatenate(
            [my_pos.flat, my_vel.flat, cfrc_ext.flat,
             other_pos.flat]
        )
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
