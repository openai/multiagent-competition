from .humanoid import Humanoid
from gym.spaces import Box
import numpy as np
from .agent import Agent
import six


def mass_center(mass, xpos):
    return (np.sum(mass * xpos, 0) / np.sum(mass))


class HumanoidKicker(Humanoid):

    def __init__(self, agent_id, xml_path=None):
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid_body.xml")
        super(HumanoidKicker, self).__init__(agent_id, xml_path)
        self.team = 'walker'
        self.TARGET = 4 if agent_id == 0 else -4
        self.TARGET_Y = 3

    def set_env(self, env):
        self.ball_jnt_id = env.model.joint_names.index(six.b('ball'))
        self.ball_jnt_nqpos = Agent.JNT_NPOS[int(env.model.jnt_type[self.ball_jnt_id])]
        super(HumanoidKicker, self).set_env(env)

    def get_ball_qpos(self):
        start_idx = int(self.env.model.jnt_qposadr[self.ball_jnt_id])
        return self.env.model.data.qpos[start_idx:start_idx+self.ball_jnt_nqpos]

    def get_ball_qvel(self):
        start_idx = int(self.env.model.jnt_dofadr[self.ball_jnt_id])
        # ball has 6 components: 3d translation, 3d rotational
        return self.env.model.data.qvel[start_idx:start_idx+6]

    def set_goal(self, goal):
        ball_ini_xyz = self.get_ball_qpos()
        self.GOAL = np.asscalar(ball_ini_xyz[0])
        self.TARGET = goal
        self.move_left = False
        if self.get_qpos()[0] - self.GOAL > 0:
            self.move_left = True
    
    def after_step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # print(action)
        _, done, rinfo = super(HumanoidKicker, self).after_step(action)
        ball_xy = self.get_ball_qpos()[:2]
        my_xy = self.get_qpos()[:2]
        ball_dist = np.sqrt(np.sum((my_xy - ball_xy)**2))
        rinfo['reward_goal_dist'] = np.asscalar(ball_dist)
        reward = rinfo['reward_forward'] - rinfo['reward_ctrl'] - rinfo['reward_contact'] + rinfo['reward_survive'] - rinfo['reward_goal_dist']
        rinfo['reward_move'] = reward
        assert np.isfinite(reward), (rinfo, action)
        return reward, done, rinfo

    def _get_obs(self):
        state = super(HumanoidKicker, self)._get_obs_relative()
        ball_xyz = self.get_ball_qpos()[:3]
        relative_xy = ball_xyz[:2] - self.get_qpos()[:2]
        relative_xyz = np.concatenate([relative_xy.flat, ball_xyz[2].flat])

        ball_goal_dist = self.TARGET - ball_xyz[0]
        ball_qvel = self.get_ball_qvel()[:3]
        ball_goal_y_dist1 = np.asarray(self.TARGET_Y - ball_xyz[1])
        ball_goal_y_dist2 = np.asarray(-self.TARGET_Y - ball_xyz[1])

        obs = np.concatenate([state.flat, relative_xyz.flat, np.asarray(ball_goal_dist).flat, ball_goal_y_dist1.flat, ball_goal_y_dist2.flat])
        assert np.isfinite(obs).all(), "Humanoid Kicker observation is not finite!!"
        return obs

    def reached_goal(self):
        return False

    def reset_agent(self):
        pass
