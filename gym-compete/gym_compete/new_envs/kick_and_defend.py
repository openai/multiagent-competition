from .multi_agent_env import MultiAgentEnv
import numpy as np
from .agents import Agent
import six
from gym import spaces

class KickAndDefend(MultiAgentEnv):
    def __init__(self, max_episode_steps=500, randomize_ball=True, **kwargs):
        super(KickAndDefend, self).__init__(**kwargs)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self.GOAL_REWARD = 1000
        self.ball_jnt_id = self.env_scene.model.joint_names.index(six.b('ball'))
        self.jnt_nqpos = Agent.JNT_NPOS[int(self.env_scene.model.jnt_type[self.ball_jnt_id])]
        if self.agents[0].team == 'walker':
            self.walker_id = 0
            self.blocker_id = 1
        else:
            self.walker_id = 1
            self.blocker_id = 0
        self.GOAL_X = self.agents[self.walker_id].TARGET
        # print("GOAL_X:", self.GOAL_X)
        self.GOAL_Y = 3
        self.randomize_ball = randomize_ball
        self.LIM_X = [(-3.5, -0.5), (1.6, 3.5)]
        # self.RANGE_X = self.LIM_X = [(-2, -2), (1.6, 3.5)]
        self.LIM_Y = [(-2, 2), (-2, 2)]
        self.RANGE_X = self.LIM_X.copy()
        self.RANGE_Y = self.LIM_Y.copy()
        self.BALL_LIM_X = (-2, 1)
        self.BALL_LIM_Y = (-4, 4)
        self.BALL_RANGE_X = self.BALL_LIM_X
        self.BALL_RANGE_Y = self.BALL_LIM_Y
        self.keeper_touched_ball = False

    def _past_limit(self):
        if self._max_episode_steps <= self._elapsed_steps:
            return True
        return False

    def get_ball_qpos(self):
        start_idx = int(self.env_scene.model.jnt_qposadr[self.ball_jnt_id])
        return self.env_scene.model.data.qpos[start_idx:start_idx+self.jnt_nqpos]

    def get_ball_qvel(self):
        start_idx = int(self.env_scene.model.jnt_dofadr[self.ball_jnt_id])
        # ball has 6 components: 3d translation, 3d rotational
        return self.env_scene.model.data.qvel[start_idx:start_idx+6]

    def get_ball_contacts(self, agent_id):
        mjcontacts = self.env_scene.data._wrapped.contents.contact
        ncon = self.env_scene.model.data.ncon
        contacts = []
        for i in range(ncon):
            ct = mjcontacts[i]
            g1 , g2 = ct.geom1, ct.geom2
            g1 = self.env_scene.model.geom_names[g1]
            g2 = self.env_scene.model.geom_names[g2]
            if g1.find(six.b('ball')) >= 0:
                if g2.find(six.b('agent' + str(agent_id))) >= 0:
                    if ct.dist < 0:
                        contacts.append((g1, g2, ct.dist))
        return contacts

    def _set_ball_xyz(self, xyz):
        start = int(self.env_scene.model.jnt_qposadr[self.ball_jnt_id])
        qpos = self.env_scene.model.data.qpos.flatten().copy()
        qpos[start:start+3] = xyz
        qvel = self.env_scene.model.data.qvel.flatten()
        self.env_scene.set_state(qpos, qvel)

    def is_goal(self):
        ball_xyz = self.get_ball_qpos()[:3]
        if self.GOAL_X > 0 and ball_xyz[0] > self.GOAL_X and abs(ball_xyz[1]) <= self.GOAL_Y:
            return True
        elif self.GOAL_X < 0 and ball_xyz[0] < self.GOAL_X and abs(ball_xyz[1]) <= self.GOAL_Y:
            return True
        return False

    def goal_rewards(self, infos=None, agent_dones=None):
        self._elapsed_steps += 1
        # print(self._elapsed_steps, self.keeper_touched_ball)
        goal_rews = [0. for _ in range(self.n_agents)]
        ball_xyz = self.get_ball_qpos()[:3]
        done = self._past_limit() or (self.GOAL_X > 0 and ball_xyz[0] > self.GOAL_X) or (self.GOAL_X < 0 and ball_xyz[0] < self.GOAL_X)
        ball_vel = self.get_ball_qvel()[:3]
        if ball_vel[0] < 0 and np.linalg.norm(ball_vel) > 1:
            done = True
            # print("Keeper stopped ball, vel:", ball_vel)
        # agent_fallen = [self.agents[i].get_qpos()[2] < 0.5 for i in range(self.n_agents)]
        # import ipdb; ipdb.set_trace()
        ball_contacts = self.get_ball_contacts(self.blocker_id)
        if len(ball_contacts) > 0:
            # print("detected contacts for keeper:", ball_contacts)
            self.keeper_touched_ball = True
        if self.is_goal():
            for i in range(self.n_agents):
                if self.agents[i].team == 'walker':
                    goal_rews[i] += self.GOAL_REWARD
                    infos[i]['winner'] = True
                else:
                    goal_rews[i] -= self.GOAL_REWARD
            done = True
        elif done or all(agent_dones):
            for i in range(self.n_agents):
                if self.agents[i].team == 'walker':
                        goal_rews[i] -= self.GOAL_REWARD
                else:
                    goal_rews[i] += self.GOAL_REWARD
                    infos[i]['winner'] = True
                    if self.keeper_touched_ball:
                        # ball contact bonus
                        goal_rews[i] += 0.5 * self.GOAL_REWARD
                    if self.agents[i].get_qpos()[2] > 0.8:
                        # standing bonus
                        goal_rews[i] += 0.5 * self.GOAL_REWARD
        else:
            keeper_penalty = False
            for i in range(self.n_agents):
                if self.agents[i].team == 'blocker':
                    if np.abs(self.GOAL_X - self.agents[i].get_qpos()[0]) > 2.5:
                        keeper_penalty = True
                        # print("keeper x:", self.agents[i].get_qpos()[0], "goal_x:", self.GOAL_X)
                        print("Keeper foul!")
                        break
            if keeper_penalty:
                done = True
                for i in range(self.n_agents):
                    if self.agents[i].team == 'blocker':
                        goal_rews[i] -= self.GOAL_REWARD
            else:
                for i in range(self.n_agents):
                    if self.agents[i].team == 'walker':
                        # goal_rews[i] -= np.abs(ball_xyz[0] - self.GOAL_X)
                        infos[i]['reward_move'] -= np.asscalar(np.abs(ball_xyz[0] - self.GOAL_X))
                    else:
                        infos[i]['reward_move'] += np.asscalar(np.abs(ball_xyz[0] - self.GOAL_X))
                        # if len(ball_contacts) > 0:
                        #     # ball contact bonus
                        #     print("detected contacts for keeper:", ball_contacts)
                        #     goal_rews[i] += 0.5 * self.GOAL_REWARD
        return goal_rews, done

    def _set_ball_vel(self, vel_xyz):
        start = int(self.env_scene.model.jnt_dofadr[self.ball_jnt_id])
        qvel = self.env_scene.model.data.qvel.flatten().copy()
        qvel[start:start+len(vel_xyz)] = vel_xyz
        qpos = self.env_scene.model.data.qpos.flatten()
        self.env_scene.set_state(qpos, qvel)

    def _set_random_ball_pos(self):
        x = np.random.uniform(*self.BALL_RANGE_X)
        y = np.random.uniform(*self.BALL_RANGE_Y)
        z = 0.35
        # print("setting ball to {}".format((x, y, z)))
        self._set_ball_xyz((x,y,z))
        if self.get_ball_qvel()[0] < 0:
            self._set_ball_vel((0.1, 0.1, 0.1))

    def _reset_range(self, version):
        decay_func = lambda x: 0.05 * np.exp(0.001 * x)
        v = decay_func(version)
        self.BALL_RANGE_X = (max(self.BALL_LIM_X[0], -v), min(self.BALL_LIM_X[1], v))
        self.BALL_RANGE_Y = (max(self.BALL_LIM_Y[0], -v), min(self.BALL_LIM_Y[1], v))
        self.RANGE_X[0] = (max(self.LIM_X[0][0], -2-v),  min(self.LIM_X[0][1], -2+v))
        self.RANGE_Y[0] = (max(self.LIM_Y[0][0], -v),  min(self.LIM_Y[0][1], v))
        self.RANGE_X[1] = (max(self.LIM_X[1][0], 2-v),  min(self.LIM_X[1][1], 2+v))
        self.RANGE_Y[1] = (max(self.LIM_Y[1][0], -v),  min(self.LIM_Y[1][1], v))
        # print(self.RANGE_X)
        # print(self.RANGE_Y)
        # print(self.BALL_RANGE_X)
        # print(self.BALL_RANGE_Y)

    def _reset(self, version=None):
        self._elapsed_steps = 0
        self.keeper_touched_ball = False
        _ = self.env_scene.reset()
        if version is not None:
            self._reset_range(version)
        for i in range(self.n_agents):
            x = np.random.uniform(*self.RANGE_X[i])
            y = np.random.uniform(*self.RANGE_Y[i])
            # print("setting agent {} to pos {}".format(i, (x,y)))
            self.agents[i].set_xyz((x, y, None))
            self.agents[i].reset_agent()
        if self.randomize_ball:
            self._set_random_ball_pos()
        return self._get_obs()

    def reset(self, margins=None, version=None):
        _ = self._reset(version=version)
        if self.agents[0].team == 'walker':
            self.walker_id = 0
            self.blocker_id = 1
        else:
            self.walker_id = 1
            self.blocker_id = 0
        self.GOAL_X = self.agents[self.walker_id].TARGET
        if margins is not None:
            for i in range(self.n_agents):
                self.agents[i].set_margin(margins[i])
        # print("GOAL_X:", self.GOAL_X)
        ob = self._get_obs()
        return ob
