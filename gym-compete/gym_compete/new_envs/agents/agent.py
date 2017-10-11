import xml.etree.ElementTree as ET
from gym.spaces import Box
import six
from ..utils import list_filter
import numpy as np

class Agent(object):
    '''
    Super class for all agents in a multi-agent mujoco environement
    Each subclass shoudl implement a move_reward method which are the moving
    rewards for that agent
    Each agent can also implement its own action space
    Over-ride set_observation_space to change observation_space
    (default is Box based on _get_obs() implementation)
    After creation, an Env reference should be given calling set_env
    '''
    JNT_NPOS = {0: 7,
                1: 4,
                2: 1,
                3: 1,
                }

    def __init__(self, agent_id, xml_path, nagents=2):
        self.id = agent_id
        self.scope = 'agent' + str(self.id)
        self._xml_path = xml_path
        print("Reading agent XML from:", xml_path)
        self.xml = ET.parse(xml_path)
        self.env = None
        self._env_init = False
        self.n_agents = nagents

    def set_env(self, env):
        self.env = env
        self._env_init = True
        self._set_body()
        self._set_joint()
        if self.n_agents > 1:
            self._set_other_joint()
        self.set_observation_space()
        self.set_action_space()

    def set_observation_space(self):
        obs = self._get_obs
        self.obs_dim = obs.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = Box(low, high)

    def set_action_space(self):
        acts = self.xml.find('actuator')
        self.action_dim = len(list(acts))
        default = self.xml.find('default')
        range_set = False
        if default is not None:
            motor = default.find('motor')
            if motor is not None:
                ctrl = motor.get('ctrlrange')
                if ctrl:
                    clow, chigh = list(map(float, ctrl.split()))
                    high = chigh * np.ones(self.action_dim)
                    low = clow * np.ones(self.action_dim)
                    range_set = True
        if not range_set:
            high = np.inf * np.ones(self.action_dim)
            low = - high
        for i, motor in enumerate(list(acts)):
            ctrl = motor.get('ctrlrange')
            if ctrl:
                clow, chigh = list(map(float, ctrl.split()))
                high[i] = chigh
                low[i] = clow
        self._low = low
        self._high = high
        self.action_space = Box(low, high)

    # @property
    # def observation_space(self):
    #     return self.observation_space
    #
    # @property
    # def action_space(self):
    #     return self.action_space

    def in_scope(self, name):
        return name.startswith(six.b(self.scope))

    def in_agent_scope(self, name, agent_id):
        return name.startswith(six.b('agent' + str(agent_id)))

    def _set_body(self):
        self.body_names = list_filter(
            lambda x: self.in_scope(x),
            self.env.model.body_names
        )
        self.body_ids = [self.env.model.body_names.index(body)
                         for body in self.body_names]
        self.body_dofnum = self.env.model.body_dofnum[self.body_ids]
        self.nv = self.body_dofnum.sum()
        self.body_dofadr = self.env.model.body_dofadr[self.body_ids]
        dof = list_filter(lambda x: x >= 0, self.body_dofadr)
        self.qvel_start_idx = int(dof[0])
        last_dof_body_id = self.body_dofnum.shape[0] - 1
        while self.body_dofnum[last_dof_body_id] == 0:
            last_dof_body_id -= 1
        self.qvel_end_idx = int(dof[-1] + self.body_dofnum[last_dof_body_id])


    def _set_joint(self):
        self.join_names = list_filter(
            lambda x: self.in_scope(x), self.env.model.joint_names
        )
        self.joint_ids = [self.env.model.joint_names.index(body)
                          for body in self.join_names]
        self.jnt_qposadr = self.env.model.jnt_qposadr[self.joint_ids]
        self.jnt_type = self.env.model.jnt_type[self.joint_ids]
        self.jnt_nqpos = [self.JNT_NPOS[int(j)] for j in self.jnt_type]
        self.nq = sum(self.jnt_nqpos)
        self.qpos_start_idx = int(self.jnt_qposadr[0])
        self.qpos_end_idx = int(self.jnt_qposadr[-1] + self.jnt_nqpos[-1])

        # self.jnt_dofadr = self.env.model.jnt_dofadr[self.joint_ids]
        # dof = list_filter(lambda x: x >= 0, self.jnt_dofadr)
        # self.qvel_start_idx = int(dof[0])
        # last_dof_body_id = self.body_dofnum.shape[0] - 1
        # while self.body_dofnum[last_dof_body_id] == 0:
        #     last_dof_body_id -= 1
        # self.qvel_end_idx = int(dof[-1] + self.body_dofnum[last_dof_body_id])

    def _set_other_joint(self):
        self._other_qpos_idx = {}
        for i in range(self.n_agents):
            if i == self.id: continue
            other_join_names = list_filter(
                lambda x: self.in_agent_scope(x, i), self.env.model.joint_names
            )
            other_joint_ids = [self.env.model.joint_names.index(body)
                               for body in other_join_names]
            other_jnt_qposadr = self.env.model.jnt_qposadr[other_joint_ids]
            jnt_type = self.env.model.jnt_type[other_joint_ids]
            jnt_nqpos = [self.JNT_NPOS[int(j)] for j in jnt_type]
            nq = sum(jnt_nqpos)
            qpos_start_idx = int(other_jnt_qposadr[0])
            qpos_end_idx = int(other_jnt_qposadr[-1] + jnt_nqpos[-1])
            assert nq == qpos_end_idx - qpos_start_idx, (i, nq, qpos_start_idx, qpos_end_idx)
            self._other_qpos_idx[i] = (qpos_start_idx, qpos_end_idx)

    def get_other_agent_qpos(self):
        other_qpos = {}
        for i in range(self.n_agents):
            if i == self.id: continue
            startid, endid = self._other_qpos_idx[i]
            qpos = self.env.model.data.qpos[startid: endid]
            other_qpos[i] = qpos
        return other_qpos

    def before_step(self):
        raise NotImplementedError

    def after_step(self):
        raise NotImplementedError

    def _get_obs(self):
        raise NotImplementedError

    def get_body_com(self, body_name):
        assert self._env_init, "Env reference is not set"
        idx = self.body_ids[self.body_names.index(six.b(self.scope + '/' + body_name))]
        return self.env.model.data.com_subtree[idx]

    def get_cfrc_ext(self):
        assert self._env_init, "Env reference is not set"
        return self.env.model.data.cfrc_ext[self.body_ids]

    def depricated_get_qpos(self):
        qpos = np.zeros((self.nq, 1))
        cnt = 0
        for j, start_idx in enumerate(self.jnt_qposadr):
            jlen = self.jnt_nqpos[j]
            qpos[cnt: cnt + jlen] = self.env.model.data.qpos[start_idx: start_idx + jlen]
            cnt += jlen
        return qpos

    def get_qpos(self):
        '''
        Note: this relies on the qpos for one agent being contiguously located
        this is generally true, use depricated_get_qpos if not
        '''
        return self.env.model.data.qpos[self.qpos_start_idx: self.qpos_end_idx]

    def get_other_qpos(self):
        '''
        Note: this relies on the qpos for one agent being contiguously located
        this is generally true, use depricated_get_qpos if not
        '''
        left_part = self.env.model.data.qpos[:self.qpos_start_idx]
        right_part = self.env.model.data.qpos[self.qpos_end_idx:]
        return np.concatenate((left_part, right_part), axis=0)

    def get_qvel(self):
        '''
        Note: this relies on the qvel for one agent being contiguously located
        this is generally true, follow depricated_get_qpos if not
        '''
        return self.env.model.data.qvel[self.qvel_start_idx: self.qvel_end_idx]

    def get_qfrc_actuator(self):
        return self.env.model.data.qfrc_actuator[self.qvel_start_idx: self.qvel_end_idx]

    def get_cvel(self):
        return self.env.model.data.cvel[self.body_ids]

    def get_body_mass(self):
        return self.env.model.body_mass[self.body_ids]

    def get_xipos(self):
        return self.env.model.data.xipos[self.body_ids]

    def get_cinert(self):
        return self.env.model.data.cinert[self.body_ids]

    def get_xmat(self):
        return self.env.model.data.xmat[self.body_ids]

    def get_torso_xmat(self):
        return self.env.model.data.xmat[self.body_ids[self.body_names.index(six.b('agent%d/torso' % self.id))]]

    # def get_ctrl(self):
    #     return self.env.model.data.ctrl[self.joint_ids]

    def set_xyz(self, xyz):
        '''
        Set (x, y, z) position of the agent any element can be None
        '''
        assert any(xyz)
        start = self.qpos_start_idx
        qpos = self.env.model.data.qpos.flatten().copy()
        if xyz[0]:
            qpos[start] = xyz[0]
        if xyz[1]:
            qpos[start+1] = xyz[1]
        if xyz[2]:
            qpos[start+2] = xyz[2]
        qvel = self.env.model.data.qvel.flatten()
        self.env.set_state(qpos, qvel)

    def set_margin(self, margin):
        agent_geom_ids = [i for i, name in enumerate(self.env.model.geom_names)
                          if self.in_scope(name)]
        m = self.env.model.geom_margin.copy()
        print("Resetting", self.scope, "margins to", margin)
        m[agent_geom_ids] = margin
        self.env.model.__setattr__('geom_margin', m)

    def reached_goal(self):
        '''
        Override this
        '''
        raise NotImplementedError

    def set_goal(self):
        '''
        Override if needed, this called when initializing the agent
        and also if goal needs to be changed on reset
        '''
        pass

    def reset_agent(self):
        '''Override this'''
        pass
