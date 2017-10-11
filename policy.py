import tensorflow as tf
import numpy as np
import gym
import logging
import copy

from tensorflow.contrib import layers


class Policy(object):
    def reset(self, **kwargs):
        pass

    def act(self, observation):
        # should return act, info
        raise NotImplementedError()

class RunningMeanStd(object):
    def __init__(self, scope="running", reuse=False, epsilon=1e-2, shape=()):
        with tf.variable_scope(scope, reuse=reuse):
            self._sum = tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(0.0),
                name="sum", trainable=False)
            self._sumsq = tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(epsilon),
                name="sumsq", trainable=False)
            self._count = tf.get_variable(
                dtype=tf.float32,
                shape=(),
                initializer=tf.constant_initializer(epsilon),
                name="count", trainable=False)
            self.shape = shape

            self.mean = tf.to_float(self._sum / self._count)
            var_est = tf.to_float(self._sumsq / self._count) - tf.square(self.mean)
            self.std = tf.sqrt(tf.maximum(var_est, 1e-2))


def dense(x, size, name, weight_init=None, bias=True):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w)
    if bias:
        b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
        return ret + b
    else:
        return ret

def switch(condition, if_exp, else_exp):
    x_shape = copy.copy(if_exp.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),
                lambda: if_exp,
                lambda: else_exp)
    x.set_shape(x_shape)
    return x


class DiagonalGaussian(object):
    def __init__(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    def mode(self):
        return self.mean


class MlpPolicyValue(Policy):
    def __init__(self, scope, *, ob_space, ac_space, hiddens, convs=[], reuse=False, normalize=False):
        self.recurrent = False
        self.normalized = normalize
        self.zero_state = np.zeros(1)
        with tf.variable_scope(scope, reuse=reuse):
            self.scope = tf.get_variable_scope().name

            assert isinstance(ob_space, gym.spaces.Box)

            self.observation_ph = tf.placeholder(tf.float32, [None] + list(ob_space.shape), name="observation")
            self.stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
            self.taken_action_ph = tf.placeholder(dtype=tf.float32, shape=[None, ac_space.shape[0]], name="taken_action")

            if self.normalized:
                if self.normalized != 'ob':
                    self.ret_rms = RunningMeanStd(scope="retfilter")
                self.ob_rms = RunningMeanStd(shape=ob_space.shape, scope="obsfilter")

            obz = self.observation_ph
            if self.normalized:
                obz = tf.clip_by_value((self.observation_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            last_out = obz
            for i, hid_size in enumerate(hiddens):
                last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i" % (i + 1)))
            self.vpredz = dense(last_out, 1, "vffinal")[:, 0]

            self.vpred = self.vpredz
            if self.normalized and self.normalized != 'ob':
                self.vpred = self.vpredz * self.ret_rms.std + self.ret_rms.mean  # raw = not standardized

            last_out = obz
            for i, hid_size in enumerate(hiddens):
                last_out = tf.nn.tanh(dense(last_out, hid_size, "polfc%i" % (i + 1)))
            mean = dense(last_out, ac_space.shape[0], "polfinal")
            logstd = tf.get_variable(name="logstd", shape=[1, ac_space.shape[0]], initializer=tf.zeros_initializer())

            self.pd = DiagonalGaussian(mean, logstd)
            self.sampled_action = switch(self.stochastic_ph, self.pd.sample(), self.pd.mode())

    def make_feed_dict(self, observation, taken_action):
        return {
            self.observation_ph: observation,
            self.taken_action_ph: taken_action
        }

    def act(self, observation, stochastic=True):
        outputs = [self.sampled_action, self.vpred]
        a, v = tf.get_default_session().run(outputs, {
            self.observation_ph: observation[None],
            self.stochastic_ph: stochastic})
        return a[0], {'vpred': v[0]}

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class LSTMPolicy(Policy):
    def __init__(self, scope, *, ob_space, ac_space, hiddens, reuse=False, normalize=False):
        self.recurrent = True
        self.normalized = normalize
        with tf.variable_scope(scope, reuse=reuse):
            self.scope = tf.get_variable_scope().name

            assert isinstance(ob_space, gym.spaces.Box)

            self.observation_ph = tf.placeholder(tf.float32, [None, None] + list(ob_space.shape), name="observation")
            self.stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
            self.taken_action_ph = tf.placeholder(dtype=tf.float32, shape=[None, None, ac_space.shape[0]], name="taken_action")

            if self.normalized:
                if self.normalized != 'ob':
                    self.ret_rms = RunningMeanStd(scope="retfilter")
                self.ob_rms = RunningMeanStd(shape=ob_space.shape, scope="obsfilter")

            obz = self.observation_ph
            if self.normalized:
                obz = tf.clip_by_value((self.observation_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            last_out = obz
            for hidden in hiddens[:-1]:
                last_out = tf.contrib.layers.fully_connected(last_out, hidden)
            self.zero_state = []
            self.state_in_ph = []
            self.state_out = []
            cell = tf.contrib.rnn.BasicLSTMCell(hiddens[-1], reuse=reuse)
            size = cell.state_size
            self.zero_state.append(np.zeros(size.c, dtype=np.float32))
            self.zero_state.append(np.zeros(size.h, dtype=np.float32))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.c], name="lstmv_c"))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.h], name="lstmv_h"))
            initial_state = tf.contrib.rnn.LSTMStateTuple(self.state_in_ph[-2], self.state_in_ph[-1])
            last_out, state_out = tf.nn.dynamic_rnn(cell, last_out, initial_state=initial_state, scope="lstmv")
            self.state_out.append(state_out)

            self.vpredz = tf.contrib.layers.fully_connected(last_out, 1, activation_fn=None)[:, :, 0]
            self.vpred = self.vpredz
            if self.normalized and self.normalized != 'ob':
                self.vpred = self.vpredz * self.ret_rms.std + self.ret_rms.mean  # raw = not standardized

            last_out = obz
            for hidden in hiddens[:-1]:
                last_out = tf.contrib.layers.fully_connected(last_out, hidden)
            cell = tf.contrib.rnn.BasicLSTMCell(hiddens[-1], reuse=reuse)
            size = cell.state_size
            self.zero_state.append(np.zeros(size.c, dtype=np.float32))
            self.zero_state.append(np.zeros(size.h, dtype=np.float32))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.c], name="lstmp_c"))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.h], name="lstmp_h"))
            initial_state = tf.contrib.rnn.LSTMStateTuple(self.state_in_ph[-2], self.state_in_ph[-1])
            last_out, state_out = tf.nn.dynamic_rnn(cell, last_out, initial_state=initial_state, scope="lstmp")
            self.state_out.append(state_out)

            mean = tf.contrib.layers.fully_connected(last_out, ac_space.shape[0], activation_fn=None)
            logstd = tf.get_variable(name="logstd", shape=[1, ac_space.shape[0]], initializer=tf.zeros_initializer())

            self.pd = DiagonalGaussian(mean, logstd)
            self.sampled_action = switch(self.stochastic_ph, self.pd.sample(), self.pd.mode())

            self.zero_state = np.array(self.zero_state)
            self.state_in_ph = tuple(self.state_in_ph)
            self.state = self.zero_state

            for p in self.get_trainable_variables():
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.square(p)))

    def make_feed_dict(self, observation, state_in, taken_action):
        return {
            self.observation_ph: observation,
            self.state_in_ph: list(np.transpose(state_in, (1, 0, 2))),
            self.taken_action_ph: taken_action
        }

    def act(self, observation, stochastic=True):
        outputs = [self.sampled_action, self.vpred, self.state_out]
        a, v, s = tf.get_default_session().run(outputs, {
            self.observation_ph: observation[None, None],
            self.state_in_ph: list(self.state[:, None, :]),
            self.stochastic_ph: stochastic})
        self.state = []
        for x in s:
            self.state.append(x.c[0])
            self.state.append(x.h[0])
        self.state = np.array(self.state)
        return a[0, 0], {'vpred': v[0, 0], 'state': self.state}

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def reset(self):
        self.state = self.zero_state
