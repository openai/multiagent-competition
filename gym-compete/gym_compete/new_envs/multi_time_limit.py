import gym
from gym.wrappers import TimeLimit

class MultiTimeLimit(TimeLimit):

    def _step(self, action):
        assert self._episode_started_at is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._past_limit():
            if self.metadata.get('semantics.autoreset'):
                _ = self.reset() # automatically reset the env
            done = tuple([True for _ in range(len(observation))])

        return observation, reward, done, info
