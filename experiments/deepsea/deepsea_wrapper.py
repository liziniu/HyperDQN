import gym
import numpy as np

try:
    from ray.tune import registry
except ImportError:
    registry = None


class DeepSeaEnv(gym.Env):
    """
    Deep sea example.

    For more information, see papers:
    [1] https://arxiv.org/abs/1703.07608
    [2] https://arxiv.org/abs/1806.03335
    """
    def __init__(self, config, randomize_actions=True):
        super().__init__()

        self._size = config["size"]
        self._deterministic = config.get("deterministic", True)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self._size, self._size), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(n=2)

        self._row = 0
        self._column = 0
        self._unscaled_move_cost = config.get("move_cost", 0.01)
        if randomize_actions:
            self._mapping_rng = np.random.RandomState(42)
            self._action_mapping = self._mapping_rng.binomial(1, 0.5, [self._size, self._size])
        else:
            self._action_mapping = np.ones([self._size, self._size])

        assert self._unscaled_move_cost * self._size <= 1, (
            "Please decrease the move cost. Otherwise the optimal decision is not go right."
        )

        self._num_state = (1 + self._size) * self._size / 2
        self._num_state = int(self._num_state)
        assert isinstance(self._num_state, int)
        self._num_action = 2
        self._horizon = self._size

    def reset(self):
        self._row = 0
        self._column = 0
        return self._get_observation()

    def _get_observation(self):
        obs = np.zeros(shape=(self._size, self._size), dtype=np.float32)
        if self._row >= self._size:  # End of episode null observation
            return obs
        obs[self._row, self._column] = 1.
        return obs

    def step(self, action: int):
        reward = 0.
        action_right = action == self._action_mapping[self._row, self._column]

        # Reward calculation
        if self._column == self._size - 1 and action_right:
            reward += 1.
        if not self._deterministic:  # Noisy rewards on the 'end' of chain.
            if self._row == self._size - 1 and self._column in [0, self._size - 1]:
                reward += np.random.rand()

        if action_right:
            if np.random.rand() > 1 / self._size or self._deterministic:
                self._column = np.clip(self._column + 1, 0, self._size - 1)
            reward -= self._unscaled_move_cost / self._size
        else:
            # You were on the right path and went wrong
            self._column = np.clip(self._column - 1, 0, self._size - 1)
        self._row += 1

        observation = self._get_observation()
        done = False
        info = {}
        if self._row == self._size:
            done = True

        return observation, reward, done, info

    def render(self, mode='human'):
        raise NotImplementedError

    def generate_reward_matrix(self):
        R = np.zeros([self.num_state, self.num_action], dtype=np.float32)

        index = self._coordinate_to_index(self._size - 1, self._size - 1)
        R[index, 1] = 1.
        R[:, 1] -= self._unscaled_move_cost / self._size
        return R

    def generate_transition_matrix(self):
        T = np.zeros([self.num_state, self.num_action, self.num_state], dtype=np.float32)
        for row in range(self._size):
            for col in range(row + 1):
                state = self._coordinate_to_index(row, col)

                if row < self._size - 1:
                    # left
                    next_state = self._coordinate_to_index(min(row + 1, self._size - 1),  max(col - 1, 0))
                    T[state, 0, next_state] = 1.
                    # right
                    next_state = self._coordinate_to_index(min(row + 1, self._size - 1), min(col + 1, self._size - 1))
                    T[state, 1, next_state] = 1.
                else:
                    # the first state
                    T[state, :, 0] = 1.

        np.testing.assert_allclose(np.sum(T, axis=-1), 1.)
        return T

    @staticmethod
    def _coordinate_to_index(row, col):
        index = np.sum(np.arange(row + 1)) + col
        return index

    @property
    def num_state(self):
        return self._num_state

    @property
    def num_action(self):
        return self._num_action

    @property
    def horizon(self):
        return self._horizon


class FlattenObservationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(env.observation_space, gym.spaces.Box)

        self.observation_space = gym.spaces.Box(low=env.observation_space.low.min(),
                                                high=env.observation_space.high.max(),
                                                shape=(np.prod(env.observation_space.shape), ),
                                                dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        return self.observation(self.env.reset(**kwargs))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def observation(self, observation):
        return observation.reshape(-1)


# Register Env in Ray
if registry:
    registry.register_env(
        "deep_sea",
        lambda config: DeepSeaEnv(config)
    )

