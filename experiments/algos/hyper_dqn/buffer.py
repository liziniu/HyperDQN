import numpy as np
from typing import Any, Dict, List, Tuple, Union, Optional

from tianshou.data import VectorReplayBuffer as VectorReplayBufferBase, Batch
from tianshou.data.buffer.manager import _create_value, _alloc_by_keys_diff


class VectorReplayBuffer(VectorReplayBufferBase):
    def __init__(
            self,
            size: int,
            buffer_num: int,
            mask_prob: float,
            noise_dim: int,
            num_ensemble: int,
            **kwargs: Any,
    ) -> None:
        super().__init__(size, buffer_num, **kwargs)

        self._mask_prob = mask_prob
        self._num_ensemble = num_ensemble
        self._noise_dim = noise_dim

        assert 0 <= self._mask_prob <= 1.

        for buffer in self.buffers:
            buffer._reserved_keys = ("obs", "act", "rew", "done", "obs_next", "info", "policy",
                                     "ensemble_mask", "noise")

    def add(
        self, batch: Batch, buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Add a batch of data into ReplayBufferManager.

        Each of the data's length (first dimension) must equal to the length of
        buffer_ids. By default buffer_ids is [0, 1, ..., buffer_num - 1].

        Return (current_index, episode_reward, episode_length, episode_start_index). If
        the episode is not finished, the return value of episode_length and
        episode_reward is 0.
        """
        # preprocess batch
        b = Batch()
        for key in set(self._reserved_keys).intersection(batch.keys()):
            b.__dict__[key] = batch[key]
        batch = b
        assert set(["obs", "act", "rew", "done"]).issubset(batch.keys())
        if self._save_only_last_obs:
            batch.obs = batch.obs[:, -1]
        if not self._save_obs_next:
            batch.pop("obs_next", None)
        elif self._save_only_last_obs:
            batch.obs_next = batch.obs_next[:, -1]

        ensemble_mask = np.random.binomial(
            1, self._mask_prob, [self.buffer_num, self._num_ensemble],
        ).astype(batch.done.dtype)
        batch.ensemble_mask = ensemble_mask
        noise = np.random.randn(self.buffer_num, self._noise_dim).astype(np.float32)
        noise /= np.linalg.norm(noise, axis=1)
        batch.noise = noise

        # get index
        if buffer_ids is None:
            buffer_ids = np.arange(self.buffer_num)
        ptrs, ep_lens, ep_rews, ep_idxs = [], [], [], []
        for batch_idx, buffer_id in enumerate(buffer_ids):
            ptr, ep_rew, ep_len, ep_idx = self.buffers[buffer_id]._add_index(
                batch.rew[batch_idx], batch.done[batch_idx]
            )
            ptrs.append(ptr + self._offset[buffer_id])
            ep_lens.append(ep_len)
            ep_rews.append(ep_rew)
            ep_idxs.append(ep_idx + self._offset[buffer_id])
            self.last_index[buffer_id] = ptr + self._offset[buffer_id]
            self._lengths[buffer_id] = len(self.buffers[buffer_id])
        ptrs = np.array(ptrs)
        try:
            self._meta[ptrs] = batch
        except ValueError:
            batch.rew = batch.rew.astype(float)
            batch.done = batch.done.astype(bool)
            if self._meta.is_empty():
                self._meta = _create_value(  # type: ignore
                    batch, self.maxsize, stack=False)
            else:  # dynamic key pops up in batch
                _alloc_by_keys_diff(self._meta, batch, self.maxsize, False)
            self._set_batch_for_children()
            self._meta[ptrs] = batch
        return ptrs, np.array(ep_rews), np.array(ep_lens), np.array(ep_idxs)

    def __getitem__(self, index: Union[slice, int, List[int], np.ndarray]) -> Batch:
        """Return a data batch: self[index].

        If stack_num is larger than 1, return the stacked obs and obs_next with shape
        (batch, len, ...).
        """
        if isinstance(index, slice):  # change slice to np array
            # buffer[:] will get all available data
            indice = self.sample_index(0) if index == slice(None) \
                else self._indices[:len(self)][index]
        else:
            indice = index
        # raise KeyError first instead of AttributeError,
        # to support np.array([ReplayBuffer()])
        obs = self.get(indice, "obs")
        if self._save_obs_next:
            obs_next = self.get(indice, "obs_next", Batch())
        else:
            obs_next = self.get(self.next(indice), "obs", Batch())
        return Batch(
            obs=obs,
            act=self.act[indice],
            rew=self.rew[indice],
            done=self.done[indice],
            obs_next=obs_next,
            info=self.get(indice, "info", Batch()),
            policy=self.get(indice, "policy", Batch()),
            ensemble_mask=self.ensemble_mask[indice],
            noise=self.noise[indice]
        )
