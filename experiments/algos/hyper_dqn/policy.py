import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, Union, Optional, List, Callable

from tianshou.policy import BasePolicy, DQNPolicy
from tianshou.policy.base import _nstep_return
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy, to_torch


class HyperDQNPolicy(DQNPolicy):
    """HyperDQN.
    There is a hyper model that takes a random vector z as the input and outputs the parameter for the base model.
    In the meantime, the base model is substantiated and takes the state as the input and outputs the q values and
    the associated greedy action.
    """
    def __init__(
        self,
        model,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        num_train_iter: int = 10,
        noise_scale: float = 0.01,
        l2_norm: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(model, optim, discount_factor, estimation_step, target_update_freq, reward_normalization,
                         **kwargs)
        self._num_train_iter = num_train_iter
        self._noise_scale = noise_scale
        self._l2_norm = l2_norm
        self._l2_norm_old = l2_norm
        self._z_train = None
        self._z_test = None

    def set_prior_scale(self, prior_scale: float, env_step: int = None):
        if hasattr(self.model, "prior_model"):
            self.model.prior_scale = prior_scale
            self.model_old.prior_scale = prior_scale
        elif hasattr(self.model, "hypermodel"):
            self.model.hypermodel.prior_scale = prior_scale
            self.model_old.hypermodel.prior_scale = prior_scale
        else:
            raise NotImplementedError
        if env_step:
            self._l2_norm = self._l2_norm_old / (env_step + 1)

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        # training
        if self.training:
            if self._z_train is None or len(batch.done.shape) == 0 or batch.done[0]:
                obs = batch[input]
                obs_ = obs.obs if hasattr(obs, "obs") else obs
                assert len(obs_) == 1, (
                    "Current only support one actor mode."
                )
                self._z_train = self.model.generate_z(len(obs_))
            return self._model_forward(batch, self._z_train, state, model, input)
        else:
            if self._z_test is None or len(batch.done.shape) == 0 or batch.done[0]:
                obs = batch[input]
                obs_ = obs.obs if hasattr(obs, "obs") else obs
                assert len(obs_) == 1, (
                    "Current only support one actor mode."
                )
                self._z_test = self.model.generate_z(len(obs_))
            return self._model_forward(batch, self._z_test, state, model, input)

    def _model_forward(
        self,
        batch: Batch,
        z: torch.Tensor,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
    ) -> Batch:
        model = getattr(self, model)
        obs = batch[input]
        obs_ = obs.obs if hasattr(obs, "obs") else obs

        logits, h = model(obs_, z, state=state, info=batch.info)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))  # (None, num_action) or (None, bz, num_action)
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[-1]
        act = to_numpy(q.max(dim=-1)[1])
        return Batch(logits=logits, act=act, state=h)

    def _target_q_func(
        self, buffer: ReplayBuffer, indice: np.ndarray, z: torch.Tensor,
    ) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs_next: s_{t+n}
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            a = self._model_forward(batch, z, input="obs_next").act  # (None, bz)
            with torch.no_grad():
                target_q = self._model_forward(
                    batch, z, model="model_old", input="obs_next"
                ).logits    # (None, bz, num_action)
            a_one_hot = F.one_hot(torch.as_tensor(a, device=target_q.device), self.max_action_num).to(torch.float32)
            target_q = torch.sum(target_q * a_one_hot, dim=-1)  # (None, bz)
        else:
            with torch.no_grad():
                target_q = self._model_forward(batch, z, input="obs_next").logits.max(dim=-1)[0]    # (None, bz)
        return target_q

    def _learn(
        self,
        batch: Batch,
        **kwargs: Any
    ) -> Dict[str, Union[float, List[float]]]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()

        z = batch.z   # (bz, nz)
        self.optim.zero_grad()
        q = self._model_forward(batch, z).logits    # (None, bz, num_action)
        if len(q.shape) == 2:
            q = q[:, None, :]
        a_one_hot = F.one_hot(
            torch.as_tensor(batch.act, device=q.device), self.max_action_num
        ).to(torch.float32)     # (None, num_action)
        q = torch.einsum('bka,ba->bk', q, a_one_hot)    # (None, bz)

        noise_a = to_torch_as(batch.noise, q)   # (None, nz)
        noise_z = torch.einsum('bn,kn->bk', noise_a, z) * self._noise_scale  # (None, bz)

        r = to_torch_as(batch.returns, q)
        td = r + noise_z - q
        loss = td.pow(2).mean()
        reg_loss = self.model.get_hyper_trainable_output(z).pow(2).mean() * self._l2_norm
        loss = loss + reg_loss

        loss.backward()
        self.optim.step()
        self._iter += 1

        result = {
            "hyper/loss": loss.item(),
            "hyper/reg_loss": reg_loss.item(),
            "hyper/noise": noise_z.abs().mean().item(),
            "hyper/td": td.mean().item(),
            "hyper/q": q.mean().item(),
            "hyper/q_std": q.std(dim=1).mean().item(),
        }

        return result

    def update(
        self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any
    ) -> Dict[str, Any]:
        """Update the policy network and replay buffer.
        """
        if buffer is None or len(buffer) < sample_size:
            return {}
        batch, indice = buffer.sample(sample_size * self._num_train_iter)
        self.updating = True
        batch = self.process_fn(batch, buffer, indice)
        result = self._learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indice)
        self.updating = False
        return result

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        """Compute the n-step return for Q-learning targets.
        """
        z = self.model.generate_z(self._num_train_iter)
        batch.z = z

        batch = self.compute_nstep_return(
            batch, buffer, indice, self._target_q_func,
            self._gamma, self._n_step, self._rew_norm)

        return batch

    @staticmethod
    def compute_nstep_return(
        batch: Batch,
        buffer: ReplayBuffer,
        indice: np.ndarray,
        target_q_fn: Callable[[ReplayBuffer, np.ndarray, torch.Tensor], torch.Tensor],
        gamma: float = 0.99,
        n_step: int = 1,
        rew_norm: bool = False,
    ) -> Batch:
        assert not rew_norm, \
            "Reward normalization in computing n-step returns is unsupported now."
        rew = buffer.rew
        bsz = len(indice)
        indices = [indice]
        for _ in range(n_step - 1):
            indices.append(buffer.next(indices[-1]))
        indices = np.stack(indices)
        # terminal indicates buffer indexes nstep after 'indice',
        # and are truncated at the end of each episode
        terminal = indices[-1]
        with torch.no_grad():
            target_q_torch = target_q_fn(buffer, terminal, batch.z)  # (None, bz)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        target_qs = []
        if len(target_q_torch.shape) == 1:
            target_q_torch = target_q_torch[:, None]
        for k in range(target_q_torch.shape[1]):
            target_q = to_numpy(target_q_torch[:, k].reshape(bsz, -1))
            target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
            target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step)
            target_q = to_torch_as(target_q.flatten(), target_q_torch)
            target_qs.append(target_q)

        batch.returns = torch.stack(target_qs, dim=1)    # (None, bz)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return batch


