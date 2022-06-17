import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Union, Optional, Sequence
from tianshou.data import to_torch


class LinearPriorNet(nn.Module):
    """
    Linear model to incorporate prior distribution information.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        prior_mean: float or np.ndarray = 0.,
        prior_std: float or np.ndarray = 1.,
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()

        # (fan-out, fan-in)
        self.weight = np.random.randn(output_size, input_size).astype(np.float32)
        self.weight = self.weight / np.linalg.norm(self.weight, axis=1, keepdims=True)

        if isinstance(prior_mean, np.ndarray):
            self.bias = prior_mean
        else:
            self.bias = np.ones(output_size, dtype=np.float32) * prior_mean

        if isinstance(prior_std, np.ndarray):
            if prior_std.ndim == 1:
                assert len(prior_std) == output_size
                self.prior_std = np.diag(prior_std).astype(np.float32)
            elif prior_std.ndim == 2:
                assert prior_std.shape == (output_size, output_size)
                self.prior_std = prior_std
            else:
                raise ValueError
        else:
            assert isinstance(prior_std, (float, int, np.float32, np.int32, np.float64, np.int64))
            self.prior_std = np.eye(output_size, dtype=np.float32) * prior_std

        self.weight = nn.Parameter(to_torch(self.prior_std @ self.weight, device=device))
        self.bias = nn.Parameter(to_torch(self.bias, device=device))

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        logits = F.linear(s, self.weight, self.bias)
        return logits


class HyperDQNWithoutPrior(nn.Module):
    """HyperDQN without prior model."""
    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        z_size: int,
        bias_coef: float = 0.01,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.z_size = z_size
        self.bias_coef = bias_coef

        self.conv_net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten())
        with torch.no_grad():
            cnn_output_dim = int(np.prod(
                self.conv_net(torch.zeros(1, c, h, w)).shape[1:]))
        self.conv_net = nn.Sequential(
            self.conv_net,
            nn.Linear(cnn_output_dim, 512), nn.ReLU(inplace=True)
        )
        self.num_action = int(np.prod(action_shape))
        self.output_dim = int(np.prod(action_shape)) * (512 + 1)

        self.hypermodel = nn.Linear(z_size, self.output_dim)

        print('hyper model output size: %d' % self.output_dim)

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        z: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Q(x, \*)."""
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        z = torch.as_tensor(z, device=self.device, dtype=torch.float32)

        feature = self.conv_net(x)
        params = self.hypermodel(z)
        qs = []
        for k in range(z.shape[0]):
            weight, bias = torch.split(params[k], [self.num_action * 512, self.num_action])
            weight = weight.reshape([self.num_action, 512])
            bias = bias * self.bias_coef

            q_value = F.linear(feature, weight, bias)
            qs.append(q_value)
        q_value = torch.stack(qs, dim=1)  # (None, num_ensemble, num_action)
        if z.shape[0] == 1:
            q_value = q_value[:, 0, :]
        return q_value, state

    def get_entropy(self):
        weight, bias = self.hypermodel.weight, self.hypermodel.bias
        entropy = torch.log(torch.trace(weight @ weight.T)).item()
        return entropy


class HyperDQNWithPrior(nn.Module):
    """HyperDQN with prior model."""
    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        z_size: int,
        prior_mean: float or np.ndarray = 0.0,
        prior_std: float or np.ndarray = 1.0,
        bias_coef: float = 0.01,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.z_size = z_size
        self.bias_coef = bias_coef

        self.conv_net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten())
        with torch.no_grad():
            cnn_output_dim = int(np.prod(
                self.conv_net(torch.zeros(1, c, h, w)).shape[1:]))
        self.conv_net = nn.Sequential(
            self.conv_net,
            nn.Linear(cnn_output_dim, 512), nn.ReLU(inplace=True)
        )
        self.num_action = int(np.prod(action_shape))
        self.output_dim = int(np.prod(action_shape)) * (512 + 1)

        self.hypermodel = LinearPriorNet(z_size, self.output_dim, prior_mean=prior_mean, prior_std=prior_std,
                                         device=device)

        for param in self.parameters():
            param.requires_grad = False

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        z: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Q(x, \*)."""
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        z = torch.as_tensor(z, device=self.device, dtype=torch.float32)

        feature = self.conv_net(x)
        params = self.hypermodel(z)
        qs = []
        for k in range(z.shape[0]):
            weight, bias = torch.split(params[k], [self.num_action * 512, self.num_action])
            weight = weight.reshape([self.num_action, 512])
            bias = bias * self.bias_coef

            q_value = F.linear(feature, weight, bias)
            qs.append(q_value)
        q_value = torch.stack(qs, dim=1)  # (None, num_ensemble, num_action)
        if z.shape[0] == 1:
            q_value = q_value[:, 0, :]
        return q_value, state


class HyperDQN(nn.Module):
    """HyperDQN for Video Game."""
    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        z_size: int,
        prior_scale: float = 0.0,
        posterior_scale: float = 1.0,
        prior_mean: float or np.ndarray = 0.0,
        prior_std: float or np.ndarray = 1.0,
        bias_coef: float = 0.01,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.z_size = z_size
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale

        self.model = HyperDQNWithoutPrior(c, h, w, action_shape, z_size,
                                          bias_coef=bias_coef,
                                          device=device)
        self.prior_model = HyperDQNWithPrior(c, h, w, action_shape, z_size,
                                             bias_coef=bias_coef,
                                             prior_mean=prior_mean, prior_std=prior_std,
                                             device=device)
        for param in self.prior_model.parameters():
            param.requires_grad = False

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        z: torch.Tensor,
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
        model: str = "all",
    ) -> Tuple[torch.Tensor, Any]:
        s = torch.as_tensor(s, device=self.device, dtype=torch.float32)
        z = torch.as_tensor(z, device=self.device, dtype=torch.float32)

        if model == "all":
            q, _ = self.model(s, z, state, info)
            q_prior, _ = self.prior_model(s, z, state, info)
            q_value = q * self.posterior_scale + q_prior * self.prior_scale
        elif model == "prior":
            q_prior, _ = self.prior_model(s, z, state, info)
            q_value = q_prior * self.prior_scale
        elif model == "posterior":
            q, _ = self.model(s, z, state, info)
            q_value = q * self.posterior_scale
        else:
            raise ValueError(model)
        return q_value, state

    def generate_z(self, batch_size: int) -> torch.Tensor:
        z = torch.randn(batch_size, self.z_size).type(torch.float32).to(self.device)
        return z

    def get_hyper_trainable_output(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.hypermodel(z)

    def get_entropy(self):
        entropy = self.model.get_entropy()
        return entropy
