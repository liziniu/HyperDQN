from copy import deepcopy
import torch
import math
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from typing import Union, Any, Dict, Optional, Tuple

from tianshou.data import to_torch


class HyperLinearWeight(nn.Module):
    """
    A linear hypermodel to generate the ``weight`` for single layer of the base model.
    """
    def __init__(
        self,
        z_size: int,
        output_shape: Tuple[int],
        target_init: str = "tf",
        has_bias: bool = True,
        trainable: bool = True,
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()
        self.z_size = z_size
        self.output_shape = output_shape
        self.target_init = target_init
        self.has_bias = has_bias
        self.trainable = trainable
        self.device = device

        self.num_param = int(np.prod(self.output_shape))

        self.weight = nn.Parameter(torch.empty([self.num_param, self.z_size], dtype=torch.float32, device=self.device))
        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros([self.num_param], dtype=torch.float32, device=self.device))
        else:
            self.bias = torch.zeros([self.num_param], dtype=torch.float32, device=self.device)

        self.reset_parameters()

        if not self.trainable:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        z: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        param = F.linear(z, self.weight, self.bias)
        param = param.reshape([len(z), *self.output_shape])
        return param

    def reset_parameters(self):
        """
        Initialize the ``weight`` term.
        TensorFlow: 1/sqrt(f_in) initialization.
        PyTorch: He Kaiming initialization.
        """
        if self.target_init == "tf":
            f_in = self.output_shape[1]
            bound = 1.0 / math.sqrt(f_in)
            nn.init.trunc_normal_(self.weight, std=bound, a=-2 * bound, b=2 * bound)
        elif self.target_init == "torch":
            f_in = self.output_shape[1]
            gain = nn.init.calculate_gain('leaky_relu', math.sqrt(5))
            std = gain / math.sqrt(f_in)
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            with torch.no_grad():
                self.weight.uniform_(-bound, bound)
        else:
            raise ValueError(self.target_init)


class HyperLinearBias(nn.Module):
    """
    A linear hypermodel to generate the ``bias`` for single layer of the base model.
    """
    def __init__(
        self,
        z_size: int,
        output_shape: Tuple[int],
        target_init: str = "tf",
        has_bias: bool = True,
        trainable: bool = True,
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()
        self.z_size = z_size
        self.output_shape = output_shape
        self.output_dim = output_shape[0]
        self.target_init = target_init
        self.has_bias = has_bias
        self.trainable = trainable
        self.device = device

        self.num_param = self.output_dim

        self.weight = nn.Parameter(torch.empty([self.num_param, self.z_size], dtype=torch.float32, device=self.device))
        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros([self.num_param], dtype=torch.float32, device=self.device))
        else:
            self.bias = torch.zeros([self.num_param], dtype=torch.float32, device=self.device)

        self.reset_parameters()

        if not self.trainable:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        z: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        param = F.linear(z, self.weight, self.bias)
        return param

    def reset_parameters(self):
        """
        Initialize the ``bias`` term.
        TensorFlow: zero initialization.
        PyTorch: Unif(1/sqrt(f_in)) initialization.
        """
        if self.target_init == "tf":
            nn.init.zeros_(self.weight)
        elif self.target_init == "torch":
            f_in = self.output_shape[1]
            bound = 1.0 / math.sqrt(f_in)
            nn.init.uniform(self.weight, -bound, bound)
        else:
            raise ValueError(self.target_init)


class HyperLinearLayer(nn.Module):
    """
    A linear hypermodel to mimic a layer of the base model.
    """
    def __init__(
        self,
        z_size: int,
        output_shape: Union[Tuple[int], Any],
        target_init: str = "tf",
        has_bias: bool = True,
        trainable: bool = True,
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()

        self.hyper_weight = HyperLinearWeight(z_size, output_shape, target_init, has_bias, trainable, device)
        self.hyper_bias = HyperLinearBias(z_size, output_shape, target_init, has_bias, trainable, device)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.hyper_weight(z)[0]
        bias = self.hyper_bias(z)[0]

        h = F.linear(x, weight, bias)
        return h


class LinearNet(nn.Linear):
    _init_type = "tf"

    def reset_parameters(self) -> None:
        if self._init_type == "tf":
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.trunc_normal_(self.weight, std=bound, a=-2*bound, b=2*bound)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        else:
            super().reset_parameters()


class HyperDQNBase(nn.Module):
    """
    HyperDQN without prior model.
    """
    def __init__(
        self,
        state_shape: tuple,
        action_shape: tuple,
        z_size: int = 32,
        bias_coef: float = 0.01,
        base_model_hidden_layer_sizes: tuple = (64, 64),
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.z_size = z_size
        self.bias_coef = bias_coef

        self.features = nn.Sequential(
            *[
                LinearNet(int(np.prod(state_shape)), base_model_hidden_layer_sizes[0]),
                nn.ReLU(inplace=True),
                LinearNet(base_model_hidden_layer_sizes[0], base_model_hidden_layer_sizes[1]),
                nn.ReLU(inplace=True)
            ]
        )

        self.num_action = int(np.prod(action_shape))
        self.hidden_dim = base_model_hidden_layer_sizes[-1]
        self.output_dim = self.num_action * (self.hidden_dim + 1)

        self.hypermodel = HyperLinearLayer(z_size, (self.num_action, base_model_hidden_layer_sizes[1]),
                                           target_init="tf", device=device)

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        z: torch.Tensor,
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        s = torch.as_tensor(s, device=self.device, dtype=torch.float32)
        z = torch.as_tensor(z, device=self.device, dtype=torch.float32)

        h = self.features(s)
        qs = []
        for k in range(z.shape[0]):
            q = self.hypermodel(h, z[k][None])
            qs.append(q)
        q = torch.stack(qs, dim=1)  # (None, num_ensemble, num_action)
        if z.shape[0] == 1:
            q = q[:, 0, :]
        return q, state


class HyperDQN(nn.Module):
    """
    HyperDQN
    """
    def __init__(
        self,
        state_shape: tuple,
        action_shape: tuple,
        z_size: int = 32,
        prior_scale: float = 0.0,
        posterior_scale: float = 1.0,
        prior_mean: float or np.ndarray = 0.0,
        prior_std: float or np.ndarray = 1.0,
        bias_coef: float = 0.01,
        base_model_hidden_layer_sizes: tuple = (64, 64),
        discrete_support: bool = False,
        device: Union[str, int, torch.device] = "cpu",
        **kwargs
    ) -> None:
        super().__init__()
        self.device = device
        self.z_size = z_size
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.discrete_support = discrete_support

        self.model = HyperDQNBase(
            state_shape, action_shape, z_size, bias_coef, base_model_hidden_layer_sizes,
            device=device
        )
        self.prior_model = HyperDQNBase(
            state_shape, action_shape, z_size, bias_coef, base_model_hidden_layer_sizes,
            device=device
        )

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
        if self.discrete_support:
            if batch_size == 1:
                z = torch.zeros(1, self.z_size).type(torch.float32).to(self.device)
                z[:, np.random.randint(self.z_size)] = 1.0
            else:
                assert batch_size == self.z_size
                z = torch.eye(self.z_size).type(torch.float32).to(self.device)
        else:
            z = torch.randn(batch_size, self.z_size).type(torch.float32).to(self.device)
            if self.z_size > 1:
                z = z / torch.norm(z, dim=1, keepdim=True)
        return z

    def get_hyper_trainable_output(self, z: torch.Tensor) -> torch.Tensor:
        hyper_weight = self.model.hypermodel.hyper_weight(z)
        hyper_bias = self.model.hypermodel.hyper_bias(z)
        params = torch.cat([hyper_weight.reshape(len(z), -1), hyper_bias.reshape(len(z), -1)], dim=1)
        return params

    def get_entropy(self):
        entropy = 0.0
        return entropy

    def compute_feature_rank(self, x: Union[torch.Tensor, np.ndarray], delta=0.01) -> int:
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        h = self.model.features(x)

        _, s, _ = torch.svd(h, compute_uv=False)
        z = torch.cumsum(s, dim=0)

        rank = torch.nonzero(z >= z[-1] * (1. - delta))[0][0] + 1
        return rank.item()
