"""Asymmetric actor-critic for Stable-Baselines3 PPO (docs/project-draft.md section 25).

The critic may see privileged simulator state; the actor must not. The
environment wrapper :class:`PrivilegedObservation` copies
``info["privileged"]`` into the observation under :data:`PRIVILEGED_KEY`.
:class:`KeyedFlattenExtractor` flattens the actor keys first and the critic-only
keys after them, and :class:`AsymmetricMlpExtractor` feeds the policy network
only the leading actor slice of those features. Changing the privileged
values therefore cannot change an action, which a test enforces.

At deployment the privileged entry can be filled with zeros (see
:mod:`manip_rubiks.policies.learned`).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

PRIVILEGED_KEY = "privileged"


class PrivilegedObservation(gym.Wrapper):
    """Expose ``info["privileged"]`` as an observation entry, for the critic only."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        dim = int(env.unwrapped.privileged_state().size)
        self.observation_space = spaces.Dict(
            {**env.observation_space.spaces, PRIVILEGED_KEY: spaces.Box(-np.inf, np.inf, (dim,), np.float32)}
        )

    @staticmethod
    def _augment(obs: dict[str, np.ndarray], info: dict[str, Any]) -> dict[str, np.ndarray]:
        return {**obs, PRIVILEGED_KEY: np.asarray(info[PRIVILEGED_KEY], dtype=np.float32)}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs, info), reward, terminated, truncated, info


class KeyedFlattenExtractor(BaseFeaturesExtractor):
    """Concatenate the flattened Dict entries in an explicit key order."""

    def __init__(self, observation_space: spaces.Dict, keys: tuple[str, ...]) -> None:
        self.keys = tuple(keys)
        self.dims = {k: int(np.prod(observation_space[k].shape)) for k in self.keys}
        super().__init__(observation_space, features_dim=sum(self.dims.values()))

    def dim_of(self, keys: tuple[str, ...]) -> int:
        return sum(self.dims[k] for k in keys)

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        return th.cat([observations[k].flatten(start_dim=1) for k in self.keys], dim=1)


def _mlp(in_dim: int, sizes: list[int], activation: type[nn.Module]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for size in sizes:
        layers += [nn.Linear(in_dim, size), activation()]
        in_dim = size
    return nn.Sequential(*layers)


class AsymmetricMlpExtractor(nn.Module):
    """Policy MLP on the actor slice of the features, value MLP on all of them."""

    def __init__(
        self,
        feature_dim: int,
        actor_dim: int,
        net_arch: dict[str, list[int]] | list[int],
        activation_fn: type[nn.Module],
        device: th.device | str = "auto",
    ) -> None:
        super().__init__()
        pi = net_arch["pi"] if isinstance(net_arch, dict) else list(net_arch)
        vf = net_arch["vf"] if isinstance(net_arch, dict) else list(net_arch)
        self.actor_dim = actor_dim
        self.policy_net = _mlp(actor_dim, pi, activation_fn)
        self.value_net = _mlp(feature_dim, vf, activation_fn)
        self.latent_dim_pi = pi[-1] if pi else actor_dim
        self.latent_dim_vf = vf[-1] if vf else feature_dim

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features[..., : self.actor_dim])

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class AsymmetricActorCriticPolicy(MultiInputActorCriticPolicy):
    """PPO policy whose actor reads ``actor_keys`` and whose critic also reads ``critic_keys``."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule,
        actor_keys: tuple[str, ...] = (),
        critic_keys: tuple[str, ...] = (PRIVILEGED_KEY,),
        **kwargs: Any,
    ) -> None:
        self.actor_keys = tuple(actor_keys) or tuple(k for k in observation_space.spaces if k not in critic_keys)
        self.critic_keys = tuple(k for k in critic_keys if k in observation_space.spaces)
        overlap = set(self.actor_keys) & {PRIVILEGED_KEY}
        if overlap:
            raise ValueError(f"actor must not observe {overlap}")
        kwargs["features_extractor_class"] = KeyedFlattenExtractor
        kwargs["features_extractor_kwargs"] = {"keys": self.actor_keys + self.critic_keys}
        kwargs["share_features_extractor"] = True
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AsymmetricMlpExtractor(
            self.features_dim,
            self.features_extractor.dim_of(self.actor_keys),
            self.net_arch,
            self.activation_fn,
            self.device,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(actor_keys=self.actor_keys, critic_keys=self.critic_keys)
        return data
