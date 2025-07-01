from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
import numpy as np
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .modeling_llama_for_rl import LlamaModelForRL, LlamaConfig
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from ..policies import ActorCriticPolicy


class LLamaActorCriticNetwork(nn.Module):
    def __init__(
            self,
            config: LlamaConfig
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = 1024
        self.latent_dim_vf = 1024

        # Policy network

        self.policy_net = LlamaModelForRL(config, self.latent_dim_pi)

        print("Number of parameters: %.2fM" % (sum(p.numel() for p in self.parameters()) / 1e6,),
              self.policy_net.num_parameters())

        # Shared Value network
        self.value_net = self.policy_net

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        output = self.forward_actor(features)

        return output, output

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(inputs_embeds=features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.forward_actor(features)


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            features_dim: int = 64,
            normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            transposed_sample = observation_space.sample()
            # transposed_sample = np.transpose(transposed_sample, (2, 0, 1))
            # Not transposing here because somehow it is done automatically

            n_flatten = self.cnn(th.as_tensor(transposed_sample[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations = observations.permute(0, 3, 1, 2)

        if len(observations.shape) == 5:  # batch, memory, c, w, h
            features = []
            for i in range(observations.size(1)):
                # Select a chunk along dim 1
                curr_stack = observations[:, i, :, :, :]
                features.append(self.linear(self.cnn(curr_stack)))
            features = th.stack(features, dim=1)
        else:
            features = self.linear(self.cnn(observations))
        return features


class PixelBasedActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            *args,
            **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=NatureCNN,

            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        self.features_extractor_kwargs["features_dim"] = 128
        return self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)

    def _build_mlp_extractor(self) -> None:
        config = LlamaConfig(
            vocab_size=-1,
            hidden_size=self.features_dim,
            intermediate_size=self.features_dim * 2,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=self.memory_length,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            attention_dropout=0.0,
        )

        self.mlp_extractor = LLamaActorCriticNetwork(config)
        print(self.mlp_extractor)
