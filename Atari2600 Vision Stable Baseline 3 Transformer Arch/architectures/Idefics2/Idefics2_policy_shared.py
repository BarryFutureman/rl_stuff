from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
import numpy as np
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .modeling_idefics2_for_rl import *
from transformers import MistralConfig
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from ..policies import ActorCriticPolicy


class VisionActorCriticNetwork(nn.Module):
    def __init__(
            self,
            config: Idefics2Config
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = 1024
        self.latent_dim_vf = 1024

        # Policy network

        self.policy_net = Idefics2Dummy(config, self.latent_dim_pi)

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
        return self.policy_net(pixel_values=features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.forward_actor(features)


class FeatureExtractor(BaseFeaturesExtractor):
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

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations


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
            features_extractor_class=FeatureExtractor,

            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        self.features_extractor_kwargs["features_dim"] = 128
        return self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)

    def _build_mlp_extractor(self) -> None:
        config = Idefics2Config(
            use_cache=False,
            image_token_id=32_001,
            tie_word_embeddings=False,
            vision_config=Idefics2VisionConfig(
                hidden_size=self.features_dim,
                intermediate_size=self.features_dim*2,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_channels=3,
                image_size=84,
                patch_size=8,
                hidden_act="gelu_pytorch_tanh",
                layer_norm_eps=1e-6,
                attention_dropout=0.0,
                initializer_range=0.02,
            ),
            perceiver_config=None,
            text_config=MistralConfig(
                vocab_size=32000,
                hidden_size=self.features_dim,
                intermediate_size=self.features_dim*2,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=1,
                hidden_act="silu",
                max_position_embeddings=self.memory_length * 64,
                initializer_range=0.02,
                rms_norm_eps=1e-6,
                use_cache=False,
                pad_token_id=None,
                bos_token_id=1,
                eos_token_id=2,
                tie_word_embeddings=False,
                rope_theta=10000.0,
                sliding_window=4096,
                attention_dropout=0.0,
            )
        )

        self.mlp_extractor = VisionActorCriticNetwork(config)
        print(self.mlp_extractor)
