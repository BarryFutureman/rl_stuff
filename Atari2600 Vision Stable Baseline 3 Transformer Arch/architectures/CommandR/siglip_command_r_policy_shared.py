from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
import numpy as np
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .modeling_command_r_for_rl import CohereModelForRL, CohereConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionModel, SiglipVisionConfig
from transformers.models.idefics2.modeling_idefics2 import Idefics2Connector
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from ..policies import ActorCriticPolicy


class JointActorCriticNetwork(nn.Module):
    def __init__(
            self,
            config: CohereConfig
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = 1024
        self.latent_dim_vf = 1024

        # Policy network

        self.policy_net = CohereModelForRL(config, self.latent_dim_pi)

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
        features = features.view(features.shape[0], -1, features.shape[-1])
        return self.policy_net(inputs_embeds=features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.forward_actor(features)


class SigLipVisionEncoder(BaseFeaturesExtractor):

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
        # self.model = SiglipVisionModel(
        #     SiglipVisionConfig(
        #         hidden_size=features_dim,
        #         intermediate_size=features_dim*2,
        #         num_hidden_layers=4,
        #         num_attention_heads=4,
        #         num_channels=3,
        #         image_size=84,
        #         patch_size=16,
        #         hidden_act="gelu_pytorch_tanh",
        #         layer_norm_eps=1e-6,
        #         attention_dropout=0.0)
        # )
        # from transformers.utils import logging
        #
        # logging.set_verbosity_info()
        # logger = logging.get_logger("transformers")
        # logger.info("INFO")
        # logger.warning("WARN")
        # logging.enable_progress_bar()

        self.model = SiglipVisionModel.from_pretrained("HuggingFaceM4/siglip-so400m-14-384-flash-attn2-navit",
                                                       cache_dir="cache/models")
        self.projector = nn.Sequential(nn.ReLU(), nn.Linear(self.model.config.hidden_size, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        print(observations.dtype)
        quit()

        if len(observations.shape) == 5:  # batch, memory, c, w, h
            # Reshape observations to merge batch and memory dimensions
            batch, memory, c, w, h = observations.shape
            observations = observations.view(batch*memory, c, w, h)
            # Pass all observations through the model in one go
            features = self.model(pixel_values=observations).last_hidden_state

            features = self.projector(features)

            # Reshape features back to original shape
            features = features.view(batch, memory, 25, self.features_dim)
        else:
            features = self.model(pixel_values=observations).last_hidden_state
            features = self.projector(features)

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
            features_extractor_class=SigLipVisionEncoder,

            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        self.features_extractor_kwargs["features_dim"] = 128
        return self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)

    def _build_mlp_extractor(self) -> None:
        config = CohereConfig(
            vocab_size=-1,
            hidden_size=self.features_dim,
            intermediate_size=self.features_dim * 2,
            logit_scale=0.0625,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=self.memory_length * 25,  # TODO: Hard coded 25 here, look into siglip, and then change this
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=5,
            eos_token_id=255001,
            tie_word_embeddings=True,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
            use_qk_norm=False,
        )

        self.mlp_extractor = JointActorCriticNetwork(config)
        print(self.mlp_extractor)
