from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
import numpy as np
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .modeling_jamba_for_rl import JambaModelForRL, JambaConfig

from stable_baselines3.common.policies import ActorCriticPolicy


class JambaActorCriticNetwork(nn.Module):
    def __init__(
            self,
            config: JambaConfig
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = 1024
        self.latent_dim_vf = 1024

        # Policy network

        self.policy_net = JambaModelForRL(config, self.latent_dim_pi)

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

        n_input_channels = observation_space.shape[-1]

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
            transposed_sample = np.transpose(transposed_sample, (0, 3, 1, 2))
            # one [0] for stack, we just want to try a single cnn pass here
            n_flatten = self.cnn(th.as_tensor(transposed_sample[0][None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations.permute(0, 1, 4, 2, 3)

        x = []
        for i in range(observations.size(1)):
            # Select a chunk along dim 1
            curr_stack = observations[:, i, :, :, :]
            x.append(self.linear(self.cnn(curr_stack)))

        return th.stack(x, dim=1)


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

        self.memory_length = observation_space.shape[0]

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=NatureCNN,

            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            raise NotImplementedError()
            # pi_features, vf_features = features
            # latent_pi = self.mlp_extractor.forward_actor(pi_features)
            # latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        self.features_extractor_kwargs["features_dim"] = 32
        return self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)

    def _build_mlp_extractor(self) -> None:
        # config = JambaConfig(
        #     vocab_size=-1,
        #     tie_word_embeddings=False,
        #     hidden_size=self.features_dim,
        #     intermediate_size=512,
        #     num_hidden_layers=16,
        #     num_attention_heads=16,
        #     num_key_value_heads=4,
        #     hidden_act="silu",
        #     initializer_range=0.02,
        #     rms_norm_eps=1e-6,
        #     use_cache=True,
        #     calc_logits_for_entire_prompt=False,
        #     output_router_logits=False,
        #     router_aux_loss_coef=0.001,
        #     pad_token_id=0,
        #     bos_token_id=1,
        #     eos_token_id=2,
        #     sliding_window=None,
        #     n_ctx=self.memory_length,
        #     attention_dropout=0.0,
        #     num_experts_per_tok=2,
        #     num_experts=8,
        #     expert_layer_period=2,
        #     expert_layer_offset=1,
        #     attn_layer_period=8,
        #     attn_layer_offset=4,
        #     use_mamba_kernels=False,  # Windows not supported
        #     mamba_d_state=16,
        #     mamba_d_conv=4,
        #     mamba_expand=2,
        #     mamba_dt_rank="auto",
        #     mamba_conv_bias=True,
        #     mamba_proj_bias=False,
        #     mamba_inner_layernorms=True,
        # )

        config = JambaConfig(
            vocab_size=-1,
            tie_word_embeddings=False,
            hidden_size=self.features_dim,
            intermediate_size=int(self.features_dim)*2,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_act="silu",
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            calc_logits_for_entire_prompt=False,
            output_router_logits=False,
            router_aux_loss_coef=0.001,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            sliding_window=None,
            n_ctx=self.memory_length,
            attention_dropout=0.0,
            num_experts_per_tok=1,
            num_experts=2,
            expert_layer_period=2,
            expert_layer_offset=1,
            attn_layer_period=2,
            attn_layer_offset=4,
            use_mamba_kernels=False,  # Windows not supported
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_dt_rank="auto",
            mamba_conv_bias=True,
            mamba_proj_bias=False,
            mamba_inner_layernorms=True,
        )
        self.mlp_extractor = JambaActorCriticNetwork(config)
        print(self.mlp_extractor)
