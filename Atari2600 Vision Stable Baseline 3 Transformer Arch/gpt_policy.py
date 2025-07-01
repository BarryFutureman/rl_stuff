from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3.common.policies import ActorCriticPolicy

from my_transformer_implementation import *
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor


class GPT(nn.Module):
    def __init__(self,
                 device: str | th.device,
                 input_t: int,
                 vector_size: int,
                 num_layers: int,
                 num_heads: int,
                 hidden_size: int,
                 kernel_init: Initialization = Initialization.KaimingHeNormal,
                 kernel_gain: float = 1.0, ):
        super().__init__()

        model_args = dict(n_layer=num_layers, n_head=num_heads, n_embd=hidden_size, block_size=input_t,
                          bias=False, hidden_size=hidden_size, dropout=0, vector_size=vector_size)
        config = TransformerConfig(**model_args)
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=linear_layer(
                config.vector_size, config.n_embd,
                kernel_init=kernel_init,
                kernel_gain=kernel_gain,
            ),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Keep context
        self.context = None

        self.transformer_head = linear_layer(
            config.n_embd,
            config.hidden_size,
            kernel_init=kernel_init,
            kernel_gain=kernel_gain,
        )

        # init all weights
        self.apply(self._init_weights)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params(non_embedding=False) / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        b, t, v = idx.size()  # v is the vector obs
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        output = self.transformer_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
        # Squeeze because we only want the last dim
        output = output.squeeze(dim=1)

        return output


class GPTActorCriticNetwork(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            t_dim: int,
            device: str | th.device,
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = GPT(device=device,
                              input_t=t_dim,
                              vector_size=feature_dim,
                              num_layers=8,
                              num_heads=4,
                              hidden_size=self.latent_dim_pi,
                              )
        # Value network
        self.value_net = GPT(device=device,
                             input_t=t_dim,
                             vector_size=feature_dim,
                             num_layers=2,
                             num_heads=2,
                             hidden_size=self.latent_dim_vf,
                             )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """


        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=observation_space.shape[-1])

    def forward(self, observations: th.Tensor):
        return observations


class CustomActorCriticPolicy(ActorCriticPolicy):
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

        kwargs["optimizer_class"] = GaLoreAdamW
        raise NotImplementedError()
        # TODO: No Idea What I am Doing

        self.memory_length = observation_space.shape[-2]
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CustomFeatureExtractor,

            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        print(self.optimizer_class)

    """def extract_features(  # type: ignore[override]
            self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        # nothing to extract here
        return obs"""

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
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GPTActorCriticNetwork(feature_dim=self.features_dim,
                                                   device=self.device,
                                                   t_dim=self.memory_length,
                                                   last_layer_dim_pi=32,
                                                   last_layer_dim_vf=32,)
