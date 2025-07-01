from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space

from stable_baselines3.common.policies import ActorCriticPolicy


class FeedForward(nn.Module):
    def __init__(self, hidden_size):
        super(FeedForward, self).__init__()

        self.mlp = nn.Linear(hidden_size, hidden_size)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.mlp(x))


class SimpleMLP(nn.Module):
    def __init__(self,
                 input_t: int,
                 vector_size: int,
                 num_layers: int,
                 hidden_size: int,
                 out_size: int):
        super().__init__()

        self.t_size = input_t
        self.in_proj = nn.Linear(vector_size*input_t, hidden_size)
        self.feed_forward_layers = nn.ModuleList(
            [FeedForward(hidden_size) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(hidden_size, out_size)

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

    def forward(self, idx):
        # b, t, v = idx.size()  # v is the vector obs

        x = idx.view(idx.size(0), -1)

        x = self.in_proj(x)
        for ff in self.feed_forward_layers:
            x = ff(x)

        return self.out_proj(x)


class MLPActorCriticNetwork(nn.Module):
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
        self.policy_net = SimpleMLP(
            input_t=t_dim,
            vector_size=feature_dim,
            num_layers=2,
            hidden_size=1024,
            out_size=self.latent_dim_pi,
        )

        # Shared Value network
        self.value_net = self.policy_net

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

        n_input_channels = observation_space.shape[1]

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
            # one [0] for stack, we just want to try a single cnn pass here
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[0][None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
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
        self.features_extractor_kwargs["features_dim"] = 128
        return self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MLPActorCriticNetwork(feature_dim=self.features_dim,
                                                   device=self.device,
                                                   t_dim=self.memory_length,
                                                   last_layer_dim_pi=512,
                                                   last_layer_dim_vf=512)
