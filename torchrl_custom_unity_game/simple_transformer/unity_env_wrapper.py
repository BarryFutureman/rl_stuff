from torchrl.envs import GymWrapper
from torchrl.envs.gym_like import *
import numpy as np
import torch
from typing import *

from model import TransformerModule


class OpponentAgent:
    def __init__(self, model_paths: List[str], device: str = "cpu"):
        self.device = device
        self.models = []
        self.obs_size = None
        self.max_position_embeddings = None
        for path in model_paths:
            model = TransformerModule(
                in_features=0,
                out_features=0,
                device=device
            )
            model.load_model_weights(path)
            model.model.eval()
            if self.obs_size is None:
                self.obs_size = model.model.config.obs_size
            if self.max_position_embeddings is None:
                self.max_position_embeddings = model.model.config.max_position_embeddings
            self.models.append(model)
        self.active_model_idx = 0
        self.memory = torch.zeros((1, self.max_position_embeddings, self.obs_size), device=self.device)

    def reset(self):
        self.memory.zero_()
        if len(self.models) > 1:
            self.active_model_idx = np.random.randint(0, len(self.models))

    @torch.no_grad()
    def predict(self, observation: torch.Tensor) -> int:
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        self.memory = torch.cat((self.memory[:, 1:], observation.unsqueeze(1).to(self.device)), dim=1)
        model = self.models[self.active_model_idx]
        logits = model(self.memory)
        action = torch.argmax(logits.squeeze(1), dim=-1)
        return int(action.item())


class UnityGymSelfPlayWrapper(GymWrapper):
    def __init__(self, env: Any = None, categorical_action_encoding=False, **kwargs):
        if env is None:
            raise NotImplementedError()
        super().__init__(env=env, categorical_action_encoding=categorical_action_encoding, **kwargs)

        self.player2_obs = None
        self.player2_agent = None

    def load_opponent(self, model_paths: List[str], device: str = "cpu"):
        self.player2_agent = OpponentAgent(model_paths, device)
        
    def read_obs(
            self, observations: Union[Dict[str, Any], torch.Tensor, np.ndarray]
    ) -> Dict[str, Any]:
        """Reads an observation from the environment and returns an observation compatible with the output TensorDict.

        Args:
            observations (observation under a format dictated by the inner env): observation to be read.

        """
        if isinstance(observations, dict):
            if "state" in observations and "observation" not in observations:
                # we rename "state" in "observation" as "observation" is the conventional name
                # for single observation in torchrl.
                # naming it 'state' will result in envs that have a different name for the state vector
                # when queried with and without pixels
                observations["observation"] = observations.pop("state")
        if not isinstance(observations, Mapping):
            for key, spec in self.observation_spec.items(True, True):
                observations_dict = {}
                player1_observations = observations[self._player1_obs_slice]
                player2_observations = observations[self._player1_obs_slice.stop:]
                observations_dict[key] = spec.encode(player1_observations, ignore_device=True)
                self.player2_obs = spec.encode(player2_observations, ignore_device=True)
                # we don't check that there is only one spec because obs spec also
                # contains the data spec of the info dict.
                break
            else:
                raise RuntimeError("Could not find any element in observation_spec.")
            observations = observations_dict
        else:
            raise NotImplementedError()
            # for key, val in observations.items():
            #     if isinstance(self.observation_spec[key], NonTensor):
            #         observations[key] = NonTensorData(val)
            #     else:
            #         observations[key] = self.observation_spec[key].encode(
            #             val, ignore_device=True
            #         )

        return observations

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get(self.action_key)
        if self._convert_actions_to_numpy:
            action = self.read_action(action)

        reward = 0
        for _ in range(self.wrapper_frame_skip):
            (
                obs,
                _reward,
                terminated,
                truncated,
                done,
                info_dict,
            ) = self._output_transform(self._env.step(action))

            if _reward is not None:
                reward = reward + _reward

            terminated, truncated, done, do_break = self.read_done(
                terminated=terminated, truncated=truncated, done=done
            )
            if do_break:
                break

        # Reset opponent memory if episode is done
        if done:
            self.reset_opponent()

        reward = self.read_reward(reward)
        obs_dict = self.read_obs(obs)
        obs_dict[self.reward_key] = reward

        # if truncated/terminated is not in the keys, we just don't pass it even if it
        # is defined.
        if terminated is None:
            terminated = done
        if truncated is not None:
            obs_dict["truncated"] = truncated
        obs_dict["done"] = done
        obs_dict["terminated"] = terminated
        validated = self.validated
        if not validated:
            tensordict_out = TensorDict(obs_dict, batch_size=tensordict.batch_size)
            if validated is None:
                # check if any value has to be recast to something else. If not, we can safely
                # build the tensordict without running checks
                self.validated = all(
                    val is tensordict_out.get(key)
                    for key, val in TensorDict(obs_dict, []).items(True, True)
                )
        else:
            tensordict_out = TensorDict._new_unsafe(
                obs_dict,
                batch_size=tensordict.batch_size,
            )
        if self.device is not None:
            tensordict_out = tensordict_out.to(self.device)

        if self.info_dict_reader and (info_dict is not None):
            if not isinstance(info_dict, dict):
                warnings.warn(
                    f"Expected info to be a dictionary but got a {type(info_dict)} with values {str(info_dict)[:100]}."
                )
            else:
                for info_dict_reader in self.info_dict_reader:
                    out = info_dict_reader(info_dict, tensordict_out)
                    if out is not None:
                        tensordict_out = out
        return tensordict_out

    def reset_opponent(self):
        if self.player2_agent is not None:
            self.player2_agent.reset()

    def read_action(self, action):
        # Generate index for player2
        if self.player2_agent is not None and self.player2_obs is not None:
            # Convert player2_obs to torch.Tensor if needed
            obs = self.player2_obs
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float32)
            player2_index = self.player2_agent.predict(obs)
        else:
            player2_index = -1

        # Convert player1's action and player2's index to one-hot encoding
        player1_one_hot = np.zeros(self._player1_action_slice.stop, dtype=np.float32)
        player1_one_hot[action] = 1.0

        player2_one_hot = np.zeros(self._player1_action_slice.stop, dtype=np.float32)
        player2_one_hot[player2_index] = 1.0

        # Concatenate player1 and player2 one-hot encodings
        full_action = np.concatenate([player1_one_hot, player2_one_hot])

        return full_action

    def _make_specs(self, env, batch_size=None):
        # Call parent to get full specs
        super()._make_specs(env, batch_size)

        # Save full specs for internal use
        self._full_action_spec = self.action_spec
        self._full_observation_spec = self.observation_spec

        # Define slices here, after specs are available
        obs_dim = self._full_observation_spec["observation"].shape[0]
        act_dim = self._full_action_spec.shape[0]
        self._player1_obs_slice = slice(0, obs_dim // 2)
        self._player1_action_slice = slice(0, act_dim // 2)

        from torchrl.data.tensor_specs import MultiCategorical, Categorical
        act_dim = self._full_action_spec.shape[0]
        discrete_dim = act_dim // 2
        self.action_spec = Categorical(discrete_dim, device=self.device)
        self.observation_spec = self._full_observation_spec[self._player1_obs_slice]
