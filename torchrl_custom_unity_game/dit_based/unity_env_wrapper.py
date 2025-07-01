from torchrl.envs import GymWrapper
from torchrl.envs.gym_like import *
import numpy as np
import torch
from typing import *


class UnityGymSelfPlayWrapper(GymWrapper):
    def __init__(self, env: Any = None, categorical_action_encoding=False, action_chunk_size: int = 4, **kwargs):
        self.action_chunk_size = action_chunk_size
        if env is None:
            raise NotImplementedError()
        super().__init__(env=env, categorical_action_encoding=categorical_action_encoding, **kwargs)

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
                observations_dict[key] = spec.encode(observations, ignore_device=True)
                # we don't check that there is only one spec because obs spec also
                # contains the data spec of the info dict.
                break
            else:
                raise RuntimeError("Could not find any element in observation_spec.")
            observations = observations_dict
        else:
            for key, val in observations.items():
                if isinstance(self.observation_spec[key], NonTensor):
                    observations[key] = NonTensorData(val)
                else:
                    observations[key] = self.observation_spec[key].encode(
                        val, ignore_device=True
                    )

        if isinstance(observations["observation"], torch.Tensor) or isinstance(observations["observation"], np.ndarray):
            observations["observation"] = observations["observation"][..., self._player1_obs_slice]
        else:
            raise NotImplementedError("What?????")
        return observations

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get(self.action_key)
        if self._convert_actions_to_numpy:
            action = self.read_action(action)

        obs_list = []
        reward_sum = 0.0
        done = False
        info_dict = None
        terminated = None
        truncated = None

        # Loop through action chunks
        for i in range(self.action_chunk_size):
            # Step the environment with the i-th chunk of the action
            step_action = action[i]
            (
                obs,
                _reward,
                terminated,
                truncated,
                done,
                info_dict,
            ) = self._output_transform(self._env.step(step_action))

            obs_list.append(obs)
            if _reward is not None:
                reward_sum += _reward

            terminated, truncated, done, do_break = self.read_done(
                terminated=terminated, truncated=truncated, done=done
            )
            if do_break:
                break

        if len(obs_list) < self.action_chunk_size:
            pad_count = self.action_chunk_size - len(obs_list)
            if obs_list:
                # Use the shape and dtype of the last observation
                zero_obs = np.zeros_like(obs_list[-1])
            else:
                # Fallback to environment's observation_space
                # shape = self._env.observation_space.shape
                # dtype = getattr(self._env.observation_space, 'dtype', np.float32)
                # zero_obs = np.zeros(shape, dtype=dtype)
                raise NotImplementedError()
            obs_list.extend([zero_obs] * pad_count)

        # Stack observations along the chunk dimension
        if isinstance(obs_list[0], dict):
            stacked_obs = {k: torch.stack([torch.as_tensor(o[k]) for o in obs_list], dim=0) for k in obs_list[0]}
        else:
            stacked_obs = torch.stack([torch.as_tensor(o) for o in obs_list], dim=0)

        reward = self.read_reward(reward_sum)
        obs_dict = self.read_obs(stacked_obs)
        obs_dict[self.reward_key] = reward

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

    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> TensorDictBase:
        obs, info = self._reset_output_transform(self._env.reset(**kwargs))

        stacked_obs = np.stack([obs for i in range(self.action_chunk_size)], axis=0)
        source = self.read_obs(stacked_obs)

        tensordict_out = TensorDict._new_unsafe(
            source=source,
            batch_size=self.batch_size,
        )
        if self.info_dict_reader and info is not None:
            for info_dict_reader in self.info_dict_reader:
                out = info_dict_reader(info, tensordict_out)
                if out is not None:
                    tensordict_out = out
        elif info is None and self.info_dict_reader:
            # populate the reset with the items we have not seen from info
            for key, item in self.observation_spec.items(True, True):
                if key not in tensordict_out.keys(True, True):
                    tensordict_out[key] = item.zero()
        if self.device is not None:
            tensordict_out = tensordict_out.to(self.device)
        return tensordict_out

    def read_action(self, action):
        T = self.action_chunk_size
        D = action.shape[-1] // T
        action = action.reshape(T, D)

        # player 2
        zeros = np.zeros_like(action)
        # Concatenate player1's action and zeros (player2's action) on the last dimension
        full_action = np.concatenate([action, zeros], axis=-1)
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
        self._player1_obs_slice = slice(0, obs_dim)  # Full slice for now
        self._player1_action_slice = slice(0, act_dim // 2)

        # Override exposed specs
        self.action_spec = self._full_action_spec[self._player1_action_slice]
        self.observation_spec = self._full_observation_spec[self._player1_obs_slice]

        # Update the observation spec shape for stacking
        # Only update the "observation" key, not the whole Composite
        obs_spec = self.observation_spec
        if "observation" in obs_spec:
            orig_shape = obs_spec["observation"].shape
            obs_spec["observation"].shape = (self.action_chunk_size,) + orig_shape
        self.observation_spec = obs_spec

        # self.action_spec.shape = (self.action_chunk_size,) + self.action_spec.shape
        # low = self.action_spec.space.low
        # high = self.action_spec.space.high
        # low_stacked = low.repeat(self.action_chunk_size, 1)
        # high_stacked = high.repeat(self.action_chunk_size, 1)
        # self.action_spec.space.low = low_stacked
        # self.action_spec.space.high = high_stacked

        # original low/high are 1-D numpy arrays of length D
        low = self.action_spec.space.low  # might already be a torch.Tensor
        high = self.action_spec.space.high

        # if they’re numpy arrays, convert them
        if isinstance(low, np.ndarray):
            low = torch.as_tensor(low, dtype=torch.float32)
            high = torch.as_tensor(high, dtype=torch.float32)

        T = self.action_chunk_size

        # now tile/repeat them T times
        # – if they’re torch.Tensors, use torch.repeat or repeat_interleave
        low_flat = low.repeat(T)  # shape (T*D,)
        high_flat = high.repeat(T)

        # update spec
        self.action_spec.shape = (T * low.shape[0],)
        self.action_spec.space.low = low_flat
        self.action_spec.space.high = high_flat
