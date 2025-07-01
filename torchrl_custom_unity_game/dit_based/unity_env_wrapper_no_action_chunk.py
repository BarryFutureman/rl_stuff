from torchrl.envs import GymWrapper
from torchrl.envs.gym_like import *
import numpy as np
import torch
from typing import *


class UnityGymSelfPlayWrapper(GymWrapper):
    def __init__(self, env: Any = None, categorical_action_encoding=False, **kwargs):
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
            observations["observation"] = observations["observation"][self._player1_obs_slice]
        else:
            raise NotImplementedError("What?????")
        return observations

    def read_action(self, action):
        # Insert player1's action into full action array, zeros for player2
        full_action = np.zeros(self._full_action_spec.shape, dtype=np.float32)
        full_action[self._player1_action_slice] = action
        # Player2's action remains zero for now
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
