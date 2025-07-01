# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import random

import gym.envs
import gymnasium
import torch.nn
import torch.optim
from tensordict.nn import TensorDictModule
from torchrl.data.tensor_specs import CategoricalBox
from torchrl.envs import (
    CatFrames,
    DoubleToFloat,
    EndOfLifeTransform,
    EnvCreator,
    ExplorationType,
    GrayScale,
    GymEnv,
    GymWrapper,
    set_gym_backend,
    NoopResetEnv,
    ParallelEnv,
    RenameTransform,
    Resize,
    RewardSum,
    SignTransform,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
    UnsqueezeTransform
)
from torchrl.modules import (
    ActorValueOperator,
    ConvNet,
    MLP,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from model import TransformerModule
import os
import shutil
import json
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from unity_env_wrapper import UnityGymSelfPlayWrapper
import portalocker

# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def get_and_increment_worker_id(file_path="worker_id.json"):
    try:
        # Check if the file exists, if not create it with initial worker_id
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump({"worker_id": 0}, f)

        with open(file_path, 'r+') as f:
            # Lock the file for exclusive access
            portalocker.lock(f, portalocker.LOCK_EX)

            # Read the current worker ID
            try:
                worker_data = json.load(f)
            except json.JSONDecodeError:
                worker_data = {"worker_id": 0}

            # Get and increment the worker ID
            worker_id = worker_data.get("worker_id", 0)
            worker_id += 1

            # Go back to the beginning of the file
            f.seek(0)

            # Save the incremented worker ID back to the file
            worker_data["worker_id"] = worker_id
            json.dump(worker_data, f, indent=4)

            # Unlock the file
            portalocker.unlock(f)

        return worker_id
    except Exception as e:
        print(f"Error reading or updating worker_id.json: {e}")
        quit()

def make_base_env(**kwargs):
    from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
    channel = EngineConfigurationChannel()
    unity_kwargs = {}
    unity_kwargs["side_channels"] = [channel]
    channel.set_configuration_parameters(time_scale=1, capture_frame_rate=-1, target_frame_rate=-1)
        
    unity_env = UnityEnvironment(file_name="C:\\Files\\UnityProjects\\RoboticsGym2021\\Builds\\RockGym\\RoboticsGym2021.exe",
                                 worker_id=get_and_increment_worker_id(),
                                 no_graphics=False,
                                 **unity_kwargs)

    # Convert the Unity environment to a Gym environment.
    env = UnityToGymWrapper(unity_env, allow_multiple_obs=False, flatten_branched=True)
    env = UnityGymSelfPlayWrapper(env=env, categorical_action_encoding=True)
    
    # If agent models are provided, load them into the wrapper
    agent_paths = kwargs.get("agents")
    if agent_paths is not None:
        env.load_opponent(agent_paths)
        
    env = TransformedEnv(env)
    return env

def make_parallel_env(env_name, num_envs, device, env_kwargs=None, context_length=4, is_test=False):
    if env_kwargs is None:
        env_kwargs = {}
    env = ParallelEnv(
        num_envs,
        EnvCreator(lambda: make_base_env(**env_kwargs)),
        serial_for_single=True,
        device=device,
    )
    env = TransformedEnv(env)
    env.append_transform(UnsqueezeTransform(dim=-2, in_keys=["observation"], out_keys=["observation"]))  # Add new dimension
    env.append_transform(CatFrames(N=context_length, dim=-2, in_keys=["observation"], out_keys=["observation"]))  # Adjust dim for CatFrames
    env.append_transform(RewardSum())
    env.append_transform(StepCounter(max_steps=4500))
    if not is_test:
        env.append_transform(SignTransform(in_keys=["reward"]))

    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------

def make_ppo_modules(proof_environment, device, model_dir=None, model_kwargs=None):
    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape

    # Define distribution class and kwargs
    if isinstance(proof_environment.action_spec.space, CategoricalBox):
        num_outputs = proof_environment.action_spec.space.n
        distribution_class = torch.distributions.Categorical
        distribution_kwargs = {}
    else:  # is ContinuousBox
        num_outputs = proof_environment.action_spec.shape
        distribution_class = TanhNormal
        distribution_kwargs = {
            "low": proof_environment.action_spec.space.low.to(device),
            "high": proof_environment.action_spec.space.high.to(device),
        }

    # Define input keys
    in_keys = ["observation"]

    # Define the transformer models for policy and value networks
    policy_transformer_model = TransformerModule(
        in_features=int(input_shape[-1]),
        out_features=num_outputs,
        device=device,
        load_path=model_dir + "/Policy" if model_dir else None,
        **model_kwargs
    )

    value_transformer_model = TransformerModule(
        in_features=int(input_shape[-1]),
        out_features=1,
        device=device,
        load_path=model_dir + "/Value" if model_dir else None,
        **model_kwargs
    )

    # Define policy module using transformer
    policy_module = TensorDictModule(
        module=policy_transformer_model,
        in_keys=in_keys,
        out_keys=["logits"],
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=["logits"],
        spec=proof_environment.full_action_spec.to(device),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define value module using transformer
    value_module = ValueOperator(
        module=value_transformer_model,
        in_keys=in_keys,
    )

    # Print model parameter sizes in billions
    total_params = sum(p.numel() for p in policy_transformer_model.parameters()) + \
                   sum(p.numel() for p in value_transformer_model.parameters())
    print(f"Total Model Parameters: {total_params / 1e6:.2f} Million")
    print("Model dtype:", next(policy_transformer_model.parameters()).dtype)
    return policy_module, value_module


def load_ppo_models(proof_environment, device, model_name, save_dir, model_kwargs):
    save_dir_full = save_dir + "/" + model_name if os.path.isdir(save_dir + "/" + model_name) else None
    if save_dir_full and os.path.isdir(save_dir_full + "/Policy") and os.path.isdir(save_dir_full + "/Value"):
        pass
    else:
        save_dir_full = None

    policy_module, value_module = make_ppo_modules(
        proof_environment,
        device=device,
        model_dir=save_dir_full,
        model_kwargs=model_kwargs
    )

    # if save_dir_full:
    #     saved_optimizer_state_dict = torch.load(f'{save_dir_full}/optimizer.pt')
    # else:
    #     saved_optimizer_state_dict = None
    saved_optimizer_state_dict = None

    return policy_module, value_module, saved_optimizer_state_dict


def save_ppo_models(actor, critic, optimizer, model_name, save_dir):
    """Saves the actor and critic models, optimizer state, and the model architecture script."""
    save_dir_full = save_dir + "/" + model_name

    policy_module = actor.module
    actor_model = policy_module[0].model
    value_module = critic.module
    critic_model = value_module.model

    actor_save_path = save_dir_full + "/Policy"
    critic_save_path = save_dir_full + "/Value"
    actor_model.save_pretrained(actor_save_path)
    critic_model.save_pretrained(critic_save_path)

    # Copy model architecture script
    modeling_ivy_path = "modeling_ivy.py"  # Assuming it's in the same directory
    shutil.copy(modeling_ivy_path, os.path.join(actor_save_path, "modeling_ivy.py"))
    shutil.copy(modeling_ivy_path, os.path.join(critic_save_path, "modeling_ivy.py"))

    # Update config.json
    def update_config_json(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        config["auto_map"] = {
            "AutoConfig": "modeling_ivy.IvyConfig",
            "AutoModel": "modeling_ivy.Ivy4RL",
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    update_config_json(os.path.join(actor_save_path, "config.json"))
    update_config_json(os.path.join(critic_save_path, "config.json"))

    # # Save optimizer state
    # torch.save(optimizer.state_dict(), f"{save_dir_full}/optimizer.pt")


def make_ppo_models(env_name, device, model_name, save_dir, model_kwargs):
    proof_environment = make_parallel_env(env_name, 1, device=device)
    policy_module, value_module, saved_optimizer_state_dict = load_ppo_models(
        proof_environment,
        device=device,
        model_name=model_name,
        save_dir=save_dir,
        model_kwargs=model_kwargs,
    )

    return policy_module, value_module, saved_optimizer_state_dict


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------

def eval_model(actor, test_env, num_episodes=3):
    test_rewards = []
    for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards.append(reward.cpu())
    del td_test
    return torch.cat(test_rewards, 0).mean()


if __name__ == '__main__':
    env = make_parallel_env("Hello-v0", num_envs=4, device="cpu")
    td = env.reset()
    print(env.full_action_spec)
    print(env.observation_spec)
    while True:
        td = env.step(env.full_action_spec.rand())
    print(env.transform)  # Print the transform to check the order
