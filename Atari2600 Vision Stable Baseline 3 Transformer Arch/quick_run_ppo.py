from typing import Any, Callable, Dict, Optional, Type, Union
from envs.env_util import make_atari_env
from trainer.ppo.ppo import PPO
# from architectures.Idefics2.Idefics2_policy_shared import PixelBasedActorCriticPolicy
# from architectures.CommandR.cnn_command_r_policy_shared import PixelBasedActorCriticPolicy
from architectures.CommandR.siglip_command_r_policy_shared import PixelBasedActorCriticPolicy


import time

number_of_envs = 4
number_of_steps = 64
calculated_batch_size = number_of_steps * number_of_envs


vec_env = make_atari_env(f"BreakoutNoFrameskip-v4", n_envs=number_of_envs)
model = PPO(PixelBasedActorCriticPolicy, env=vec_env,
            learning_rate=3e-4,
            n_steps=number_of_steps,
            batch_size=calculated_batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            stats_window_size=100,
            verbose=1,
            device="cuda"
            )

# Learn =========================================
import sys
from stable_baselines3.common.utils import safe_mean

iteration = 0
log_interval = 1
total_timesteps, callback = model._setup_learn(
    total_timesteps=1024,
    callback=None,
    reset_num_timesteps=False,
    tb_log_name="PPO",
    progress_bar=False,
)

callback.on_training_start(locals(), globals())

assert model.env is not None

progress_percentage = 0
while model.num_timesteps < total_timesteps:
    rollout_generator = model.collect_rollouts(model.env, callback, model.rollout_buffer,
                                               n_rollout_steps=model.n_steps)
    for rollout in rollout_generator:
        # print("rollout", rollout.shape)
        pass


    # if not continue_training:
    #     break

    iteration += 1
    model._update_current_progress_remaining(model.num_timesteps, total_timesteps)

    progress_percentage = 1 - model._current_progress_remaining

    # Display training infos
    if log_interval is not None and iteration % log_interval == 0:
        assert model.ep_info_buffer is not None
        time_elapsed = max((time.time_ns() - model.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((model.num_timesteps - model._num_timesteps_at_start) / time_elapsed)
        model.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(model.ep_info_buffer) > 0 and len(model.ep_info_buffer[0]) > 0:
            model.logger.record("rollout/ep_rew_mean",
                                safe_mean([ep_info["r"] for ep_info in model.ep_info_buffer]))
            model.logger.record("rollout/ep_len_mean",
                                safe_mean([ep_info["l"] for ep_info in model.ep_info_buffer]))

        model.logger.record("time/fps", fps)
        model.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        model.logger.record("time/total_timesteps", model.num_timesteps, exclude="tensorboard")

        model.logger.dump(step=model.num_timesteps)

    model.train()

    loss = model.logger.name_to_value.get("train/loss")
    entropy_loss = model.logger.name_to_value.get("train/entropy_loss")
    gradient_loss = model.logger.name_to_value.get("train/policy_gradient_loss")
    value_loss = model.logger.name_to_value.get("train/value_loss")

callback.on_training_end()
