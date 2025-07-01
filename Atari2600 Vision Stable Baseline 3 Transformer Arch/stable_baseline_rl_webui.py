import torch.cuda
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env, DummyVecEnv
from envs.my_cartpole import CartPoleEnv
from gpt_policy import CustomActorCriticPolicy

import time

import gradio as gr
from gradio.themes.soft import Soft
from gradio.themes.utils import colors, fonts, sizes
import pandas as pd


class Softy(Soft):
    def __init__(
            self,
            *,
            primary_hue: colors.Color | str = colors.slate,
            secondary_hue: colors.Color | str = colors.gray,
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=colors.gray,
            radius_size="none",
            font=[gr.themes.GoogleFont("Red Hat Display"), gr.themes.GoogleFont("Nabla")],
        )
        super().set(

        )


def train_model(
        total_timesteps=25000,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        stats_window_size=100,
        verbose=1,
        device="cuda",

        # progress=gr.Progress(),
):

    # progress(0, desc="Starting...")
    # Parallel environments
    vec_env = make_vec_env(CartPoleEnv, n_envs=8)

    try:
        model = PPO.load("ppo_cartpole.zip", env=vec_env,
                         learning_rate=learning_rate,
                         n_steps=n_steps,
                         batch_size=batch_size,
                         n_epochs=n_epochs,
                         gamma=gamma,
                         gae_lambda=gae_lambda,
                         clip_range=clip_range,
                         ent_coef=ent_coef,
                         vf_coef=vf_coef,
                         max_grad_norm=max_grad_norm,
                         stats_window_size=stats_window_size,
                         verbose=verbose,
                         device=device)
    except FileNotFoundError as e:
        print("File not exist, initializing model...")

        model = PPO(CustomActorCriticPolicy, vec_env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm,
                    stats_window_size=stats_window_size,
                    verbose=verbose,
                    device=device)

    # Learn =========================================
    import sys
    from stable_baselines3.common.utils import safe_mean

    iteration = 0
    log_interval = 1
    total_timesteps, callback = model._setup_learn(
        total_timesteps=total_timesteps,
        callback=None,
        reset_num_timesteps=True,
        tb_log_name="PPO",
        progress_bar=False,
    )

    callback.on_training_start(locals(), globals())

    assert model.env is not None

    while model.num_timesteps < total_timesteps:
        continue_training = model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)

        if not continue_training:
            break

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
                model.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in model.ep_info_buffer]))
                model.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in model.ep_info_buffer]))
            model.logger.record("time/fps", fps)
            model.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
            model.logger.record("time/total_timesteps", model.num_timesteps, exclude="tensorboard")

            model.logger.dump(step=model.num_timesteps)

        model.train()

        loss = model.logger.name_to_value.get("train/loss")
        entropy_loss = model.logger.name_to_value.get("train/entropy_loss")
        gradient_loss = model.logger.name_to_value.get("train/policy_gradient_loss")
        value_loss = model.logger.name_to_value.get("train/value_loss")
        global loss_curves_dict
        loss_curves_dict["loss"].append(loss)
        loss_curves_dict["entropy_loss"].append(entropy_loss)
        loss_curves_dict["policy_gradient_loss"].append(gradient_loss)
        loss_curves_dict["value_loss"].append(value_loss)

        yield gr.Markdown(f"## Progress {round(progress_percentage*100, 2)}%"), plot_loss_curve()

    callback.on_training_end()

    model.save("ppo_cartpole")
    del model
    torch.cuda.empty_cache()
    vec_env.close()
    del vec_env

    return gr.Markdown("Done!"), plot_loss_curve()


def run_pretrained_model():
    loaded_model = PPO.load("ppo_cartpole", device="cpu")
    vec_env = make_vec_env(CartPoleEnv, n_envs=4)
    obs = vec_env.reset()

    for i in range(512):
        action, _states = loaded_model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        render_img = vec_env.render("rgb_array")
        yield render_img

    vec_env.close()


def plot_loss_curve():
    # Plot the mean scores
    iteration_labels = [g for g in range(len(loss_curves_dict["loss"]))]
    line_labels = []
    for line_key in loss_curves_dict:
        line_labels.extend([line_key for _ in loss_curves_dict[line_key]])
    data_points = []
    for line_key in loss_curves_dict:
        data_points.extend([_ for _ in loss_curves_dict[line_key]])
    data = {"Lines": line_labels, 'Iterations': iteration_labels*len(loss_curves_dict), 'Data': data_points}
    pd_df = pd.DataFrame(data)

    return gr.LinePlot(
        value=pd_df,
        x="Iterations",
        y="Data",
        title="Loss",
        color="Lines",
        color_legend_position="bottom",
        width=1024,
        height=256,
    )


# Global variables
loss_curves_dict = {
    "loss": [],
    "entropy_loss": [],
    "policy_gradient_loss": [],
    "value_loss": [],
}


with gr.Blocks(title="RL WebUI", theme=Softy()) as demo:
    with gr.Tab("Stable Baseline 3"):
        with gr.Tab("Train"):
            model_kwargs = dict(
                total_timesteps=25000,
                learning_rate=3e-4,
                n_steps=128,
                batch_size=64,
                n_epochs=1,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                stats_window_size=100,
                verbose=1,
                device="cuda",
            )

            param_text_boxes = []
            for arg_k in model_kwargs.keys():
                v = model_kwargs[arg_k]
                if v is None:
                    pass  # Skip, don't want to mess with this yet
                elif type(v) is int:
                    param_text_boxes.append(gr.Number(label=arg_k, value=v, step=1))
                elif type(v) is float:
                    param_text_boxes.append(gr.Number(label=arg_k, value=v))
                elif type(v) is str:
                    param_text_boxes.append(gr.Textbox(label=arg_k, value=v))

            train_model_btn = gr.Button(value="Run", variant="primary")
            status_md = gr.Markdown(value="")

            loss_curve_plot = plot_loss_curve()

            train_model_btn.click(fn=train_model, inputs=param_text_boxes, outputs=[status_md, loss_curve_plot])

        with gr.Tab("Run"):
            with gr.Column(variant="panel"):
                img_display = gr.Image()
                run_pretrained_model_btn = gr.Button(value="Run", variant="primary")

                run_pretrained_model_btn.click(fn=run_pretrained_model, inputs=[], outputs=[img_display])

demo.queue()
demo.launch(max_threads=256, server_port=7861, share=False)
