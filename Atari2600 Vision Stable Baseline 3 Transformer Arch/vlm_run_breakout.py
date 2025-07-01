import random
import gradio as gr
from gradio.themes.monochrome import Monochrome
from gradio.themes.utils import colors, fonts, sizes
import gymnasium as gym
from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from envs.env_util import make_atari_env

model_id = "cache/models/breakout_llava_phi3"

prompt = "You are playing the classic Atari2600 Breakout.\nPAST ACTIONS::\n<image>\nNEXT ACTION::"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    cache_dir="cache/models"
).to(0)

processor = AutoProcessor.from_pretrained(model_id)


def run():
    action_space = {
        "NOOP": 0,
        "FIRE": 1,
        "RIGHT": 2,
        "LEFT": 3
    }

    vec_env = make_atari_env(f"BreakoutNoFrameskip-v4", n_envs=1)
    obs = vec_env.reset()
    render_stack = []
    num_frame_stack = 3
    for _ in range(1000):
        render_stack.append(obs[0])
        if len(render_stack) > num_frame_stack:
            render_stack.pop(0)
        blended_obs = render_stack[0]
        for img_index in range(1, len(render_stack)):
            blended_obs = blended_obs * 0.8 * np.array([0.5, 0.8, 1.2]) + render_stack[img_index]
        blended_obs = np.clip(np.round(blended_obs), 0, 255).astype(np.uint8)

        img_obs = Image.fromarray(blended_obs)

        inputs = processor(prompt, [img_obs], return_tensors='pt').to(0, torch.float16)
        output = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        act_tok = processor.decode(output[0][2:], skip_special_tokens=False).strip()
        print(act_tok)
        action = 0
        if act_tok in action_space.keys():
            action = action_space[act_tok]
        else:
            action = random.randint(0, 3)

        obs, rewards, dones, info = vec_env.step(np.array([action]))

        yield img_obs, obs[0], None


with gr.Blocks(title="VLM WebUI", theme=Monochrome()) as demo:
    with gr.Tab("VLM Run"):
        with gr.Column(variant="panel"):
            with gr.Row():
                ai_view_display = gr.Image()
                img_display = gr.Image()
                ai_info_md = gr.Markdown()
            run_pretrained_model_btn = gr.Button(value="Run", variant="primary")

            run_pretrained_model_btn.click(fn=run, inputs=[],
                                           outputs=[ai_view_display, img_display, ai_info_md])

demo.queue()
demo.launch(max_threads=256, server_port=7862, share=False)
