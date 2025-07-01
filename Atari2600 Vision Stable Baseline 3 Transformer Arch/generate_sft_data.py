import time

from trainer.ppo.ppo import PPO
from envs.env_util import make_atari_env
from PIL import Image
import json
import os
# import pandas as pd
# from datasets import Dataset, DatasetDict
# import shutil
import numpy as np

game_name = "Breakout"
action_space = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT"
}
# action_space = {
#     0: "NOOP",
#     1: "FIRE",
#     2: "UP",
#     3: "RIGHT",
#     4: "LEFT",
#     5: "DOWN",
#     6: "UPRIGHT",
#     7: "UPLEFT",
#     8: "DOWNRIGHT",
#     9: "DOWNLEFT",
#     10: "UPFIRE",
#     11: "RIGHTFIRE",
#     12: "LEFTFIRE",
#     13: "DOWNFIRE",
#     14: "UPRIGHTFIRE",
#     15: "UPLEFTFIRE",
#     16: "DOWNRIGHTFIRE",
#     17: "DOWNLEFTFIRE"
# }


def act2action_string(act):
    return action_space[act[0]]


loaded_model = PPO.load(f"atari_{game_name}", device="cpu")
vec_env = make_atari_env(f"{game_name}NoFrameskip-v4", n_envs=1)
obs = vec_env.reset()

obs, total_rewards, dones, info, action = obs, 0, None, None, None
action_history = []
generated_data = []
render_stack = []
num_frame_stack = 3
os.makedirs("generated_datasets/images", exist_ok=True)

for i in range(1024):
    action, _states = loaded_model.predict(obs)
    action_history.append(action)
    obs, rewards, dones, info = vec_env.step(action)

    render_stack.append(obs[0])
    if len(render_stack) > num_frame_stack:
        render_stack.pop(0)

    blended_obs = render_stack[0]
    for img_index in range(1, len(render_stack)):
        blended_obs = blended_obs * 0.8 * np.array([0.5, 0.8, 1.2]) + render_stack[img_index]
    blended_obs = np.clip(np.round(blended_obs), 0, 255).astype(np.uint8)

    frame_image = Image.fromarray(blended_obs)
    frame_image.save(f'generated_datasets/images/img_{i}.png')

    system_prompt = "You are playing the classic Atari2600 Breakout."
    game_context = "\nPAST ACTIONS::"
    context_images = []
    for j in range(num_frame_stack):
        frame_num = i - num_frame_stack + j
        if frame_num >= 0:
            action_at_frame = action_history[frame_num]
            game_context += f"{act2action_string(action_at_frame)}=>"
    current_frame_prompt = f"\n<image>\nNEXT ACTION::"
    context_prompt = system_prompt + game_context + current_frame_prompt

    prediction = f"{act2action_string(action)}"
    context_images.append(f"images/img_{i}.png")

    new_json_item = {"messages": [
                        {"role": "system", "content": context_prompt},
                        {"role": "assistant", "content": prediction},
                        ],
                     "images": context_images}
    generated_data.append(new_json_item)
    with open(f'generated_datasets/breakout.json', 'w') as file:
        json.dump(generated_data, file, indent=4)

    total_rewards += rewards
    display_text = f"{'#'*100}\n# Rewards: {total_rewards}\n## Actions: {action}\n### Dones: {dones}\n### Info: {info}"

    print(display_text)

# df = pd.DataFrame(generated_data)
# df_dict = df.to_dict(orient='list')
# hf_dataset = Dataset.from_dict(df_dict)
#
# hf_dataset.save_to_disk("generated_datasets/breakout")

# time.sleep(2)
# shutil.rmtree("generated_datasets/images")

