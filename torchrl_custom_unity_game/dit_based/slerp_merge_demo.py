import torch
import json
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge
import shutil

from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Dict
import shutil
import numpy as np

from transformers import Qwen2Config, Qwen2ForCausalLM


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995, eps=1e-8):
    """Perform SLERP (Spherical Linear Interpolation) between two tensors."""
    is_torch = False
    if not isinstance(v0, np.ndarray):
        is_torch = True
        v0 = v0.detach().cpu().float().numpy()
    if not isinstance(v1, np.ndarray):
        is_torch = True
        v1 = v1.detach().cpu().float().numpy()

    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)

    v0 = normalize(v0, eps)
    v1 = normalize(v1, eps)

    dot = np.sum(v0 * v1)

    if np.abs(dot) > DOT_THRESHOLD:
        res = lerp(t, v0_copy, v1_copy)
        return maybe_torch(res, is_torch)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = s0 * v0_copy + s1 * v1_copy

    return maybe_torch(res, is_torch)

def lerp(t, v0, v1):
    return (1 - t) * v0 + t * v1

def maybe_torch(v, is_torch):
    if is_torch:
        return torch.from_numpy(v)
    return v

def normalize(v, eps):
    norm_v = np.linalg.norm(v)
    if norm_v > eps:
        v = v / norm_v
    return v


def load_model_from_folder(folder_path: str):
    """Load a Hugging Face transformer model from a folder."""
    config = AutoConfig.from_pretrained(folder_path)
    model = AutoModel.from_pretrained(folder_path, config=config)
    return model

def interpolate_t(layer_idx, num_layers, t_curve):
    """Interpolate t value for the given layer index based on the t_curve."""
    if layer_idx < 0:
        return t_curve[0]
    if layer_idx >= num_layers - 1:
        return t_curve[-1]
    position = layer_idx / (num_layers - 1) * (len(t_curve) - 1)
    lower_idx = int(position)
    upper_idx = min(lower_idx + 1, len(t_curve) - 1)
    lower_t = t_curve[lower_idx]
    upper_t = t_curve[upper_idx]
    return lerp(position - lower_idx, lower_t, upper_t)

def run_slerp_merge_from_config(merge_config_dict: Dict):
    """Run SLERP-based merging based on a MergeKit-style configuration."""

    slices = merge_config_dict['slices'][0]['sources']
    model_1_path, model_2_path = slices[0]['model'], slices[1]['model']

    model_1 = load_model_from_folder(model_1_path)
    model_2 = load_model_from_folder(model_2_path)

    config_1 = AutoConfig.from_pretrained(model_1_path)
    config_2 = AutoConfig.from_pretrained(model_2_path)

    num_layers = min(config_1.num_hidden_layers, config_2.num_hidden_layers)

    # Extract interpolation parameters from the config
    param_t = {param["filter"]: param["value"] for param in merge_config_dict["parameters"]["t"] if "filter" in param}
    global_t = next((param["value"] for param in merge_config_dict["parameters"]["t"] if "filter" not in param), 0.5)

    model_merged = AutoModel.from_config(model_1.config)  # Create an empty model

    state_dict_1 = model_1.state_dict()
    state_dict_2 = model_2.state_dict()
    merged_state_dict = {}

    for key in state_dict_1.keys():
        if "layer" in key:
            layer_idx = int(key.split(".")[1])  # Extract layer index
            if layer_idx >= num_layers:
                continue

            if "self_attn" in key and "self_attn" in param_t:
                t = interpolate_t(layer_idx, num_layers, param_t["self_attn"])
            elif "mlp" in key and "mlp" in param_t:
                t = interpolate_t(layer_idx, num_layers, param_t["mlp"])
            else:
                t = global_t  # Use global interpolation value if not specified

        else:
            t = global_t  # Use global interpolation for non-layer parameters

        print(key, t)

        slerp_result = slerp(t, state_dict_1[key], state_dict_2[key])
        merged_state_dict[key] = slerp_result

    model_merged.load_state_dict(merged_state_dict)

    # Save merged model
    merge_output_path = "merged_model_my_slerp"
    model_merged.save_pretrained(merge_output_path)

    torch.cuda.empty_cache()
    print("SLERP merging complete! Model saved at:", merge_output_path)

    return merge_output_path


def run_slerp_merge(p1_folder, p2_folder):
    with open(f"{p1_folder}/config.json", 'r') as f:
        p1_config = json.load(f)
    with open(f"{p2_folder}/config.json", 'r') as f:
        p2_config = json.load(f)

    num_layers = min(p1_config['num_hidden_layers'], p2_config['num_hidden_layers'])

    self_attn_t_curve = [0, 0.5, 0.3, 0.7, 1]
    mlp_t_curve = [1, 0.5, 0.7, 0.3, 0]

    merge_config_dict = {'slices': [
        {'sources': [{'model': p1_folder, 'layer_range': [0, num_layers]},
                     {'model': p2_folder, 'layer_range': [0, num_layers]}]}],
        'merge_method': 'slerp',
        'base_model': p1_folder,
        'parameters': {'t': [
            {'filter': 'self_attn', 'value': self_attn_t_curve},
            {'filter': 'mlp', 'value': mlp_t_curve},
            {'value': 0.5}
        ]},
        'dtype': 'float32',
        'tokenizer_source': None}

    merge_output_path = f"merged_model_merge_kit"

    merge_config = MergeConfiguration.model_validate(merge_config_dict)

    my_slerp_model_path = run_slerp_merge_from_config(merge_config_dict)

    run_merge(
        merge_config,
        out_path=merge_output_path,
        options=MergeOptions(
            lora_merge_cache="/tmp",
            cuda=False,
            copy_tokenizer=False,
            lazy_unpickle=False,
            low_cpu_memory=False,
        ),
    )

    torch.cuda.empty_cache()
    print("Done!")

    return merge_output_path, my_slerp_model_path


def create_and_save_random_model(model_path: str):
    """Create and save a small random Qwen model."""
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,)
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(model_path)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", cache_dir="./cache")
    tokenizer.save_pretrained(model_path)


def compare_models(model_path_1: str, model_path_2: str, tolerance: float = 1e-5):
    """Compare two models to check if they are the same within a tolerance."""
    model_1 = AutoModel.from_pretrained(model_path_1)
    model_2 = AutoModel.from_pretrained(model_path_2)

    state_dict_1 = model_1.state_dict()
    state_dict_2 = model_2.state_dict()

    for key in state_dict_1.keys():
        if "layer" in key:
            print("=================")
            print(state_dict_1[key])
            print(state_dict_2[key])
            if not torch.allclose(state_dict_1[key], state_dict_2[key], atol=tolerance):
                print(f"Difference found in layer: {key}")
                return False
    return True


if __name__ == '__main__':
    # Create and save two small random Qwen models
    create_and_save_random_model("model_1")
    create_and_save_random_model("model_2")

    # Merge models using both implementations
    merge_output_path, my_slerp_model_path = run_slerp_merge("model_1", "model_2")

    # Compare the merged models
    if compare_models(merge_output_path, my_slerp_model_path):
        print("The merged models are the same.")
    else:
        print("The merged models are different.")

    # Delete the model files
    shutil.rmtree("model_1")
    shutil.rmtree("model_2")
    shutil.rmtree(merge_output_path)
    shutil.rmtree(my_slerp_model_path)

