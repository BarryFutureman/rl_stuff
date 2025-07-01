import torch
from utils_vectors import eval_model, make_parallel_env, load_ppo_models


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="PPO Inference")
    parser.add_argument("--model_name", type=str, default="katana_arm")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--context_length", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--num_episodes", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Load the environment
    test_env = make_parallel_env("haha", args.num_envs, device=device, context_length=args.context_length)

    # Load the models
    actor, critic, _ = load_ppo_models(test_env, device, args.model_name, args.save_dir, model_kwargs={})

    actor_model = actor.module[0].model

    test_env = make_parallel_env("haha", args.num_envs, device=device,
                                 context_length=actor.module[0].config.max_position_embeddings, is_test=True)

    # Evaluate the model
    avg_reward = eval_model(actor, test_env, num_episodes=args.num_episodes)
    print(f"Average Reward over {args.num_episodes} episodes: {avg_reward}")


if __name__ == "__main__":
    main()
