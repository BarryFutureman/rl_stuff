import argparse
from typing import Optional, List
from mlagents.trainers.learn import run_cli
from mlagents.trainers.settings import RunOptions
from mlagents.trainers.cli_utils import load_config

from mlagents.plugins.trainer_type import register_trainer_plugins


def parse_command_line(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--experiment_config_path",
                        default="C:/Files/PycharmProjects/Agents/ml-agents/config/transformers/RollerBot.yaml")
    parser.add_argument("--run_id", default="BallBot")
    parser.add_argument("--force", type=bool, default="true")
    return parser.parse_args(argv)


def main():
    """
    Provides an alternative CLI interface to mlagents-learn, 'mlagents-run-experiment'.
    Accepts a JSON/YAML formatted mlagents.trainers.learn.RunOptions object, and executes
    the run loop as defined in mlagents.trainers.learn.run_cli.
    """
    args = parse_command_line()
    expt_config = load_config(args.experiment_config_path)
    _, _ = register_trainer_plugins()
    args = vars(args)
    args.update(expt_config)
    print(args)

    options = RunOptions.from_dict(args)

    options.checkpoint_settings.force = True
    options.checkpoint_settings.resume = False
    options.checkpoint_settings.inference = False
    options.engine_settings.time_scale = 20
    options.engine_settings.capture_frame_rate = 60  # default 60

    print(options)
    run_cli(options)


if __name__ == "__main__":
    main()
