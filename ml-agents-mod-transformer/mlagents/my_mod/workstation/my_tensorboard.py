import time

from tensorboard import program


def launch_tensorboard():
    tb = program.TensorBoard()

    tb.configure(argv=[None,
                       '--logdir', "results/ppo",
                       "--port", '6006',
                       "--reload_interval", '5'])

    url = tb.launch()
    print(f'TensorBoard is running at {url}')

    while True:
        time.sleep(1)


if __name__ == '__main__':
    import subprocess

    # Run the tensorboard command
    command = ["tensorboard", "--logdir=results/ppo", "--port=6006", "--reload_interval=1"]
    subprocess.run(command)
