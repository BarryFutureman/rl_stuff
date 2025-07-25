import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot(scores, mean_scores):
    plt.style.use('dark_background')
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Reward Graph')
    plt.xlabel('Round')
    plt.ylabel('Reward')
    plt.plot(scores, color=(63/255, 114/255, 175/255), label='scores')
    plt.plot(mean_scores, color=(252/255, 255/255, 231/255), label='mean')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
