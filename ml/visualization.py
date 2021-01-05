import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

class Visualizer:
    def __init__(self, w=10, h=10, init_probs=np.random.rand(10, 10)):
        self.h = h
        self.w = w
        fig, axs = plt.subplots()
        z = init_probs

        ax = axs
        c = ax.imshow(z, cmap='viridis', vmin=0, vmax=1,
                    extent=[0, w, 0, h],
                    interpolation='nearest', origin='lower', aspect='auto')
        ax.set_title('image (nearest, aspect="auto")')
        fig.colorbar(c, ax=ax)
        self.write_probs(ax, z)

        self.data = init_probs
        self.c = c
        self.ax = ax
        self.fig = fig

    def write_probs(self, ax, z):
        w, h = z.shape
        for i in range(w):
            for j in range(h):
                text = ax.text(j+0.5, i+0.5, f"{z[i, j]:.2f}",
                        ha="center", va="center", color="w")

    def show(self, t=None, block=False):
        plt.show(block=block)
        if t is not None:
            plt.pause(t)

    def update_index(self, index, value):
        self.data[index] = value
        self.c.set_data(self.data)
        self.write_probs(self.ax, self.data)

    def mask_index(self, index):
        self.data[index] = np.nan
        self.c.set_data(np.ma.masked_values(self.data, np.nan))  

    def draw(self, t=None):
        plt.draw()
        if t is not None:
            plt.pause(t)

def random_idxs(w, h):
    x, y = np.random.randint(0, w), np.random.randint(0, h)
    return x, y

viz = Visualizer()

viz.show(1)

N = 10

for n in range(N):
    idx = random_idxs(10, 10)
    viz.mask_index(idx)
    viz.draw(1)
