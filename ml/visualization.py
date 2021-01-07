import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import graphics as g


class GUI:
    def __init__(self, size):
        self.colors = ['red1', 'blue1', 'blue2', 'blue3', 'blue4', 'DeepSkyBlue', 
                        'DeepSkyBlue1', 
                        'DeepSkyBlue2', 
                        'DeepSkyBlue3', 
                        'DeepSkyBlue4']
        self.size = size
        self.win = g.GraphWin(height = 800, width = 800) # create a window
        self.win.setCoords(0, 0, size, size)
        self.squares = []
        self.numbers = []
        for i in range(size):
            self.squares.append([])
            self.numbers.append([])
            for j in range(size):
                mySquare = g.Rectangle(g.Point(i, j),
                                          g.Point(i+1, j+1))
                myNum = g.Text(g.Point(i+0.5, j+0.5), "")
                mySquare.draw(self.win)
                myNum.draw(self.win)
                self.squares[i].append(mySquare)
                self.numbers[i].append(myNum)
        
    def loadMap(self, board1, board2, percents, coords):
        for i in range(2, self.size+2):
            for j in range(2, self.size+2):
                if board2[i][j] == False:
                    self.squares[i-2][j-2].setFill(self.colors[int(board1[i, j])+1])
                    self.numbers[i-2][j-2].setText(str(int(board1[i, j])))

                else:
                    self.squares[i-2][j-2].setFill("Green")
                    self.numbers[i-2][j-2].setText(str(round(percents[coords.index((i, j))], 2)) + "%")
                    
                
    def loadColor(self, x, y, color):
        self.squares[x-2][y-2].setFill(color)
       
    def close(self):
        self.win.close()


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
