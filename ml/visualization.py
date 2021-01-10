import graphics as g
import numpy as np


class GUI:
    def __init__(self, size = 10):
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
        
    def loadMap(self, board1, board2, percents):
        for i in range(self.size):
            for j in range(self.size):
                if board2[i][j] == False:
                    self.squares[i][j].setFill(self.colors[int(board1[i, j])+1])
                    self.numbers[i][j].setText(str(int(board1[i, j])))

                else:
                    self.squares[i][j].setFill("Green")
                    self.numbers[i][j].setText(str(np.around(percents[(i, j)], 2)) + "%")
                   
                
    def loadColor(self, x, y, color):
        self.squares[x][y].setFill(color)
       
    def close(self):
        self.win.close()