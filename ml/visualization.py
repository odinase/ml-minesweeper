import graphics as g


class GUI:
    def __init__(self, size = 10, wallSize = 2):
        self.colors = ['red1', 'blue1', 'blue2', 'blue3', 'blue4', 'DeepSkyBlue', 
                        'DeepSkyBlue1', 
                        'DeepSkyBlue2', 
                        'DeepSkyBlue3', 
                        'DeepSkyBlue4']
        self.size = size
        self.wallSize = wallSize
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
        wallSize = self.wallSize
        for i in range(wallSize, self.size+wallSize):
            for j in range(wallSize, self.size+wallSize):
                if board2[i][j] == False:
                    self.squares[i-wallSize][j-wallSize].setFill(self.colors[int(board1[i, j])+1])
                    self.numbers[i-wallSize][j-wallSize].setText(str(int(board1[i, j])))

                else:
                    self.squares[i-wallSize][j-wallSize].setFill("Green")
                    self.numbers[i-wallSize][j-wallSize].setText(str(round(percents[coords.index((i, j))], 2)) + "%")
                   
                
    def loadColor(self, x, y, color):
        self.squares[x-self.wallSize][y-self.wallSize].setFill(color)
       
    def close(self):
        self.win.close()