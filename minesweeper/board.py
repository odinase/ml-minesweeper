from enum import Enum
import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
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

class Board:
    class Tile(Enum):
        OutOfBounds = -2
        Bomb = -1
        Blank = 0
        One = 1
        Two = 2
        Three = 3
        Four = 4
        Five = 5
        Six = 6
        Seven = 7
        Eight = 8

    class State(Enum):
        Playing = 0
        GameOver = 1
        Won = 2
    
    def __init__(self, grid_size=(10, 10), num_bombs=10, wallSize = 2):
        self.wallSize = wallSize
        self.grid_size = grid_size
        self.num_bombs = num_bombs
        self.w, self.h = grid_size
        self._covered_board = np.full((self.w+4, self.h+4), False)
        self._covered_board[2:(self.w+2), 2:(self.h+2)] = True
        self._board = self.__make_board()
        self._state = Board.State.Playing
        
    def reset(self):
        self._covered_board = np.full((self.w+self.wallSize*2, self.h+self.wallSize*2), False)
        self._covered_board[self.wallSize:(self.w+self.wallSize), self.wallSize:(self.h+self.wallSize)] = True
        self._board = self.__make_board()
        self._state = Board.State.Playing
        
        
    def __make_board(self):
        # Place bombs randomly on grid
        self._board = np.zeros(self.grid_size)
        self._board.ravel()[:self.num_bombs] = -1
        np.random.shuffle(self._board.ravel())

        for i in range(self.w):
            for j in range(self.h):
                if self._board[i,j] == -1:
                    continue
                neighs = self.neighbours((i, j))
                num_bombs = sum([1 for n in neighs if n == -1])
                self._board[i, j] = num_bombs
        
        for i in range(self.wallSize):
            self._board = np.vstack((
                np.full(self.w, -2),
                self._board,
                np.full(self.w, -2),
            ))
            self._board = np.hstack((np.full((self.h+self.wallSize*2,1), -2), self._board, np.full((self.h+self.wallSize*2,1), -2)))
        return self._board

    def tile_click(self, coord):
        is_covered = self._covered_board[coord]
        assert is_covered, "Found tile is not Covered!"
        tile = Board.Tile(self._board[coord])
        x, y = coord
        
        self._covered_board[x, y] = False
        
        if tile == Board.Tile.Bomb:
            self.state = Board.State.GameOver
            return tile
        
        if self._board[coord] > 0:
            return tile
        
        for i in range(max(x-self.wallSize, 0), min(x+self.wallSize+1, self.w+self.wallSize)):
            for j in range(max(y-self.wallSize, 0), min(y+1+self.wallSize, self.h+self.wallSize)):
                if self._board[i, j] >= 0 and self._covered_board[i, j] == True:
                    self.tile_click((i, j))

        return tile

    def neighbours(self, coords):
        x, y = coords
        return self._board[max(x-1,0):min(x+2,self.w+1), max(y-1,0):min(y+2,self.h+1)].flatten()

    def print_board(self):
        for j in range(0, self.h+1):
            if j == 0:
                print("-", end = '')
            print(f"{j: > 2} ", end=' ')
        print()
        for j in range(0, 4*self.h+7):
            print("-", end='')
        print()
        for i in range(self.wallSize, self.w+self.wallSize):
            print(f'{i}: |', end='')
            for j in range(self.wallSize, self.h+self.wallSize):
                t = int(self._board[i,j])
                if self._covered_board[i,j]:
                    t = "x"
                print(f" {t:>2} ", end='')
            print('|')
        print('')

    
if __name__ == "__main__":
    
    
    dataPoints = []
    wallSize = 3
    neighRange = 2
    game = Board(num_bombs = 20, wallSize = wallSize)
    while(len(dataPoints) < 100000):
        a = np.random.randint(wallSize, 10+wallSize)
        b = np.random.randint(wallSize, 10+wallSize)
        tile = game.tile_click((a, b))
        if tile == Board.Tile.Bomb:
            continue
        for y in range(wallSize, 10+wallSize):
            for x in range(wallSize, 10+wallSize):
                    features = np.empty((0,), dtype=int)
                    numInList = False
                    if game._covered_board[x, y] == True:
                        for i in range(x-neighRange, x+neighRange+1):
                            for j in range(y-neighRange, y+1+neighRange):
                                one_hot = np.zeros(11, dtype=int)
                                if i == x and j == y:
                                    continue
                                if(game._covered_board[i, j] == True):
                                    one_hot[9] = 1
                                    features = np.append(features, one_hot)
                                elif(game._board[i, j] == -2):
                                    one_hot[10] = 1
                                    features = np.append(features, one_hot)
                                else:
                                    one_hot[int(game._board[i, j])] = 1
                                    features = np.append(features, one_hot)
                        if game._board[x, y] == -1:
                            features = np.append(features, 1)
                        else:
                            features = np.append(features, 0)
                        dataPoints.append(features)
                        
                            
        game.reset()
    dataPoints = np.array([np.array(a) for a in dataPoints])
    X_train = dataPoints[:,:-1]
    Y_train = dataPoints[:,-1]
    
    print("")
    print("Starting training....")
    print("")
    
    
    model = Sequential()
    model.add(Dense(110, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2))
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model.fit(X_train, Y_train, epochs=100)
    
    
    
    
    
    probability_model = tf.keras.Sequential([model, 
                                          tf.keras.layers.Softmax()])
    
    # clf = RandomForestClassifier()
    # clf.fit(X_train, Y_train)
    
    print("Training complete. I am at your service, Master!")
    game.reset() 
    game.state = Board.State.Playing
    a = np.random.randint(wallSize, 10+wallSize)
    b = np.random.randint(wallSize, 10+wallSize)
    game.tile_click((a, b))
    
    graphics = GUI(10)
    while(True):
        probs = []    
        coords = []
        for x in range(wallSize, 10+wallSize):
            for y in range(wallSize, 10+wallSize):
                features = np.empty((0,), dtype=int)
                if game._covered_board[x, y] == True:
                            for i in range(x-wallSize, x+1+wallSize):
                                for j in range(y-wallSize, y+1+wallSize):
                                    one_hot = np.zeros(11)
                                    if i == x and j == y:
                                        continue
                                    if(game._covered_board[i, j] == True):
                                        one_hot[9] = 1
                                        features = np.append(features, one_hot)
                                    elif(game._board[i, j] == -2):
                                        one_hot[10] = 1
                                        features = np.append(features, one_hot)
                                    else:
                                        one_hot[int(game._board[i, j])] = 1
                                        features = np.append(features, one_hot)
                            z = probability_model.predict(np.array(features).reshape(1, -1))[0][1]*100
                            # z = clf.predict_proba(np.array(features).reshape(1, -1))[0][1]*100
                            probs.append(z)
                            coords.append((x, y))
                            
        graphics.loadMap(game._board, game._covered_board, probs, coords)
        graphics.loadColor(coords[np.argmin(probs)][0], coords[np.argmin(probs)][1], 'yellow')
        graphics.win.getMouse()
        tile = game.tile_click(coords[np.argmin(probs)])
        if game.state == Board.State.GameOver:
            print()
            print("You lost... RIP")
            game.reset() 
            game.state = Board.State.Playing
            a = np.random.randint(wallSize, 10+wallSize)
            b = np.random.randint(wallSize, 10+wallSize)
            game.tile_click((a, b))
    
    
    
    
    
    
    
