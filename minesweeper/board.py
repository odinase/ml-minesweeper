from enum import Enum
import numpy as np

from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

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
    
    def __init__(self, grid_size=(10, 10), num_bombs=7):
        self.grid_size = grid_size
        self.num_bombs = num_bombs
        self.w, self.h = grid_size
        self._covered_board = np.full((self.w+2, self.h+2), False)
        self._covered_board[1:(self.w+1), 1:(self.h+1)] = True
        self._board = self.__make_board(grid_size, num_bombs)
        self._state = Board.State.Playing
        
    def reset(self):
        self._covered_board = np.full((self.w+2, self.h+2), False)
        self._covered_board[1:(self.w+1), 1:(self.h+1)] = True
        self._board = self.__make_board(self.grid_size, self.num_bombs)
        self._state = Board.State.Playing
        
        
    def __make_board(self, grid_size=(10, 10), num_bombs=10):
        # Place bombs randomly on grid
        self._board = np.zeros(grid_size)
        self._board.ravel()[:num_bombs] = -1
        np.random.shuffle(self._board.ravel())

        for i in range(self.w):
            for j in range(self.h):
                if self._board[i,j] == -1:
                    continue
                neighs = self.neighbours((i, j))
                num_bombs = sum([1 for n in neighs if n == -1])
                self._board[i, j] = num_bombs

        self._board = np.vstack((
            np.full(self.w, -2),
            self._board,
            np.full(self.w, -2)
        ))

        self._board = np.hstack((np.full((self.h+2,1), -2), self._board, np.full((self.h+2,1), -2)))
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
        
        for i in range(max(x-1, 0), min(x+2, self.w+1)):
            for j in range(max(y-1, 0), min(y+2, self.h+1)):
                if self._board[i, j] >= 0 and self._covered_board[i, j] == True:
                    self.tile_click((i, j))

        return tile

    def uncover_tile(tile):
        pass

    

    def neighbours(self, coords):
        x, y = coords
        assert 0 <= x < self.w and 0 <= y < self.h, f"out of bound: x: {x} y: {y}"
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
        for i in range(1, self.w+1):
            print(f'{i}: |', end='')
            for j in range(1, self.h+1):
                t = int(self._board[i,j])
                if self._covered_board[i,j]:
                    t = "x"
                print(f" {t:>2} ", end='')
            print('|')
        print('')

    
if __name__ == "__main__":
    
    
    dataPoints = []
    game = Board()
    iii = 0
    iiii = 0
    while(len(dataPoints) < 1000000):
        a = np.random.randint(1, 11)
        b = np.random.randint(1, 11)
        game.tile_click((a, b))
        iii += 1
        if (sum(sum(game._covered_board)) == 99):
            iiii += 1
            game.reset()
            continue
        for y in range(1, 11):
            for x in range(1, 11):
                    appendList = []
                    numInList = False
                    if game._covered_board[x, y] == True:
                        for i in range(x-1, x+2):
                            for j in range(y-1, y+2):
                                if i == x and j == y:
                                    continue
                                if(game._covered_board[i, j] == True):
                                    appendList.append(100)
                                elif(game._board[i, j] == -2):
                                    appendList.append(-1000)
                                else:
                                    appendList.append(game._board[i, j])
                                    numInList = True
                    if numInList:   
                        if game._board[x, y] == -1:
                            appendList.append(1)
                        else:
                            appendList.append(0)
                        dataPoints.append(appendList)
                        
                            
        game.reset()
    dataPoints = np.array([np.array(a) for a in dataPoints])
    
    X_train = dataPoints[:,:-1]
    Y_train = dataPoints[:,-1]
    print(round((iiii/iii)*100, 2))
    print("")
    print("Starting training....")
    print("")
    
    
    model = Sequential()
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2))
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model.fit(X_train, Y_train, epochs=100)
    
    
    
    
    
    probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
    
    print("Training complete. I am at your service, Master!")
    summ = 99
    game.reset() 
    game.state = Board.State.Playing
    a = np.random.randint(1, 11)
    b = np.random.randint(1, 11)
    game.tile_click((a, b))
    game.print_board()
    print((a, b))
    while(True):
        probs = []    
        coords = []
        for x in range(1, 11):
            for y in range(1, 11):
                features = []
                if game._covered_board[x, y] == True:
                            for i in range(x-1, x+2):
                                for j in range(y-1, y+2):
                                    if i == x and j == y:
                                        continue
                                    if(game._covered_board[i, j] == True):
                                        features.append(100)
                                    elif(game._board[i, j] == -2):
                                        features.append(-1000)
                                    else:
                                        features.append(game._board[i, j])
                            z = probability_model.predict(np.array(features).reshape(1, -1))[0][0]
                            # print(z)
                            probs.append(z)
                            coords.append((x, y))
        tile = game.tile_click(coords[np.argmax(probs)])
        game.print_board()
        print(coords[np.argmax(probs)])
        input()
        if game.state == Board.State.GameOver:
            print()
            print("You lost... RIP")
            game.reset() 
            game.state = Board.State.Playing
            a = np.random.randint(1, 11)
            b = np.random.randint(1, 11)
            game.tile_click((a, b))
            game.print_board()
    
    
    
    
    
    
    
