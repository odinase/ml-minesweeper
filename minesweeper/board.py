from enum import Enum
import numpy as np


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
        self._covered_board = np.full((self.w+2*wallSize, self.h+2*wallSize), False)
        self._covered_board[wallSize:(self.w+wallSize), wallSize:(self.h+wallSize)] = True
        self._board = None
        self._state = Board.State.Playing
        self.first_clicked = False
    
    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state
        
    def reset(self):
        self._covered_board = np.full((self.w+self.wallSize*2, self.h+self.wallSize*2), False)
        self._covered_board[self.wallSize:(self.w+self.wallSize), self.wallSize:(self.h+self.wallSize)] = True
        self._board = None
        self._state = Board.State.Playing
        self.first_clicked = False
        
        
    def __make_board(self, coord):
        
        # Place bombs randomly on grid
        self._board = np.zeros(self.grid_size)
        self._board.ravel()[:self.num_bombs] = -1
        np.random.shuffle(self._board.ravel())
        
        if self._board[coord[0] - self.wallSize, coord[1] - self.wallSize] == -1:
            not_bomb_grid = self._board == 0
            r_idx, c_idx = not_bomb_grid.nonzero() # corresponding row and col idxs of bomb-free tiles
            other_tile = np.random.randint(0, r_idx.shape[0])
            self._board[r_idx[other_tile], c_idx[other_tile]] = -1
            self._board[coord[0] - self.wallSize, coord[1] - self.wallSize] = 0
        self._covered_board[coord[0] - self.wallSize, coord[1] - self.wallSize] = False
        for i in range(self.w):
            for j in range(self.h):
                if self._board[i,j] == -1:
                    continue
                neighs = self.neighbours((i, j))
                num_bombs = sum([1 for n in neighs if n == -1])
                self._board[i, j] = num_bombs
        
        for i in range(self.wallSize):
            self._board = np.vstack((
                np.full(self.w + 2*i, -2),
                self._board,
                np.full(self.w + 2*i, -2),
            ))
            self._board = np.hstack((np.full((self.h+2*(i+1),1), -2), self._board, np.full((self.h+2*(i+1),1), -2)))
        return self._board

    def tile_click(self, coord):
        if self.first_clicked == False:
            self._board = self.__make_board(coord)
            self.first_clicked = True
        is_covered = self._covered_board[coord]
        assert is_covered, "Found tile is not Covered!"
        tile = Board.Tile(self._board[coord])
        x, y = coord
        
        self._covered_board[x, y] = False
        
        if tile == Board.Tile.Bomb:
            self._state = Board.State.GameOver
            return tile
        
        if self._board[coord] > 0:
            return tile
        
        for i in range(max(x-1, self.wallSize), min(x+2, self.w+self.wallSize)):
            for j in range(max(y-1, self.wallSize), min(y+2, self.h+self.wallSize)):
                if self._board[i, j] >= 0 and self._covered_board[i, j] == True:
                    self.tile_click((i, j))
                    
        if self.tiles_left() == 0:
            self.set_state(Board.State.Won)
        return tile
    
    
    def tiles_left(self):
        return(sum(sum(self._covered_board))-self.num_bombs)
        
        
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