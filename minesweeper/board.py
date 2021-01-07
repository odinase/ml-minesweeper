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
    
    def __init__(self, grid_size=(10, 10), num_bombs=10):
        self.grid_size = grid_size
        self.num_bombs = num_bombs
        self.w, self.h = grid_size
        self._covered_board = np.full((self.w+4, self.h+4), False)
        self._covered_board[2:(self.w+2), 2:(self.h+2)] = True
        self._board = self.__make_board()
        self._state = Board.State.Playing
        
    def reset(self):
        self._covered_board = np.full((self.w+4, self.h+4), False)
        self._covered_board[2:(self.w+2), 2:(self.h+2)] = True
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

        self._board = np.vstack((
            np.full(self.w, -2),
            np.full(self.w, -2),
            self._board,
            np.full(self.w, -2),
            np.full(self.w, -2)
        ))

        self._board = np.hstack((np.full((self.h+4,1), -2), np.full((self.h+4,1), -2), self._board, np.full((self.h+4,1), -2), np.full((self.h+4,1), -2)))
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
        
        for i in range(max(x-2, 0), min(x+3, self.w+2)):
            for j in range(max(y-2, 0), min(y+3, self.h+2)):
                if self._board[i, j] >= 0 and self._covered_board[i, j] == True:
                    self.tile_click((i, j))

        return tile

    def neighbours(self, coords):
        x, y = coords
        # assert 2 <= x < self.w+2 and 2 <= y < self.h+2, f"out of bound: x: {x} y: {y}"
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
        for i in range(2, self.w+2):
            print(f'{i}: |', end='')
            for j in range(2, self.h+2):
                t = int(self._board[i,j])
                if self._covered_board[i,j]:
                    t = "x"
                print(f" {t:>2} ", end='')
            print('|')
        print('')

    
if __name__ == "__main__":
    pass