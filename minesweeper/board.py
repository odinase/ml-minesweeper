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
        self.w, self.h = grid_size
        self._covered_board = np.full((self.w, self.h), True)
        self._board = self.__make_board(grid_size, num_bombs)
        self._state = Board.State.Playing

    def __make_board(self, grid_size, num_bombs):
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
        self.print_board()
        return self._board

    def tile_click(self, coord):
        is_covered = self._covered_board[coord]
        assert is_covered, "Found tile is not Covered!"
        tile = Board.Tile(self._board[coord[0]+1, coord[1]+1])

        self._covered_board[coord] = False

        if tile == Board.Tile.Bomb:
            self.state = Board.State.GameOver

        return tile

    def uncover_tile(tile):
        pass

    

    def neighbours(self, coords):
        x, y = coords
        assert 0 <= x < self.w and 0 <= y < self.h, f"out of bound: x: {x} y: {y}"
        return self._board[max(x-1,0):min(x+2,self.w+1), max(y-1,0):min(y+2,self.h+1)].flatten()

    def print_board(self):
        for i in range(1, self.w+1):
            print('|', end='')
            for j in range(1, self.h+1):
                t = int(self._board[i,j])
                if self._covered_board[i-1,j-1]:
                    t = "x"
                print(f" {t:>2} ", end='')
            print('|')
        print('')

    
if __name__ == "__main__":
    game = Board()
    tile = game.tile_click((3, 3))

    print(tile)
    game.print_board()
    tile = game.tile_click((5, 5))

    game.print_board()
