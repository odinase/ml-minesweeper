from minesweeper.board import Board
from ml.visualization import GUI
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten


def generate_data_point(game, num_data_points=1000):
    dataPoints = []
    wallSize = 3
    neighRange = 2
    game = Board(num_bombs = 20, wallSize = wallSize)
    while(len(dataPoints) < num_data_points):
        a = np.random.randint(wallSize, 10+wallSize)
        b = np.random.randint(wallSize, 10+wallSize)
        tile = game.tile_click((a, b))
        if tile == Board.Tile.Bomb:
            continue
        for y in range(wallSize, 10+wallSize):
            for x in range(wallSize, 10+wallSize):
                    features = np.empty((0,), dtype=int)
                    if game._covered_board[x, y] == True:
                        for i in range(x-neighRange, x+neighRange+1):
                            for j in range(y-neighRange, y+1+neighRange):
                                one_hot = np.zeros(11, dtype=int)
                                if (i, j) == (x, y):
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

    return X_train, Y_train


if __name__ == "__main__":
    neighRange = 2
    wallSize = neighRange
    size = 10
    game = Board(num_bombs = 20, grid_size=(size, size), wallSize = wallSize)
    num_data_points = 1000
    X_train, Y_train = generate_data_point(game, num_data_points=num_data_points)
    
    print("")
    print("Starting training....")
    print("")
    
    # Train
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
    a = np.random.randint(wallSize, size+wallSize)
    b = np.random.randint(wallSize, size+wallSize)
    game.tile_click((a, b))
    
    graphics = GUI(size=size, wallSize=wallSize)
    while(True):
        probs = []    
        coords = []
        for x in range(wallSize, size+wallSize):
            for y in range(wallSize, size+wallSize):
                features = np.empty((0,), dtype=int)
                if game._covered_board[x, y] == True:
                        for i in range(x-wallSize, x+1+wallSize):
                            for j in range(y-wallSize, y+1+wallSize):
                                one_hot = np.zeros(11)
                                if (i, j) == (x, y):
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
                        probs.append(z)
                        coords.append((x, y))
                            
        graphics.loadMap(game._board, game._covered_board, probs, coords)
        graphics.loadColor(coords[np.argmin(probs)][0], coords[np.argmin(probs)][1], 'yellow')
        graphics.win.getMouse()
        tile = game.tile_click(coords[np.argmin(probs)])
        if game.get_state() == Board.State.GameOver:
            print()
            print("You lost... RIP")
            game.reset() 
            a = np.random.randint(wallSize, size+wallSize)
            b = np.random.randint(wallSize, size+wallSize)
            game.tile_click((a, b))
