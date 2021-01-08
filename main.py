from minesweeper.board import Board
from ml.visualization import GUI

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import os
import pandas as pd
import time


def generate_data_point(game, num_bombs = 20, size = 10, num_data_points=1000, wallSize = 2, neighRange = 2):
    print("\nGathering data...")
    dataPoints = []
    game = Board(grid_size = (size, size), num_bombs = num_bombs, wallSize = wallSize)
    count = 0
    start = time.time()
    while(len(dataPoints) < num_data_points):
        count+= 1
        if count == 100:
            count = 0
            decimal = len(dataPoints)/num_data_points
            print("We're {0}% of the way with ETA: {1}".format(decimal*100, round((1/decimal)*(time.time()-start), 1)))
        a = np.random.randint(wallSize, 10+wallSize)
        b = np.random.randint(wallSize, 10+wallSize)
        game.tile_click((a, b))
        for y in range(wallSize, 10+wallSize):
            for x in range(wallSize, 10+wallSize):
                    features = np.empty((0,), dtype=int)
                    if game._covered_board[x, y] == True:
                        for i in range(x-neighRange, x+1+neighRange):
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
    path = ('./data/dataPoints' + str(num_data_points)+'_'+str(num_bombs)+
            '_'+str(size)+'_'+str(neighRange)+'.csv')
    pd.DataFrame(dataPoints).to_csv(path)

    return X_train, Y_train

def fit_model(X_train, Y_train, epochs = 25):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    model = Sequential()
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2))
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model.fit(X_train, Y_train, epochs=epochs, use_multiprocessing=True, workers = 240)    
    
    return tf.keras.Sequential([model, 
                                          tf.keras.layers.Softmax()])
    
if __name__ == "__main__":
    neighRange = 4
    wallSize = neighRange
    size = 10
    num_data_points = 1000000
    getNewData = False
    num_bombs = 15
    
    game = Board(num_bombs = num_bombs, grid_size=(size, size), wallSize = wallSize)
    path = ('./data/dataPoints' + str(num_data_points)+'_'+str(num_bombs)+
            '_'+str(size)+'_'+str(neighRange)+'.csv')
    if getNewData == False:
        try:
            dataPoints = pd.read_csv(path,header=0).to_numpy()[:num_data_points]
            X_train = dataPoints[:,:-1]
            Y_train = dataPoints[:,-1]
            print('\nLoaded csv')
        except:
            X_train, Y_train = generate_data_point(game, num_data_points=num_data_points,
                                              wallSize=wallSize, neighRange=neighRange)
    else:
        X_train, Y_train = generate_data_point(game, num_data_points=num_data_points,
                                              wallSize=wallSize, neighRange=neighRange)
        
    print("\nStarting training....")
    
    model = fit_model(X_train, Y_train)

    print("\nTraining complete. I am at your service, Master!")
    game.reset() 
    a = np.random.randint(wallSize, size+wallSize)
    b = np.random.randint(wallSize, size+wallSize)
    game.tile_click((a, b))
    
    graphics = GUI(size=size, wallSize=wallSize)
    while(True):
        probs = []    
        coords = []
        dataPoints = []
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
                        dataPoints.append(features)
                        coords.append((x, y))
        
        probs = model.predict(np.array(dataPoints))[:,-1]*100
        graphics.loadMap(game._board, game._covered_board, probs, coords)
        graphics.loadColor(coords[np.argmin(probs)][0], coords[np.argmin(probs)][1], 'yellow')
        graphics.win.getMouse()
        tile = game.tile_click(coords[np.argmin(probs)])
        print("\n There are {} more squares you need to uncover!".format(game.tiles_left()))
        if game.get_state() != Board.State.Playing:
            if game.get_state() == Board.State.GameOver:
                print("\nYou lost... RIP")
            else:
                print("\nWow, you actually won!")
            graphics.loadMap(game._board, game._covered_board, probs, coords)
            graphics.win.getMouse()
            game.reset() 
            a = np.random.randint(wallSize, size+wallSize)
            b = np.random.randint(wallSize, size+wallSize)
            game.tile_click((a, b))
