from minesweeper.board import Board
from ml.visualization import GUI

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import os
import pandas as pd
import time

from pybind_testing import create_datapoint


import profilehooks
import line_profiler
import atexit

profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)

MAX_DATA_POINTS = 1000000

def create_datapoint_python(game, wallSize, neighRange):
    data_point = []
    for y in range(wallSize, 10+wallSize):
        for x in range(wallSize, 10+wallSize):
            features = np.empty((0,), dtype=int)
            if game._covered_board[x, y] == True:
                for i in range(x-neighRange, x+1+neighRange):
                    for j in range(y-neighRange, y+1+neighRange):
                        one_hot = np.zeros(10, dtype=int)
                        if (i, j) == (x, y):
                            continue
                        if game._covered_board[i, j] == True:
                            one_hot[9] = 1
                            features = np.append(features, one_hot)
                        #One hot with drop
                        else:
                            if(game._board[i, j] != -2):
                                if int(game._board[i, j]) == -1:
                                    print("ooops")
                                one_hot[int(game._board[i, j])] = 1
                            features = np.append(features, one_hot)
                if game._board[x, y] == -1:
                    features = np.append(features, 1)
                else:
                    features = np.append(features, 0)
                data_point.append(features)

    return data_point

@profile
def generate_data_point(game, num_bombs = 20, size = 10, num_data_points=1000, wallSize = 2, neighRange = 2):
    print("\nGathering data...")
    dataPoints = []
    dataPointsCpp = []
    game = Board(grid_size = (size, size), num_bombs = num_bombs, wallSize = wallSize)
    count = 0
    start = time.time()

    if num_data_points > MAX_DATA_POINTS:
        num_of_files = int(num_data_points/MAX_DATA_POINTS)
    else:
        num_of_files = 1
    for files in range(num_of_files):
        curr_num_data_points = len(dataPoints)
        start = time.time()
        while(curr_num_data_points < num_data_points):
            count+= 1
            if count == 100:
                count = 0
                decimal = curr_num_data_points/num_data_points
                print("We're {0}% of the way with file {2} out of {3} ETA: {1}"
                        .format(round(decimal*100, 1),
                                round((((1/decimal))*(time.time()-start) - 
                                        (time.time()-start))*(num_of_files/(files+1)), 0),
                        files+1, num_of_files))
            a = np.random.randint(wallSize, 10+wallSize)
            b = np.random.randint(wallSize, 10+wallSize)
            game.tile_click((a, b))
            
            # data_point_py = create_datapoint_python(game, wallSize, neighRange)

            data_point = np.array(
                create_datapoint(game._covered_board, game._board.astype(np.int32), wallSize, neighRange, size)
            ).reshape(-1, 801) # We have ((2*neighRange+1)*(2*neighRange+1) - 1)*10 = 800 features and 1 label

            dataPoints.append(data_point)
            curr_num_data_points += data_point.shape[0]#np.vstack(dataPoints).shape[0]
            # assert np.allclose(np.array(dataPoints).ravel(), np.vstack(dataPointsCpp).ravel()), "Smell"
            game.reset()
        end = time.time()
        print(f"done, spent {end-start} s")
        dataPoints = np.vstack(dataPoints)
        # dataPoints = np.array([np.array(a) for a in dataPoints])
        path = ('./data/dataPoints' + str(num_data_points)+'_'+str(num_bombs)+
                '_'+str(size)+'_'+str(neighRange)+'_num='+str(files+1)+'.csv')
        pd.DataFrame(dataPoints).to_csv(path)

def get_model():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    model = Sequential()
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2))

    return tf.keras.Sequential([model, tf.keras.layers.Softmax()])



def fit_model(model, X_train, Y_train, epochs = 25):
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=epochs, use_multiprocessing=True, workers = 240)
    
    
if __name__ == "__main__":
    neighRange = 4
    wallSize = neighRange
    size = 10
    num_data_points = 100_000#1000000
    getNewData = False
    num_bombs = 15
    
    game = Board(num_bombs = num_bombs, grid_size=(size, size), wallSize = wallSize)
    path = ('./data/dataPoints' + str(num_data_points)+'_'+str(num_bombs)+
            '_'+str(size)+'_'+str(neighRange)+'.csv')
    
    if num_data_points > MAX_DATA_POINTS:
        num_of_files = int(num_data_points/MAX_DATA_POINTS)
    else:
        num_of_files = 1
    
    # try:
    #     for files in range(num_of_files):
    #         assert os.path.exists(('./data/dataPoints' + str(num_data_points)+'_'+str(num_bombs)+
    #             '_'+str(size)+'_'+str(neighRange)+'_num='+str(files+1)+'.csv'))
            
    # except:
    start = time.time()
    generate_data_point(game, num_bombs = num_bombs, num_data_points=num_data_points,
                                            wallSize=wallSize, neighRange=neighRange)
    end = time.time()
    print(f"took {end-start} s")
    
    
    print("\nStarting training....")
    model = get_model()
    for files in range(num_of_files):
        path = ('./data/dataPoints' + str(num_data_points)+'_'+str(num_bombs)+
                '_'+str(size)+'_'+str(neighRange)+'_num='+str(files+1)+'.csv')
        dataPoints = pd.read_csv(path).to_numpy()
        X_train = dataPoints[:,1:-1]
        Y_train = dataPoints[:,-1]
        fit_model(model, X_train, Y_train)

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
