from minesweeper.board import Board
from ml.visualization import GUI

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import os
import pandas as pd
import time

#from pybind_testing import create_datapoint, generate_predict_features


# import profilehooks
# import line_profiler
# import atexit

# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)

MAX_DATA_POINTS = 1000
NUM_BOMBS = 15
SIZE = 10

def create_datapoint_python(game):
    # Makes a copy, just in case, to not cause shenanigans outside of function
    b = game.board(with_walls=False)
    categories = np.arange(9).reshape(-1,1) # 0:8 is the value of the tile, 9 is covered. Bomb tiles not necessary due to redundant

    labels = (b == -1).reshape(-1,1).astype(int)
    # print(labels.ravel())
    # input()
    features = (categories == b.ravel()).T.astype(int)

    data_point = np.hstack((features.ravel(), labels.ravel()))

    return data_point

def get_path(files):
    return str('./data/dataPoints' + str(MAX_DATA_POINTS)+'_'+str(NUM_BOMBS)+
            '_'+str(SIZE)+'_num='+str(files)+'.csv')

def generate_features_predict_python(game):
    # Makes a copy, just in case, to not cause shenanigans outside of function
    b = game.board(with_walls=False)
    categories = np.arange(9).reshape(-1,1) # 0:8 is the value of the tile, 9 is covered. Bomb tiles not necessary due to redundant

    features = (categories == b.ravel()).T.astype(int).flatten()

    return features

# @profile
def generate_data_point(game, files):
    (files_left, files_made) = files
    count = 0
    dataPoints = []
    curr_num_data_points = 0
    start = time.time()
    
    while(curr_num_data_points < MAX_DATA_POINTS):
        count+= 1
        decimal = curr_num_data_points/MAX_DATA_POINTS
        if count < decimal:
            count = int(np.ceil(decimal))
            print("We're {0}% of the way with file {1} out of {2}".format(decimal*100, files_made+1, files_made+files_left))
        
        game.rand_click(SIZE)
        dataPoints.append(create_datapoint_python(game))
        
        curr_num_data_points += 1
        game.reset()
        
    dataPoints = np.vstack(dataPoints)

    start = time.time()
    pd.DataFrame(dataPoints).to_csv(get_path(files_made), compression='gzip', chunksize = 100000)
    
    print("Saving took: {0}s".format(round(time.time()-start, 1)))
    if files_left > 0:
        generate_data_point(game, (files_left-1, files_made+1))

def get_model():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    input_shape = (MAX_DATA_POINTS, SIZE, SIZE, 1)
    model = Sequential()
    model.add(Conv2D(SIZE, (3, 3), input_shape = input_shape, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(400, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(SIZE*SIZE))

    return tf.keras.Sequential([model, tf.keras.layers.Softmax()])



def fit_model(model, X_train, Y_train, epochs = 1):
    model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['BinaryAccuracy'])
    model.fit(X_train, Y_train, epochs=epochs, use_multiprocessing=True, workers = 240)
    
    
if __name__ == "__main__":
    num_data_points = MAX_DATA_POINTS*1
    getNewData = False
    
    game = Board(num_bombs = NUM_BOMBS, grid_size=(SIZE, SIZE))
    
    num_of_files = int(np.ceil(num_data_points/MAX_DATA_POINTS))
    start = time.time()
    
    
    for files in range(num_of_files):
        try:
            assert os.path.exists(get_path(files))
        except:
            generate_data_point(game, (num_of_files-files-1, files))
    end = time.time()
    print(f"took {end-start} s")
    
    
    print("\nStarting training....")
    model = get_model()
    print(model.summary)
    input()
    for files in range(num_of_files):
        dataPoints = pd.read_csv(get_path(files), compression = 'gzip').to_numpy()

        X_train = dataPoints[:,1:-(SIZE*SIZE)]
        Y_train = dataPoints[:,-(SIZE*SIZE):]
        X_train = X_train.reshape(MAX_DATA_POINTS, SIZE*SIZE, 9, 1)
        
        fit_model(model, X_train, Y_train)

    print("\nTraining complete. I am at your service, Master!")
    game.reset() 
    game.rand_click(SIZE)
    graphics = GUI(size=SIZE)
    
    
    while(True):
        probs = []

        dataPoint = generate_features_predict_python(game)

        probs = model.predict_on_batch(dataPoint.reshape(1, -1))
        
        probs = (probs[0]*100).reshape(SIZE, SIZE)
        np.set_printoptions(linewidth=180, precision=2, suppress=True)
        
        prob2 = probs.copy()
        while(True):
            i, j = np.argwhere(prob2 == prob2.min())[0]
            if game._covered_board[(i, j)] == True:
                break
            else:
                prob2[(i, j)] = 10000
        
        graphics.loadMap(game._board, game._covered_board, probs)
        graphics.loadColor(i, j, 'yellow')
        graphics.win.getMouse()

        tile = game.tile_click((i, j))
        print("\n There are {} more squares you need to uncover!".format(game.tiles_left()))
        if game.get_state() != Board.State.Playing:
            if game.get_state() == Board.State.GameOver:
                print("\nYou lost... RIP")
            else:
                print("\nWow, you actually won!")
            graphics.loadMap(game._board, game._covered_board, probs)
            graphics.win.getMouse()
            game.reset() 
            game.rand_click(SIZE)































