from minesweeper.board import Board
from ml.visualization import GUI

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import os
import pandas as pd
import time

from pybind_testing import create_datapoint, generate_predict_features


# import profilehooks
# import line_profiler
# import atexit

# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)

MAX_DATA_POINTS = 100000

def create_datapoint_python(game, wallSize, neighRange, size):
    # Makes a copy, just in case, to not cause shenanigans outside of function
    b = game.board(with_walls=False).copy()
    cb = game.covered_board(with_walls=False)
    categories = np.arange(10).reshape(-1,1) # 0:8 is the value of the tile, 9 is covered. Bomb tiles not necessary due to redundant

    labels = (b == -1).reshape(-1,1).astype(int)
    
    # Set covered tiles to 9
    b[cb] = 9
    features = (categories == b.ravel()).T.astype(int)

    data_point = np.hstack((features, labels))

    return data_point

def generate_features_predict_python(game, wallSize, neighRange, size):
    # Makes a copy, just in case, to not cause shenanigans outside of function
    b = game.board(with_walls=False).copy()
    cb = game.covered_board(with_walls=False)
    categories = np.arange(10).reshape(-1,1) # 0:8 is the value of the tile, 9 is covered. Bomb tiles not necessary due to redundant

    # Set covered tiles to 9
    b[cb] = 9
    features = (categories == b.ravel()).T.astype(int)

    idxs = np.vstack(cb.nonzero()).T + wallSize

    return features, idxs

# @profile
def generate_data_point(game, board, files_left, num_bombs = 20, size = 10, num_data_points=1000, wallSize = 2, neighRange = 2, ):
    print("\nGathering data...")
    game = Board(grid_size = (size, size), num_bombs = num_bombs, wallSize = wallSize)
    count = 0
    start = time.time()

    if num_data_points > MAX_DATA_POINTS:
        num_of_files = int(num_data_points/MAX_DATA_POINTS)
    else:
        num_of_files = 1
    for files in range(num_of_files):
        dataPoints = []
        curr_num_data_points = len(dataPoints)
        start = time.time()
        while(curr_num_data_points < num_data_points):
            count+= 1
            decimal = curr_num_data_points/num_data_points
            if count < decimal:
                count = int(np.ceil(decimal))
                print("We're {0}% of the way with file {2} out of {3} ETA: {1}"
                        .format(round(decimal*100, 1),
                                round(((((num_of_files/(files+1)/(decimal))))*(time.time()-start) - 
                                        (time.time()-start)), 0),
                        files+1, num_of_files))
            
            game.rand_click()
            data_point = create_datapoint_python(game, wallSize, neighRange, size)

            # data_point = np.array(
            #     create_datapoint(game._covered_board, game._board.astype(np.int32), wallSize, neighRange, size), dtype=np.uint8
            # ).reshape(-1, 801) # We have ((2*neighRange+1)*(2*neighRange+1) - 1)*10 = 800 features and 1 label

            dataPoints.append(data_point)
            curr_num_data_points += data_point.shape[0]
            game.reset()
            
            
            
        end = time.time()
        print(f"done, spent {0}s".format(round(end-start, 1)))
        dataPoints = np.vstack(dataPoints)
        # dataPoints = np.array([np.array(a) for a in dataPoints])
        path = ('./data/dataPoints' + str(num_data_points)+'_'+str(num_bombs)+
                '_'+str(size)+'_'+str(neighRange)+'_num='+str(files+1)+'.csv')
        start = time.time()
        pd.DataFrame(dataPoints).to_csv(path, compression='gzip', chunksize = 100000)
        
        # np.savetxt(path, dataPoints)
        print("Saving took: {0}s".format(round(time.time()-start, 1)))

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
    num_data_points = 1_000_000
    getNewData = False
    num_bombs = 15
    
    game = Board(num_bombs = num_bombs, grid_size=(size, size), wallSize = wallSize)
    path = ('./data/dataPoints' + str(num_data_points)+'_'+str(num_bombs)+
            '_'+str(size)+'_'+str(neighRange)+'.csv')
    
    num_of_files = int(np.ceil(num_data_points/MAX_DATA_POINTS))
    start = time.time()
    
    try:
        for files in range(num_of_files):
            assert os.path.exists(('./data/dataPoints' + str(num_data_points)+'_'+str(num_bombs)+
                '_'+str(size)+'_'+str(neighRange)+'_num='+str(files+1)+'.csv'))
            
    except:
        print("\n\n FSDGSDGSGS \n\n")
        generate_data_point(game, num_bombs = num_bombs, num_data_points=num_data_points,
                                            wallSize=wallSize, neighRange=neighRange)
    end = time.time()
    print(f"took {end-start} s")
    
    
    print("\nStarting training....")
    model = get_model()
    for files in range(num_of_files):
        path = ('./data/dataPoints' + str(num_data_points)+'_'+str(num_bombs)+
                '_'+str(size)+'_'+str(neighRange)+'_num='+str(files+1)+'.csv')
        dataPoints = pd.read_csv(path, compression = 'gzip').to_numpy()
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

        # (dataPoints, coords) = generate_predict_features(game._covered_board, game._board.astype(np.int32), wallSize, neighRange, size)

        dataPoints, coords = generate_features_predict_python(game, wallSize, neighRange, size)

        # Dirty hack to make compatible with Audun code
        coords = [tuple(coord) for coord in coords]

        # data_pointCpp = np.array(dataPoints)
        # dataPoints = dataPoints.reshape(-1, 800) # We have ((2*neighRange+1)*(2*neighRange+1) - 1)*10 = 800 features

        probs = model.predict(dataPoints)[:,-1]*100
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
