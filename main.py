from minesweeper.board import Board
import numpy as np

from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten


if __name__ == "__main__":
    dataPoints = []
    game = Board()
    iii = 0
    iiii = 0
    while(len(dataPoints) < 1000):
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
    
    
    
    
    
    
    
