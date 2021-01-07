import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


class ML:

    def __init__(self):
    model = Sequential()
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2))

    print("")
    print("Starting training....")
    print("")
    
    
    
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
    a = np.random.randint(2, 12)
    b = np.random.randint(2, 12)
    game.tile_click((a, b))
    game.print_board()
    
    graphics = GUI(10)
    while(True):
        probs = []    
        coords = []
        for x in range(2, 12):
            for y in range(2, 12):
                features = []
                if game._covered_board[x, y] == True:
                            for i in range(x-2, x+3):
                                for j in range(y-2, y+3):
                                    if i == x and j == y:
                                        continue
                                    if(game._covered_board[i, j] == True):
                                        features.append(20)
                                    elif(game._board[i, j] == -2):
                                        features.append(20)
                                    else:
                                        features.append(game._board[i, j])  
                            z = probability_model.predict(np.array(features).reshape(1, -1))[0][1]*100
                            probs.append(z)
                            coords.append((x, y))
                            
        graphics.loadMap(game._board, game._covered_board, probs, coords)
        graphics.loadColor(coords[np.argmin(probs)][0], coords[np.argmin(probs)][1], 'yellow')
        game.print_board()
        graphics.win.getMouse()
        tile = game.tile_click(coords[np.argmin(probs)])
        if game.state == Board.State.GameOver:
            print()
            print("You lost... RIP")
            game.reset() 
            game.state = Board.State.Playing
            a = np.random.randint(2, 12)
            b = np.random.randint(2, 12)
            game.tile_click((a, b))
            game.print_board()
