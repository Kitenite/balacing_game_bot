import gym
import tensorflow as tf
from tensorflow import keras
from numpy import save, load, array, argmax
import os

env = gym.make('CartPole-v0')
save_path = 'my_model'

# Metric
game_duration = 500
initial_games = 10000
initial_score_cutoff = 50

# create initial dataset
def play_initial_games():
    # Data we're collecting
    # [Observations, moves]
    training_data = []
    for i in range(initial_games):
        observation = env.reset()
        current_score = 0
        # Observation that resulted in our move
        previous_observation = []
        # Moves we made this game [observation, resulting action]
        game_memory = []

        for t in range(game_duration):
            #random action for initial games
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            # Save game data. Ignore first frame no observation
            if len(previous_observation) > 0 :
                game_memory.append([previous_observation, action])
            # This is used to have access to past action
            previous_observation = observation
            current_score += reward

            if done:
                break
        # End of game, save qualified data
        if current_score >= initial_score_cutoff:
            # memory = [observation, action]
            for memory in game_memory:
                # Reformat action into neural network output
                output_action = memory[1]
                if output_action == 0:
                    formatted_action = [1,0]
                elif output_action == 1:
                    formatted_action = [0,1]
                training_data.append([memory[0], formatted_action])
    return training_data

def create_model(input_size):
    if os.path.exists(save_path):
        print("LOAD MODEL")
        # Load model
        model = tf.keras.models.load_model(save_path)
        return model, True

    model = keras.Sequential([
        keras.layers.Dense(input_size, input_shape=(input_size,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
        ])
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model, False

def get_training_data(create_new = True):
    if create_new:
        training_data = play_initial_games()
        save('data.npy', training_data)
    else:
        training_data = load('data.npy', allow_pickle=True)
    return training_data

def train_model(model, training_data):
    # Format data. Reshape into array of shape (-1,4,1)
    x = []
    y = []
    for data in training_data:
        observations = []
        for obs in data[0]:
            observations.append(obs)
        x.append(observations)
        y.append(data[1])
    model.fit(x,y,epochs = 5)
    model.save(save_path)

def play_game(model):
    observation = env.reset()
    score = 0
    previous_observation = []
    for _ in range(game_duration):
        env.render()
        # Random first move
        if len(previous_observation)==0:
            action = 0
        else:
            input_data = []
            for data in previous_observation:
                input_data.append(data)
            action = argmax(model.predict([input_data]))

        new_observation, reward, done, info = env.step(action)
        score+=reward
        previous_observation = new_observation
        if done:
            break
    print "Score: ", score


def main():
    training_data = get_training_data(False)
    input_size = len(training_data[0][0])
    model, trained = create_model(input_size)
    if not trained:
        train_model(model, training_data)
    play_game(model)
    env.close()

if __name__ == '__main__':
    main()
