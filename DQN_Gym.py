import random
import numpy as np
import gym
import imageio  # write env render to mp4
import datetime
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

'''
Original paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- DQN model with Dense layers only
- Model input is changed to take current and n previous states where n = time_steps
- Multiple states are concatenated before given to the model
- Uses target model for more stable training
- More states was shown to have better performance for CartPole env
'''


class DQN:
    def __init__(
            self,
            env,
            memory_cap=1000,
            time_steps=3,
            gamma=0.85,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            learning_rate=0.005,
            batch_size=32,
            tau=0.125
    ):
        self.env = env
        self.memory = deque(maxlen=memory_cap)
        self.state_shape = env.observation_space.shape
        self.time_steps = time_steps
        self.stored_states = np.zeros((self.time_steps, self.state_shape[0]))

        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # amount of randomness in e-greedy policy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay  # exponential decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = tau  # target model update

        self.model = self.create_model_new()
        self.target_model = self.create_model_new()
        self.target_model.trainable = False
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer =  tf.keras.optimizers.Adam(lr=self.learning_rate)

        self.summaries = {}

    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_shape[0] * self.time_steps, activation="relu"))
        model.add(Dense(16, activation="relu"))
        # model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    def create_model_new(self):
        input = Input(shape=self.state_shape[0] * self.time_steps)
        hidden = Dense(16, activation="relu")(input)
        hidden = Dense(8, activation="relu")(hidden)
        q = Dense(self.env.action_space.n)(hidden)
        model = Model(inputs=input, outputs=q)
        return model

    def update_states(self, new_state):
        # move the oldest state to the end of array and replace with new state
        self.stored_states = np.roll(self.stored_states, -1, axis=0)
        self.stored_states[-1] = new_state

    def act(self, test=False):
        states = self.stored_states.reshape((1, self.state_shape[0] * self.time_steps))
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        epsilon = 0.01 if test else self.epsilon  # use epsilon = 0.01 when testing
        q_values = self.model.predict(states)[0]
        self.summaries['q_val'] = max(q_values)
        if np.random.random() < epsilon:
            return self.env.action_space.sample()  # sample random action
        return np.argmax(q_values)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        s = []
        a = []
        r = []
        s_ = []
        done = []
        for sample in samples:
            states, action, reward, new_states, d = sample
            s.append(states)
            a.append([action])
            r.append([reward])
            s_.append(new_states)
            if d:
                done.append([1])
            else:
                done.append([0])
        done = np.asarray(done)

        q_next = self.target_model(np.reshape(s_, (len(s_), self.state_shape[0] * self.time_steps)))
        q_target = r + self.gamma * (1 - done) * tf.reduce_max(q_next, axis=1, keepdims=True)
        with tf.GradientTape() as tape:
            q = self.model(
                np.reshape(s, (len(s), self.state_shape[0] * self.time_steps)))  # (batch_size, s_shape*time_step)
            q_eval = tf.gather(params=q, indices=np.asarray(a), axis=1, batch_dims=1)
            td_error = q_target - q_eval
            q_loss = tf.reduce_mean(tf.square(td_error))
        grads = tape.gradient(q_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.summaries['loss'] = q_loss

    def target_update(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):  # set tau% of target model to be new weights
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        # save model to file, give file name with .h5 extension
        self.model.save(fn)

    def load_model(self, fn):
        # load model from .h5 file
        self.model = tf.keras.models.load_model(fn)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def train(self, max_episodes=300, max_steps=100, save_freq=10):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/DQN_basic_time_step{}/'.format(self.time_steps) + current_time
        summary_writer = tf.summary.create_file_writer(train_log_dir)
        episode = 0
        epoch = 0
        while episode < max_episodes:
            self.stored_states = np.zeros((self.time_steps, self.state_shape[0]))
            done, cur_state, steps, total_reward = False, self.env.reset(), 0, 0
            self.update_states(cur_state)  # update stored states
            while steps < max_steps:
                action = self.act()  # model determine action, states taken from self.stored_states
                new_state, reward, done, _ = self.env.step(action)  # perform action on env
                # modified_reward = 1 - abs(new_state[2] / (np.pi / 2))  # modified for CartPole env, reward based on angle
                prev_stored_states = self.stored_states
                self.update_states(new_state)  # update stored states
                self.remember(prev_stored_states, action, reward, self.stored_states, done)  # add to memory
                self.replay()  # iterates default (prediction) model through memory replay
                if steps%10==0:
                    self.target_update()  # iterates target model

                total_reward += reward
                steps += 1
                epoch += 1
                if done:
                    #if episode % save_freq == 0:  # save model every n episodes
                        #self.save_model("dqn_basic_episode{}_time_step{}.h5".format(episode, self.time_steps))
                    break
                # Tensorboard update
                with summary_writer.as_default():
                    if len(self.memory) > self.batch_size:
                        tf.summary.scalar('Stats/loss', self.summaries['loss'], step=epoch)
                    tf.summary.scalar('Stats/q_val', self.summaries['q_val'], step=epoch)
                    tf.summary.scalar('Main/step_reward', reward, step=epoch)
            with summary_writer.as_default():
                tf.summary.scalar('Main/episode_reward', total_reward, step=episode)
                tf.summary.scalar('Main/episode_steps', steps, step=episode)
            summary_writer.flush()
            print("episode {}: steps:{}  {} reward".format(episode, steps, total_reward))
            episode += 1
        self.save_model("./model/dqn_basic_final_episode{}_time_step{}.h5".format(episode, self.time_steps))

    def test(self, render=True, fps=30, filename='test_render.mp4'):
        cur_state, done, rewards = self.env.reset(), False, 0
        video = imageio.get_writer(filename, fps=fps)
        while not done:
            action = self.act(test=True)
            new_state, reward, done, _ = self.env.step(action)
            self.update_states(new_state)
            rewards += reward
            if render:
                video.append_data(self.env.render(mode='rgb_array'))
        video.close()
        return rewards


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    dqn_agent = DQN(env, time_steps=4)
    # dqn_agent.load_model("basic_models/time_step4/dqn_basic_episode50_time_step4.h5")
    # rewards = dqn_agent.test()
    # print("Total rewards: ", rewards)
    dqn_agent.train()