import random
import numpy as np
import gym
import imageio  # write env render to mp4
import datetime
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten
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
            memory_cap=3000,
            time_steps=3,
            gamma=0.85,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            learning_rate=0.005,
            batch_size=256,
            tau=0.125
    ):
        self.env = EnvDrones(map_size=50, drone_num=1, view_range=10, tree_num=30, human_num=1)
        self.full_state_shape = self.env.full_state_shape
        self.drones_shape = self.env.drones_shape
        self.action_dim = self.env.action_dim
        self.memory = deque(maxlen=memory_cap)

        self.time_steps = time_steps
        #self.stored_states = np.zeros((self.time_steps, self.drones_shape))

        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # amount of randomness in e-greedy policy
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay  # exponential decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = tau  # target model update

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.trainable = False
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer =  tf.keras.optimizers.Adam(lr=self.learning_rate)

        self.summaries = {}

    def create_model(self):
        input1 = Input(shape=self.drones_shape)
        input2 = Input(shape=self.full_state_shape)
        conv1 = Conv2D(16, kernel_size=[4, 4], strides=[1, 1], activation='relu', padding="valid")(input1)
        conv2 = Conv2D(16, kernel_size=[4, 4], strides=[1, 1], activation='relu', padding="valid")(input2)
        f1 = Flatten()(conv1)
        f2 = Flatten()(conv2)
        f = tf.concat((f1, f2), axis=1)
        hidden = Dense(64, activation="relu")(f)
        hidden = Dense(32, activation="relu")(hidden)
        q = Dense(self.env.action_dim)(hidden)
        model = Model(inputs=[input1,input2], outputs=q)
        return model

    def update_states(self, new_state):
        # move the oldest state to the end of array and replace with new state
        self.stored_states = np.roll(self.stored_states, -1, axis=0)
        self.stored_states[-1] = new_state

    def act(self, drone_obs, states, test=False):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        epsilon = 0.01 if test else self.epsilon  # use epsilon = 0.01 when testing
        q_values = self.model.predict([drone_obs, states])[0]
        self.summaries['q_val'] = max(q_values)
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)  # sample random action
        return np.argmax(q_values)

    def remember(self, state, action, reward, new_state, done, all, all_):
        self.memory.append([state, action, reward, new_state, done, all, all_])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        s = []
        a = []
        r = []
        s_ = []
        done = []
        all_s = []
        all_s_ = []
        for sample in samples:
            states, action, reward, new_states, d, all, all_ = sample
            s.append(states)
            a.append(action)
            r.append(reward)
            s_.append(new_states)
            all_s.append(all)
            all_s_.append(all_)
            if d:
                done.append([1])
            else:
                done.append([0])
        done = np.asarray(done)

        q_next = self.target_model([np.asarray(s_), np.asarray(all_s)])
        q_target = r + self.gamma * (1 - done) * tf.reduce_max(q_next, axis=1, keepdims=True)
        with tf.GradientTape() as tape:
            q = self.model([np.asarray(s), np.asarray(all_s_)])  # (batch_size, s_shape*time_step)
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

    def train(self, max_episodes=1000, max_steps=100, save_freq=10):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/DQN_basic_time_step{}/'.format(self.time_steps) + current_time
        summary_writer = tf.summary.create_file_writer(train_log_dir)
        episode = 0
        epoch = 0
        while episode < max_episodes:
            self.env.reset()
            done, steps, total_reward = False, 0, 0
            cur_states = self.env.get_drones_obs()
            all_s = self.env.get_full_obs()
            while steps < max_steps:
                action = self.act(cur_states, all_s)  # model determine action, states taken from self.stored_states
                reward, done = self.env.step(human_act_list=[np.random.randint(0,4) for i in range(self.env.human_num)], drone_act_list=[action])  # perform action on env
                new_state = self.env.get_drones_obs()
                all_s_ = self.env.get_full_obs()
                self.remember(cur_states[0], [action], reward, new_state[0], done, all_s[0], all_s_[0])  # add to memory
                cur_states = new_state
                all_s = all_s_
                self.replay()  # iterates default (prediction) model through memory replay
                if steps%10==0:
                    self.target_update()  # iterates target model
                total_reward += reward[0]
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
                    tf.summary.scalar('Main/step_reward', reward[0], step=epoch)
            with summary_writer.as_default():
                tf.summary.scalar('Main/episode_reward', total_reward, step=episode)
                tf.summary.scalar('Main/episode_steps', steps, step=episode)
            summary_writer.flush()
            print("episode {}: steps:{}  {} reward".format(episode, steps, total_reward))
            episode += 1
        self.save_model("./model/dqn_basic_final_episode{}_time_step{}.h5".format(episode, self.time_steps))

    def test(self,max_episodes=300, max_steps=100):
        self.load_model(fn="./model/dqn_basic_final_episode{}_time_step{}.h5".format(1,1))
        episode = 0

        while episode < max_episodes:
            self.env.reset()
            done, steps, total_reward = False, 0, 0
            cur_states = self.env.get_drones_obs()
            while steps < max_steps:
                action = self.act(states=cur_states)  # model determine action, states taken from self.stored_states
                reward, done = self.env.drone_step(drone_act_list=[action])  # perform action on env
                new_state = self.env.get_drones_obs()
                cur_states = new_state
                total_reward += reward[0]
                steps += 1

                if done:
                    # if episode % save_freq == 0:  # save model every n episodes
                    # self.save_model("dqn_basic_episode{}_time_step{}.h5".format(episode, self.time_steps))
                    break

            print("episode {}: steps:{}  {} reward".format(episode, steps, total_reward))
            episode += 1

from MAEnv.env_Drones.env_Drones import EnvDrones
if __name__ == "__main__":
    dqn_agent = DQN()
    # dqn_agent.load_model("basic_models/time_step4/dqn_basic_episode50_time_step4.h5")
    # rewards = dqn_agent.test()
    # print("Total rewards: ", rewards)
    dqn_agent.train()