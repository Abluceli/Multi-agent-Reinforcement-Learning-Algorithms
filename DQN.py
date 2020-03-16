import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv3D, Dense, Flatten, Input, Dense, Concatenate
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Model

class critic_q_all(tf.keras.Model):
    def __init__(self, action_size):
        super(critic_q_all, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding='valid', activation='relu')
        self.flatten1 = Flatten()
        #self.layer_x_embeding = Dense(32, activation='tanh')
        self.layer_1 = Dense(16, activation='relu')
        self.layer_2 = Dense(8, activation='relu')
        self.q_value = Dense(action_size)


    def call(self, s):
        conv = self.conv1(s)
        h = self.flatten1(conv)
        h = self.layer_1(h)
        h = self.layer_2(h)
        q = self.q_value(h)
        return q

def Q_network(state_shape, action_shape):
    state = Input(shape=state_shape)
    conv1 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding='valid', activation='relu')(state)
    x = Dense(16, activation="relu")(Flatten()(conv1))
    q = Dense(action_shape)(x)
    model = Model(inputs=state, outputs=q)
    return model


class replay_buffer():
    def __init__(self, buffer_len, batch_size):
        self.buffer = deque(maxlen=buffer_len)
        self.batch_size = batch_size

    def transfrom_store(self, state, action, reward, new_state, done):
        self.buffer.append([state, action, reward, new_state, done])

    def sample(self):
        if len(self.buffer) < self.batch_size:
            samples = -1
        else:
            samples = random.sample(self.buffer, self.batch_size)
        return samples

class DQN():
    def __init__(self, action_dim,
                 lr=5e-4,
                 epslion = 0.9,
                 gamma = 0.995
                 ):
        self.epslion = epslion
        self.gamma = gamma
        self.action_dim = action_dim
        self.q_net = Q_network(state_shape=(3,3,3), action_shape=5)
        self.q_target_net = Q_network(state_shape=(3,3,3), action_shape=5)
        self.replay_buffer = replay_buffer(buffer_len=1000, batch_size=128)
        self.update_target_net_weights()
        #self.lr = tf.keras.optimizers.schedules.PolynomialDecay(lr, max_episode, 1e-10, power=1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def update_target_net_weights(self, ployak=None):
        if ployak is None:
            tf.group([t.assign(s) for t, s in zip(self.q_target_net.weights, self.q_net.weights)])
        else:
            tf.group([t.assign(ployak * t + (1 - ployak) * s) for t, s in zip(self.q_target_net.weights, self.q_net.weights)])

    def choose_action(self, s, evaluation=False):
        action = np.zeros(self.action_dim)
        if evaluation:
            q_values = self.q_net(tf.convert_to_tensor(s, dtype=tf.float32))
            # print(q_values)
            action[tf.argmax(q_values[0])] = 1

        else:
            if np.random.rand(1) >= self.epslion:  # epslion greedy
                action_index = np.random.randint(0, self.action_dim)
                action[action_index] = 1
            else:
                q_values = self.q_net(tf.convert_to_tensor(s, dtype=tf.float32))
                #print(q_values)
                action[tf.argmax(q_values[0])] = 1
        return action

    def learn(self):
        samples = self.replay_buffer.sample()
        if samples == -1:
            return
        else:
            s = []
            a = []
            r = []
            s_ = []
            done = []
            for smaple in samples:
                s.append(smaple[0])
                a.append(smaple[1])
                r.append(smaple[2])
                s_.append(smaple[3])
                done.append(smaple[4])
            s = tf.convert_to_tensor(s, dtype=tf.float32)
            #a = tf.convert_to_tensor(a)
            #r = tf.convert_to_tensor(r)
            s_ = tf.convert_to_tensor(s_, dtype=tf.float32)
            done = np.array(done).astype(int)
            td_error, summaries = self.train(s, a, r, s_, done)

    def train(self, s, a, r, s_, done):
        with tf.GradientTape() as tape:
            q = self.q_net(s)
            q_next = self.q_target_net(s_)
            q_eval = tf.reduce_sum(tf.multiply(q, a), axis=1, keepdims=True)
            q_target = tf.stop_gradient(r + self.gamma * (1 - done) * tf.reduce_max(q_next, axis=1, keepdims=True))
            td_error = q_eval - q_target
            q_loss = tf.reduce_mean(tf.square(td_error))
        grads = tape.gradient(q_loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
        return td_error, dict([
            ['LOSS/loss', q_loss],
            ['Statistics/q_max', tf.reduce_max(q_eval)],
            ['Statistics/q_min', tf.reduce_min(q_eval)],
            ['Statistics/q_mean', tf.reduce_mean(q_eval)]
        ])

from MAEnv.env_FindGoals.env_FindGoals import EnvFindGoals
if __name__ == '__main__':

    env = EnvFindGoals()

    agent = DQN(action_dim=5)
    for i_ep in range(500):
        total_reward = 0
        env.reset()
        state1 = env.get_agt1_obs()

        for t in range(200):
            #env.render()
            action1 = agent.choose_action([state1])
            reward, done = env.step([np.argmax(action1),4])
            next_state1 = env.get_agt1_obs()

            agent.replay_buffer.transfrom_store(state1, action1, reward[0], next_state1, done)

            state1 = next_state1

            total_reward = total_reward + reward[0]
            #agent.writer.add_scalar('live/finish_step', t+1, global_step=i_ep)
            if t % 20 == 0:
                agent.learn()
            if done:
                break
        if i_ep % 5 == 0:
            agent.update_target_net_weights()
        print("episodes {}, total_reward is {} ".format(i_ep, total_reward))
