import gym
import random
import imageio
import datetime
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
# Original paper: https://arxiv.org/pdf/1509.02971.pdf

tf.keras.backend.set_floatx('float64')


def actor_discrete(state_shape, action_dim):
    state = Input(shape=state_shape)
    x = Dense(32, activation='tanh')(state)
    x = Dense(16, activation='tanh')(x)
    output = Dense(action_dim)(x)
    model = Model(inputs=state, outputs=output)
    return model


def critic(state_shapes, action_dims):
    inputs = []
    for state_shape in state_shapes:
        inputs.append(Input(shape=state_shape))
    for action_dim in action_dims:
        inputs.append(Input(shape=action_dim))
    concat = Concatenate(axis=-1)(inputs)

    x = Dense(64, activation='tanh')(concat)
    x = Dense(32, activation='tanh')(x)
    x = Dense(16, activation='tanh')(x)
    output = Dense(1)(x)
    model = Model(inputs=inputs, outputs=output)

    return model


def update_target_weights(model, target_model, tau=0.005):
    weights = model.get_weights()
    target_weights = target_model.get_weights()
    for i in range(len(target_weights)):  # set tau% of target model to be new weights
        target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
    target_model.set_weights(target_weights)


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
#noise function for countinus action type
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
class NormalNoise:
    def __init__(self, mu, sigma=0.15):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(scale=self.sigma, size=self.mu.shape)

    def reset(self):
        pass


class MADDPG:
    def __init__(
            self,
            lr_actor=1e-2,
            lr_critic=1e-2,
            noise='norm',
            sigma=0.15,
            tau=0.0125,
            gamma=0.95,
            batch_size=1024,
            memory_cap=100000
    ):
        self.env = make_env()
        self.state_shapes = [os.shape for os in self.env.observation_space]  # shape of observations
        self.action_dims = [a_s.n for a_s in self.env.action_space]  # number of actions

        self.memory = deque(maxlen=memory_cap)
        #if noise == 'ou':
           # self.noises = [OrnsteinUhlenbeckNoise(mu=np.zeros(action_dim), sigma=sigma) for action_dim in self.action_dims]
        #else:
         #   self.noises = [NormalNoise(mu=np.zeros(action_dim), sigma=sigma) for action_dim in self.action_dims]
        self.gumbel_dist = tfp.distributions.Gumbel(0, 1)
        # Define and initialize Actor network
        self.actors = [actor_discrete(state_shape, action_dim) for state_shape, action_dim in zip(self.state_shapes, self.action_dims)]
        self.actor_targets = [actor_discrete(state_shape, action_dim) for state_shape, action_dim in zip(self.state_shapes, self.action_dims)]
        self.actor_optimizers = [tf.keras.optimizers.Adam(learning_rate=lr_actor) for i in range(self.env.n)]

        for i in range(self.env.n):
            update_target_weights(self.actors[i], self.actor_targets[i], tau=1.)

        # Define and initialize Critic network
        self.critics = [critic(self.state_shapes, self.action_dims) for i in range(self.env.n)]
        self.critic_targets = [critic(self.state_shapes, self.action_dims) for i in range(self.env.n)]
        self.critic_optimizers = [Adam(learning_rate=lr_critic) for i in range(self.env.n)]

        #self.critic.compile(loss="mean_squared_error", optimizer=self.critic_optimizer)
        for i in range(self.env.n):
            update_target_weights(self.critics[i], self.critic_targets[i], tau=1.)

        # Set hyperparameters
        self.gamma = gamma  # discount factor
        self.tau = tau  # target model update
        self.batch_size = batch_size

        # Tensorboard
        self.summaries = {}

    def act(self, states, evaluation=False):
        actions_evaluation = []
        actions_noise = []
        for i in range(self.env.n):
            logits = self.actors[i].predict(np.expand_dims(states[i], axis=0))
            mu = np.argmax(logits, -1)
            action_onehot_eval = np.eye(self.action_dims[i])[mu]
            actions_evaluation.append(action_onehot_eval[0])

            cate_dist = tfp.distributions.Categorical(logits)
            pi = cate_dist.sample()
            action_onehot = np.eye(self.action_dims[i])[pi]
            actions_noise.append(action_onehot)

        if evaluation == True:
            return actions_evaluation
        else:
            return actions_noise

    def save_model(self, a_fn, c_fn):
        for i in range(self.env.n):
            self.actors[i].save(a_fn+str(i)+".h5")
            self.critics[i].save(c_fn+str(i)+".h5")

    def load_actor(self, a_fn):
        for i in range(self.env.n):
            self.actors[i].load_weights(a_fn+str(i)+".h5")
        #self.actor_target.load_weights(a_fn)
        #print(self.actor.summary())

    def load_critic(self, c_fn):
        self.critic.load_weights(c_fn)
        self.critic_target.load_weights(c_fn)
        print(self.critic.summary())

    def remember(self, states, actions, rewards, next_states, dones):
        self.memory.append([states, actions, rewards, next_states, dones])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        for agent_id in range(self.env.n):
            #bacth_size x [states, actions, rewards, next_states, dones]
            samples = random.sample(self.memory, self.batch_size)

            #*n_agents*batch_size*具体维度
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_next_states = []
            batch_dones = []
            for i in range(self.env.n):
                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []
                for sample in samples:
                    states.append(sample[0][i])
                    actions.append(sample[1][i])
                    rewards.append(sample[2][i])
                    next_states.append(sample[3][i])
                    dones.append(sample[4][i])
                batch_states.append(np.asarray(states))
                batch_actions.append(np.asarray(actions))
                batch_rewards.append(np.asarray(rewards))
                batch_next_states.append(np.asarray(next_states))
                batch_dones.append(np.asarray(dones))

            with tf.GradientTape(persistent=True) as tape:
                critic_inputs = []
                for i in range(self.env.n):
                    critic_inputs.append(batch_states[i])
                for i in range(self.env.n):
                    critic_inputs.append(batch_actions[i])
                eval_q = self.critics[agent_id](critic_inputs) # batch_size*1


                next_critic_inputs = []
                for i in range(self.env.n):
                    next_critic_inputs.append(batch_next_states[i])
                for i, actor_target in enumerate(self.actor_targets):
                    target_logits = actor_target(batch_next_states[i])
                    #target_cate_dist = tfp.distributions.Categorical(target_logits)
                    #target_pi = target_cate_dist.sample()
                    #action_target = tf.one_hot(target_pi, self.action_dims[i], dtype=tf.float64)
                    action_target = tf.one_hot(tf.argmax(tf.nn.softmax(target_logits),axis=-1), self.action_dims[i], dtype=tf.float64)
                    next_critic_inputs.append(action_target)

                q_future = self.critic_targets[agent_id](next_critic_inputs) # batch_size*1
                target_qs = batch_rewards[agent_id] + q_future * self.gamma * (1. - batch_dones[agent_id])

                critic_loss = tf.reduce_mean(tf.square(target_qs - eval_q))

                logits = self.actors[agent_id](batch_states[agent_id])
                #logp_all = tf.nn.log_softmax(logits)
                #gumbel_noise = tf.cast(self.gumbel_dist.sample([self.batch_size, self.action_dims[agent_id]]), dtype=tf.float64)
                #_pi = tf.nn.softmax((logp_all + gumbel_noise) / 1.0)
                _pi = tf.nn.softmax(logits)
                _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.action_dims[agent_id], dtype=tf.float64)
                _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                actions = _pi_diff + _pi

                c_inputs = []
                for i in range(self.env.n):
                    c_inputs.append(batch_states[i])
                for i in range(self.env.n):
                    if i == agent_id:
                        c_inputs.append(actions)
                    else:
                        c_inputs.append(batch_actions[i])
                actor_loss = -tf.reduce_mean(self.critics[agent_id](c_inputs))

            critic_grad = tape.gradient(critic_loss, self.critics[agent_id].trainable_variables)  # compute actor gradient
            self.critic_optimizers[agent_id].apply_gradients(zip(critic_grad, self.critics[agent_id].trainable_variables))

            actor_grad = tape.gradient(actor_loss, self.actors[agent_id].trainable_variables)  # compute actor gradient
            self.actor_optimizers[agent_id].apply_gradients(zip(actor_grad, self.actors[agent_id].trainable_variables))

            # tensorboard info
            #self.summaries['critic_loss'] = np.mean(hist.history['loss'])
            #self.summaries['actor_loss'] = actor_loss

    def train(self, max_episodes=10000, max_epochs=8000, max_steps=50, save_freq=50):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/DDPG_basic_' + current_time
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        episode = 0
        epoch = 0
        while episode < max_episodes:
            cur_state = self.env.reset()
            total_reward = 0
            steps = 0
            while steps < max_steps:
                #self.env.render()
                action = self.act(cur_state, evaluation=False)  # model determine action given state
                next_state, reward, done, _ = self.env.step(action)  # perform action on env
                D = all(done)
                #print(reward)
                total_reward += np.sum(reward)
                done = [[1] if d else [0] for d in done]
                reward = [[r] for r in reward]
                self.remember(cur_state, action, reward, next_state, done)  # add to memory
                if epoch%100 == 0:
                    self.replay()  # train models through memory replay
                    for i in range(self.env.n):
                        update_target_weights(self.actors[i], self.actor_targets[i], tau=self.tau)
                        update_target_weights(self.critics[i], self.critic_targets[i], tau=self.tau)

                cur_state = next_state
                steps += 1
                epoch += 1

                if D:
                    break

            print("episode {}: {} total reward, {} steps, {} epochs".format(
                episode, total_reward, steps, epoch))
            episode += 1
            # Tensorboard update
            with summary_writer.as_default():
                tf.summary.scalar('Main/total_reward', total_reward, step=episode)
            summary_writer.flush()

        self.save_model("ddpg_actor_", "ddpg_critic_")

    def test(self, episode_num = 100, max_episode_steps = 100):
        episode = 0
        while episode< episode_num:
            rewards = 0
            cur_state = self.env.reset()
            step = 0
            while step < max_episode_steps:
                self.env.render()
                action = self.act(cur_state, evaluation=True)
                next_state, reward, done, _ = self.env.step(action)  # perform action on env
                cur_state = next_state
                rewards += np.sum(reward)
                step += 1
            episode += 1
            print("episode {}: {} total reward, {} steps".format(
                episode, rewards, step))


def make_env():
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load("simple_spread.py").Scenario()
    # create world
    world = scenario.make_world()
    # create openai_multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def test_env():
    env = make_env()
    print(env.observation_space[0])
    print(env.n)
    print(env.action_space)
    env.reset()
    obs_n, reward_n, done_n, info_n = env.step([np.zeros((5)) for i in range(env.n)])
    print(reward_n)
    print(done_n)
if __name__ == "__main__":
    maddpg = MADDPG()
    #maddpg.train()
    maddpg.load_actor("ddpg_actor_")
    maddpg.test()
    #test_env()

