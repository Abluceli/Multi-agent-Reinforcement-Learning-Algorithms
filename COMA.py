import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GRU, Dense, Flatten, Input, Dense, Concatenate
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Model
import torch
import os

def Actor_network(obs_shape, num_agents, action_shape):
    obs = Input(shape=obs_shape)
    agent_onhot = Input(shape=num_agents)
    old_action = Input(shape=action_shape)
    input_embeding = Dense(64, activation="tanh")(tf.concat((obs, agent_onhot, old_action), axis=-1))
    #h = GRU(units=16, activation="relu", return_sequences=False)(input_embeding)
    hidden = Dense(32, activation="tanh")(input_embeding)
    pi = Dense(action_shape, activation="softmax")(hidden)
    model = Model(inputs=[obs, agent_onhot, old_action], outputs=pi)
    return model

def Critic_network(obs_shape, action_shape, num_agents, state_shape):
    obs = Input(shape=obs_shape)
    state = Input(shape=state_shape)
    agent_onhot = Input(shape=num_agents)
    old_actions = Input(shape=action_shape*num_agents)
    current_actions_without_agent = Input(shape=action_shape*num_agents)
    input_embeding = Dense(128, activation="tanh")(tf.concat((current_actions_without_agent, state, obs, agent_onhot, old_actions), axis=-1))
    hidden = Dense(64, activation="tanh")(input_embeding)
    q = Dense(action_shape)(hidden)
    model = Model(inputs=[current_actions_without_agent, state, obs, agent_onhot, old_actions], outputs=q)
    return model

class COMA():
    def __init__(self, n_actions, n_agents, state_shape, obs_shape):
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.state_shape = state_shape
        self.obs_shape = obs_shape
        self.eval_policy = Actor_network(obs_shape=obs_shape, num_agents=n_agents, action_shape=n_actions)
        # 得到当前agent的所有可执行动作对应的联合Q值，得到之后需要用该Q值和actor网络输出的概率计算advantage
        self.eval_critic = Critic_network(obs_shape=obs_shape, action_shape=n_actions, num_agents=n_agents, state_shape=state_shape)
        self.target_critic = Critic_network(obs_shape=obs_shape, action_shape=n_actions, num_agents=n_agents, state_shape=state_shape)
        self.target_critic.trainable = False
        self.update_target_weights()
        print(self.eval_policy.summary())
        print(self.eval_critic.summary())
        print(self.target_critic.summary())
        self.eval_policy_optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
        #self.eval_critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)
        self.eval_critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)

        self.summary_writer = tf.summary.create_file_writer("./log/coma_rmsprop_optimizer")

    def update_target_weights(self, tau=0.05):
        weights = self.eval_critic.get_weights()
        target_weights = self.target_critic.get_weights()
        for i in range(len(target_weights)):  # set tau% of target model to be new weights
            target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
        self.target_critic.set_weights(target_weights)

    def choose_action(self, obs, last_action, agent_num, epsilon, evaluate=False):
        # 传入的agent_num是一个整数，代表第几个agent，现在要把他变成一个onehot向量
        agent_onehot = np.zeros(self.n_agents)
        agent_onehot[agent_num] = 1

        obs = tf.convert_to_tensor([obs], dtype=tf.float32)
        agent_onehot = tf.convert_to_tensor([agent_onehot], dtype=tf.float32)
        last_action = tf.convert_to_tensor([last_action], dtype=tf.float32)

        pi = self.eval_policy.predict([obs, agent_onehot, last_action])
        if evaluate:
            action = np.argmax(pi[0])
        else:
            if np.random.rand(1) >= epsilon:  # epslion greedy
                action = np.random.randint(0, self.n_actions)
            else:
                action = np.argmax(pi[0])
        return action


    def learn(self, batch, max_episode_len, train_step, epsilon=0.9):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        # bacth中的每一项(n_episodes, episode_len, n_agents, 具体维度)
        # 我们采用最简化设计，n_episodes = 1， 即只采一轮数据就进行训练
        u, r, terminated = batch['u'], batch['r'], batch['terminated']

        with tf.GradientTape(persistent=True) as tape:
            # 根据经验计算每个agent的Ｑ值,从而跟新Critic网络。然后计算各个动作执行的概率，从而计算advantage去更新Actor。
            q_values, critic_loss = self._train_critic(batch, max_episode_len, train_step)  # 训练critic网络，并且得到每个agent的所有动作的Ｑ值
            action_prob = self._get_action_prob(batch, max_episode_len, epsilon)  # 每个agent的所有动作的概率
            q_taken = tf.squeeze(tf.gather(q_values, axis=3, indices=u, batch_dims=3), axis=3)  # 每个agent的选择的动作对应的Ｑ值
            pi_taken = tf.squeeze(tf.gather(action_prob, axis=3, indices=u, batch_dims=3), axis=3)  # 每个agent的选择的动作对应的概率
            log_pi_taken = tf.math.log(pi_taken)
            # 计算advantage
            baseline = tf.stop_gradient(tf.squeeze(tf.reduce_sum(tf.multiply(q_values, action_prob), axis=3, keepdims=True), axis=3))
            advantage = tf.stop_gradient(q_taken - baseline)
            policy_loss = - tf.reduce_sum(tf.multiply(advantage, log_pi_taken))


        critic_grads = tape.gradient(critic_loss, self.eval_critic.trainable_variables)
        self.eval_critic_optimizer.apply_gradients(zip(critic_grads, self.eval_critic.trainable_variables))

        policy_grads = tape.gradient(policy_loss, self.eval_policy.trainable_variables)
        self.eval_policy_optimizer.apply_gradients(zip(policy_grads, self.eval_policy.trainable_variables))

        if train_step > 0 and train_step % 10 == 0:
            self.update_target_weights()

        with self.summary_writer.as_default():
            tf.summary.scalar('Loss/polcy_loss', policy_loss, step=train_step)
            tf.summary.scalar('Loss/critic_loss', critic_loss, step=train_step)
        self.summary_writer.flush()

    def _get_action_prob(self, batch, max_episode_len, epsilon):
        episode_num = batch['o'].shape[0]

        action_prob = []
        obs = batch['o']  # (n_episodes, episode_len, n_agents, n_actions)
        agent_onhot = batch['a_onehot']  # (n_episodes, episode_len, n_agents, n_actions)
        u_onehot = batch['u_onehot']  # (n_episodes, episode_len, n_agents, n_actions)
        old_u_onehot = u_onehot[:, :-1]
        padded_old_u_onehot = np.zeros((u_onehot.shape[0], 1, u_onehot.shape[2], u_onehot.shape[3]))  # (n_episodes, episode_len, n_agents, 具体维度)

        old_u_onehot = np.concatenate((padded_old_u_onehot, old_u_onehot), axis=1)
        obs = np.reshape(obs, (obs.shape[0]*obs.shape[1]*obs.shape[2], obs.shape[3]))
        agent_onhot = np.reshape(agent_onhot, (agent_onhot.shape[0] * agent_onhot.shape[1] * agent_onhot.shape[2], agent_onhot.shape[3]))
        old_u_onehot = np.reshape(old_u_onehot, (old_u_onehot.shape[0] * old_u_onehot.shape[1] * old_u_onehot.shape[2], old_u_onehot.shape[3]))
        #model = Model(inputs=[obs, agent_onhot, old_action], outputs=pi)
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        agent_onhot = tf.convert_to_tensor(agent_onhot, dtype=tf.float32)
        old_u_onehot = tf.convert_to_tensor(old_u_onehot, dtype=tf.float32)
        action_prob = self.eval_policy([obs, agent_onhot, old_u_onehot])
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组

        action_prob = ((1 - epsilon) * action_prob + tf.ones_like(action_prob) * epsilon / self.n_actions)
        # 因为上面把不能执行的动作概率置为0，所以概率和不为1了，这里要重新正则化一下。执行过程中Categorical会自己正则化。

        action_prob = action_prob/tf.reduce_sum(action_prob, axis=-1, keepdims=True)
        action_prob = tf.reshape(action_prob, (batch['u_onehot'].shape[0], batch['u_onehot'].shape[1], batch['u_onehot'].shape[2], batch['u_onehot'].shape[3]))
        return action_prob


    def _train_critic(self, batch, max_episode_len, train_step):
        # bacth中的每一项(n_episodes, episode_len, n_agents, 具体维度)
        # 我们采用最简化设计，n_episodes = 1， 即只采一轮数据就进行训练
        u, r, terminated = batch['u'], batch['r'], batch['terminated']
        u_next = u[:, 1:]
        padded_u_next = np.zeros((u.shape[0], 1, u.shape[2], u.shape[3]), dtype=np.int)#(n_episodes, episode_len, n_agents, 具体维度)
        u_next = np.concatenate((u_next, padded_u_next), axis=1)

        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents，n_actions)
        # q_next_target为下一个状态-动作对应的target网络输出的Q值，没有包括reward
        # model = Model(inputs=[current_actions_without_agent, state, obs, agent_onhot, old_actions], outputs=q)
        '''
                           o=observations.copy(),
                           s=states.copy(),
                           u=actions_.copy(),
                           r=rewards.copy(),
                           o_next=observations_next.copy(),
                           s_next=states_next.copy(),
                           u_onehot=actions_onehots.copy(),
                           terminated=dones.copy(),
                           a_onehot = agents_onehot.copy()
        '''
        #get eval_critic inputs
        u_onehot = batch['u_onehot'] #(n_episodes, episode_len, n_agents, n_actions)
        old_u_onehot = u_onehot[:, :-1]
        padded_old_u_onehot = np.zeros((u_onehot.shape[0], 1, u_onehot.shape[2], u_onehot.shape[3]))  # (n_episodes, episode_len, n_agents, 具体维度)
        old_u_onehot = np.concatenate((padded_old_u_onehot, old_u_onehot), axis=1)

        next_u_onehot = u_onehot[:, 1:]
        padded_next_u_onehot = np.zeros((u_onehot.shape[0], 1, u_onehot.shape[2], u_onehot.shape[3]))  # (n_episodes, episode_len, n_agents, 具体维度)
        next_u_onehot = np.concatenate((padded_next_u_onehot, next_u_onehot), axis=1)

        u_onehot = np.reshape(u_onehot, (batch['u_onehot'].shape[0]*batch['u_onehot'].shape[1],
                                     batch['u_onehot'].shape[2]*batch['u_onehot'].shape[3]))
        u_onehot = np.repeat(np.expand_dims(u_onehot, axis=1), repeats=self.n_agents, axis=1)
        #(n_episodes*episode_len, n_agents, n_agents*n_actions)
        u_onehot_for_eval_critic = u_onehot
        u_onehot_for_target_critic = u_onehot

        u_onehot_for_eval_critic_set_agent_o = np.ones(shape=(u_onehot_for_eval_critic.shape[0], u_onehot_for_eval_critic.shape[1], u_onehot_for_eval_critic.shape[2]))
        for i in range(self.n_agents):
            u_onehot_for_eval_critic_set_agent_o[:, i, i*self.n_actions:i*self.n_actions+self.n_actions] = 0
        u_onehot_for_eval_critic = u_onehot_for_eval_critic * u_onehot_for_eval_critic_set_agent_o
        #u_onehot_for_eval_critic = tf.convert_to_tensor(u_onehot_for_eval_critic, dtype=tf.float32)

        old_u_onehot = np.reshape(old_u_onehot, (batch['u_onehot'].shape[0] * batch['u_onehot'].shape[1],
                                             batch['u_onehot'].shape[2] * batch['u_onehot'].shape[3]))
        old_u_onehot = np.repeat(np.expand_dims(old_u_onehot, axis=1), repeats=self.n_agents, axis=1)
        # (n_episodes*episode_len, n_agents, n_agents*n_actions)

        current_actions_without_agent = np.reshape(u_onehot_for_eval_critic, (u_onehot_for_eval_critic.shape[0]*u_onehot_for_eval_critic.shape[1], u_onehot_for_eval_critic.shape[2]))
        state = np.reshape(batch['s'], (batch['s'].shape[0] * batch['s'].shape[1] * batch['s'].shape[2], -1))
        obs = np.reshape(batch['o'], (batch['o'].shape[0] * batch['o'].shape[1] * batch['o'].shape[2], -1))
        agent_onhot = np.reshape(batch['a_onehot'], (batch['a_onehot'].shape[0] * batch['a_onehot'].shape[1] * batch['a_onehot'].shape[2], -1))
        old_actions = np.reshape(old_u_onehot, (old_u_onehot.shape[0]*old_u_onehot.shape[1], old_u_onehot.shape[2]))

        # get target_critic inputs
        next_u_onehot = np.reshape(next_u_onehot, (batch['u_onehot'].shape[0] * batch['u_onehot'].shape[1],
                                     batch['u_onehot'].shape[2] * batch['u_onehot'].shape[3]))
        next_u_onehot = np.repeat(np.expand_dims(next_u_onehot, axis=1), repeats=self.n_agents, axis=1)
        # (n_episodes*episode_len, n_agents, n_agents*n_actions)
        next_u_onehot = next_u_onehot * u_onehot_for_eval_critic_set_agent_o

        target_current_actions_without_agent = np.reshape(next_u_onehot, (next_u_onehot.shape[0] * next_u_onehot.shape[1], next_u_onehot.shape[2]))
        target_state = np.reshape(batch['s_next'], (batch['s_next'].shape[0] * batch['s_next'].shape[1] * batch['s_next'].shape[2], -1))
        target_obs = np.reshape(batch['o_next'], (batch['o_next'].shape[0] * batch['o_next'].shape[1] * batch['o_next'].shape[2], -1))
        target_agent_onhot = np.reshape(batch['a_onehot'], (batch['a_onehot'].shape[0] * batch['a_onehot'].shape[1] * batch['a_onehot'].shape[2], -1))
        target_old_actions = np.reshape(u_onehot_for_target_critic, (u_onehot_for_target_critic.shape[0] * u_onehot_for_target_critic.shape[1], u_onehot_for_target_critic.shape[2]))


        current_actions_without_agent = tf.convert_to_tensor(current_actions_without_agent, dtype=tf.float32)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        agent_onhot = tf.convert_to_tensor(agent_onhot, dtype=tf.float32)
        old_actions = tf.convert_to_tensor(old_actions, dtype=tf.float32)
        target_current_actions_without_agent = tf.convert_to_tensor(target_current_actions_without_agent, dtype=tf.float32)
        target_state = tf.convert_to_tensor(target_state, dtype=tf.float32)
        target_obs = tf.convert_to_tensor(target_obs, tf.float32)
        target_agent_onhot = tf.convert_to_tensor(target_agent_onhot, dtype=tf.float32)
        target_old_actions = tf.convert_to_tensor(target_old_actions, dtype=tf.float32)


        # (n_episodes*episode_len*n_agents, n_actions)
        q_evals = self.eval_critic([current_actions_without_agent, state, obs, agent_onhot, old_actions])
        q_next_target = self.target_critic([target_current_actions_without_agent, target_state, target_obs, target_agent_onhot, target_old_actions])
        q_evals = tf.reshape(q_evals, (batch['o_next'].shape[0], batch['o_next'].shape[1], batch['o_next'].shape[2], self.n_actions))
        q_next_target = tf.reshape(q_next_target, (batch['o_next'].shape[0], batch['o_next'].shape[1], batch['o_next'].shape[2], self.n_actions))
        # (n_episodes, episode_len, n_agents, n_actions)

        q_values = q_evals  # 在函数的最后返回，用来计算advantage从而更新actor
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了

        # u: (n_episodes, episode_len, n_agents, 1)
        # q_evals: (n_episodes, episode_len, n_agents, n_actions)
        q_evals = tf.squeeze(tf.gather(params=q_evals, indices=u, axis=3, batch_dims=3), axis=3)# (n_episodes, episode_len, n_agents)

        q_next_target = tf.squeeze(tf.gather(params=q_next_target, indices=u_next, axis=3, batch_dims=3), axis=3)# (n_episodes, episode_len, n_agents)

        targets = self.td_lambda_target(batch, max_episode_len, q_next_target)#(episode_num, max_episode_len, n_agents)

        td_error = tf.stop_gradient(tf.convert_to_tensor(targets, dtype=tf.float32)) - q_evals#(episode_num, max_episode_len, n_agents)

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = tf.reduce_sum(tf.square(td_error))
        # print('Loss is ', loss)

        return q_values, loss

    def td_lambda_target(self, batch, max_episode_len, q_targets, gamma=0.99, td_lambda=0.8):  # 用来通过TD(lambda)计算y
        # batch维度为(episode个数, max_episode_len， n_agents，n_actions)
        # q_targets维度为(episode个数, max_episode_len， n_agents)
        episode_num = batch['o'].shape[0]
        #mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        terminated = (1 - batch["terminated"]).repeat(repeats= self.n_agents, axis=2)# 用来把episode最后一条经验中的q_target置0
        # 把reward维度从(episode个数, max_episode_len, 1)变成(episode个数, max_episode_len, n_agents)
        r = batch['r'].repeat(repeats= self.n_agents, axis=2)
        # 计算n_step_return

        '''
        1. 每条经验都有若干个n_step_return，所以给一个最大的max_episode_len维度用来装n_step_return
        最后一维,第n个数代表 n+1 step。
        2. 因为batch中各个episode的长度不一样，所以需要用mask将多出的n-step return置为0，
        否则的话会影响后面的lambda return。第t条经验的lambda return是和它后面的所有n-step return有关的，
        如果没有置0，在计算td-error后再置0是来不及的
        3. terminated用来将超出当前episode长度的q_targets和r置为0
        '''
        n_step_return = np.zeros((episode_num, max_episode_len, self.n_agents, max_episode_len))
        for transition_idx in range(max_episode_len - 1, -1, -1):
            # 最后计算1 step return
            n_step_return[:, transition_idx, :, 0] = (r[:, transition_idx] + gamma * q_targets[:, transition_idx] * terminated[:, transition_idx])
            # 经验transition_idx上的obs有max_episode_len - transition_idx个return, 分别计算每种step return
            # 同时要注意n step return对应的index为n-1
            for n in range(1, max_episode_len - transition_idx):
                # t时刻的n step return =r + gamma * (t + 1 时刻的 n-1 step return)
                # n=1除外, 1 step return =r + gamma * (t + 1 时刻的 Q)
                n_step_return[:, transition_idx, :, n] = (r[:, transition_idx] + gamma *
                                                          n_step_return[:, transition_idx + 1, :, n - 1])
            # 计算lambda return
        '''
        lambda_return 维度为(episode个数, max_episode_len， n_agents)，每条经验中，每个agent都有一个lambda return
        '''
        lambda_return = np.zeros((episode_num, max_episode_len, self.n_agents))
        for transition_idx in range(max_episode_len):
            returns = np.zeros((episode_num, self.n_agents))
            for n in range(1, max_episode_len - transition_idx):
                returns += pow(td_lambda, n - 1) * n_step_return[:, transition_idx, :, n - 1]
            lambda_return[:, transition_idx] = (1 - td_lambda) * returns + \
                                               pow(td_lambda, max_episode_len - transition_idx - 1) * \
                                               n_step_return[:, transition_idx, :, max_episode_len - transition_idx - 1]
        return lambda_return


    def save_model(self, a_fn, c_fn):
        self.eval_policy.save(a_fn)
        self.eval_critic.save(c_fn)

    def load_model(self, a_fn, c_fn):
        self.eval_policy.load_weights(a_fn)
        self.eval_critic.load_weights(c_fn)
        self.target_critic.load_weights(c_fn)
        print(self.eval_policy.summary())
        print(self.eval_critic.summary())
        print(self.target_critic.summary())


from MAEnv.env_FindGoals.env_FindGoals import EnvFindGoals
from MAEnv.env_Drones.env_Drones import EnvDrones
def run():
    train_steps = 0
    n_epoch = 1000
    n_episodes = 1
    max_episode_len = 100
    env = EnvDrones(map_size=50, drone_num=5, view_range=10, tree_num=30, human_num=20)
    n_actions = 4
    n_agents = 5
    state_shape = 50 * 50 * 3
    obs_shape = (2 * 10 - 1) * (2 * 10 - 1) * 3
    epsilon = 0.8
    agents = COMA(n_actions=n_actions, n_agents=n_agents, state_shape=state_shape, obs_shape=obs_shape)
    for epoch in range(n_epoch):
        episodes = []
        # 收集self.args.n_episodes个episodes
        for episode_idx in range(n_episodes):
            env.reset()
            observations, actions_, rewards, states, actions_onehots, agents_onehots, dones = [], [], [], [], [], [], []
            step = 0
            episode_reward = 0
            last_action = np.zeros((n_agents, n_actions))

            for i in range(max_episode_len):
                #env.render()
                obs = [env.get_drone_obs(env.drone_list[i]).reshape((1, -1))[0] for i in range(len(env.drone_list))]
                state = env.get_joint_obs().reshape(1,-1).repeat(n_agents, axis=0)
                actions, actions_onehot = [], []
                agents_onehot = np.zeros((n_agents,n_agents))
                for agent_id in range(n_agents): #n_agents
                    # 输入当前agent上一个时刻的动作
                    action = agents.choose_action(obs[agent_id], last_action[agent_id],
                                                  agent_id, epsilon=epsilon, evaluate=False)
                    # 生成对应动作的0 1向量
                    action_onehot = np.zeros(n_actions)#n_actions
                    action_onehot[action] = 1

                    agents_onehot[agent_id, agent_id] = 1
                    actions.append(action)
                    actions_onehot.append(action_onehot)
                    last_action[agent_id] = action_onehot

                agents_onehots.append(agents_onehot)

                reward, done = env.drone_step(drone_act_list=actions)

                observations.append(obs)
                states.append(state)
                actions_.append(np.reshape(actions, [n_agents, 1]))#[n_agents, 1]
                actions_onehots.append(actions_onehot)
                rewards.append([reward])

                episode_reward += reward
                step += 1
                if done:
                    dones.append([1])
                    break
                else:
                    dones.append([0])
            # 处理最后一个obs
            observations.append(obs)
            states.append(state)
            observations_next = observations[1:]
            states_next = states[1:]
            observations = observations[:-1]
            states = states[:-1]

            episode = dict(o=observations.copy(),
                           s=states.copy(),
                           u=actions_.copy(),
                           r=rewards.copy(),
                           o_next=observations_next.copy(),
                           s_next=states_next.copy(),
                           u_onehot=actions_onehots.copy(),
                           terminated=dones.copy(),
                           a_onehot = agents_onehots.copy()
                           )
            for key in episode.keys():
                episode[key] = np.array([episode[key]])
            episodes.append(episode)
            print('epoch: {}, episode: {}, episode_reward: {}, episode_steps:{}'.format(epoch, episode_idx, episode_reward, step))
            with agents.summary_writer.as_default():
                tf.summary.scalar('Main/episode_reward', episode_reward, step=epoch)
            agents.summary_writer.flush()
        # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs,action等信息拼在一起

        episode_batch = episodes[0]
        episodes.pop(0)
        for episode in episodes:
            for key in episode_batch.keys():
                episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
        #episode_bacth中的每一项(n_episodes, episode_len, n_agents, 具体维度)
        #我们采用最简化设计，n_episodes = 1， 即只采一轮数据就进行训练
        terminated = episode_batch['terminated']
        max_episode_len = terminated.shape[1]
        agents.learn(batch=episode_batch, max_episode_len=max_episode_len, train_step=train_steps, epsilon=epsilon)
        train_steps += 1
    agents.save_model(a_fn="./model/eval_policy.h5", c_fn="./model/eval_critic.h5")

if __name__ == '__main__':
    run()