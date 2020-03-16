import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GRU, Dense, Flatten, Input, Dense, Concatenate
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Model


def Actor_network(obs_shape, num_agents, action_shape):
    obs = Input(shape=obs_shape)
    agent_onhot = Input(shape=num_agents)
    old_action = Input(shape=action_shape)
    input_embeding = Dense(32, activation="relu")(tf.concat((obs, agent_onhot, old_action), axis=-1))
    h = GRU(units=16, activation="relu", return_sequences=False)(input_embeding)
    pi = Dense(action_shape, activation="softmax")(h)
    model = Model(inputs=[obs, agent_onhot, old_action], outputs=pi)
    return model

def Critic_network(obs_shape, action_shape, num_agents, state_shape):
    obs = Input(shape=obs_shape)
    state = Input(shape=state_shape)
    agent_onhot = Input(shape=num_agents)
    old_actions = Input(shape=action_shape*num_agents)
    current_actions_without_agent = Input(shape=action_shape * (num_agents-1))
    input_embeding = Dense(32, activation="relu")(tf.concat((current_actions_without_agent, state, obs, agent_onhot, old_actions), axis=-1))
    q = Dense(action_shape)(input_embeding)
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

        self.target_critic.set_weights(self.eval_critic.get_weights())

        self.eval_policy_optimizer = tf.keras.optimizers.RMSprop(learning_rate=5e-4)
        self.eval_critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=5e-4)

    def choose_action(self, obs, last_action, agent_num, epsilon, evaluate=False):
        # 传入的agent_num是一个整数，代表第几个agent，现在要把他变成一个onehot向量
        agent_onehot = np.zeros(self.n_agents)
        agent_onehot[agent_num] = 1.

        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        agent_onehot = tf.convert_to_tensor(agent_onehot, dtype=tf.float32)
        last_action = tf.convert_to_tensor(last_action, dtype=tf.float32)

        pi = self.eval_policy.predict([obs, agent_onehot, last_action])
        if evaluate:
            action = tf.argmax(pi[0])
        else:
            if np.random.rand(1) >= epsilon:  # epslion greedy
                action = np.random.randint(0, self.n_actions)
            else:
                action = tf.argmax(pi[0])
        return action


    def learn(self, batch, max_episode_len, train_step, epsilon=0.9):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        # bacth中的每一项(n_episodes, episode_len, n_agents, 具体维度)
        # 我们采用最简化设计，n_episodes = 1， 即只采一轮数据就进行训练
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = tf.convert_to_tensor(batch[key], dtype=tf.int32)
            else:
                batch[key] = tf.convert_to_tensor(batch[key], dtype=tf.float32)
        u, r, terminated = batch['u'], batch['r'], batch['terminated']
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        if self.args.cuda:
            u = u.cuda()
            mask = mask.cuda()
        # 根据经验计算每个agent的Ｑ值,从而跟新Critic网络。然后计算各个动作执行的概率，从而计算advantage去更新Actor。
        q_values = self._train_critic(batch, max_episode_len, train_step)  # 训练critic网络，并且得到每个agent的所有动作的Ｑ值
        action_prob = self._get_action_prob(batch, max_episode_len, epsilon)  # 每个agent的所有动作的概率

        q_taken = torch.gather(q_values, dim=3, index=u).squeeze(3)  # 每个agent的选择的动作对应的Ｑ值
        pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)  # 每个agent的选择的动作对应的概率
        pi_taken[mask == 0] = 1.0  # 因为要取对数，对于那些填充的经验，所有概率都为0，取了log就是负无穷了，所以让它们变成1
        log_pi_taken = torch.log(pi_taken)


        # 计算advantage
        baseline = (q_values * action_prob).sum(dim=3, keepdim=True).squeeze(3).detach()
        advantage = (q_taken - baseline).detach()
        loss = - ((advantage * log_pi_taken) * mask).sum() / mask.sum()
        self.rnn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
        self.rnn_optimizer.step()


from MAEnv.env_FindGoals.env_FindGoals import EnvFindGoals
def run():
    train_steps = 0
    n_epoch = 1000
    n_episodes = 1
    max_episode_len = 200
    env = EnvFindGoals()
    agents = COMA(n_actions=5, n_agents=2, state_shape=4*10*3, obs_shape=3*3*3)
    for epoch in range(n_epoch):
        episodes = []
        # 收集self.args.n_episodes个episodes
        for episode_idx in range(n_episodes):
            observations, actions_, rewards, states, actions_onehots, dones = [], [], [], [], [], []
            step = 0
            episode_reward = 0
            last_action = np.zeros((2, 5))
            for i in range(max_episode_len):
                obs = [env.get_agt1_obs().reshape(1,-1)[0], env.get_agt2_obs().reshape(1,-1)[0]]
                state = env.get_full_obs().reshape(1,-1)[0]
                actions, actions_onehot = [], []
                for agent_id in range(2): #n_agents
                    # 输入当前agent上一个时刻的动作
                    action = agents.choose_action(obs[agent_id], last_action[agent_id],
                                                  agent_id, epsilon=0.9, evaluate=False)
                    # 生成对应动作的0 1向量
                    action_onehot = np.zeros(5)#n_actions
                    action_onehot[action] = 1
                    actions.append(action)
                    actions_onehot.append(action_onehot)
                    last_action[agent_id] = action_onehot

                reward, done = env.step(actions)

                observations.append(obs)
                states.append(state)
                actions_.append(np.reshape(actions, [2, 1]))#[n_agents, 1]
                actions_onehots.append(actions_onehot)
                rewards.append([reward])

                episode_reward += reward
                step += 1
                if done or step == max_episode_len - 1:
                    dones.append([1])
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

                           terminated=dones.copy()
                           )
            for key in episode.keys():
                episode[key] = np.array([episode[key]])
            episodes.append(episode)

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
        agents.learn(batch=episode_batch, max_episode_len=max_episode_len, train_step=train_steps, epsilon=0.9)
        train_steps += 1
