from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
#from tensorboardX import SummaryWriter
from MAEnv.env_FindGoals.env_FindGoals import EnvFindGoals
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, in_channels=3, num_actions=5):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)

class DQN_RAM(nn.Module):
    def __init__(self, in_features=4, num_actions=18):
        """
        Initialize a deep Q-learning network for testing algorithm
            in_features: number of features of input.
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN_RAM, self).__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(16, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)





class DQN_MODEL():

    capacity = 1000
    learning_rate = 1e-3
    memory_count = 0
    batch_size = 128
    gamma = 0.995
    update_count = 0

    def __init__(self):
        super(DQN_MODEL, self).__init__()
        self.target_net, self.act_net = DQN(in_channels=3, num_actions=5), DQN(in_channels=3, num_actions=5)
        self.update_taget_net()
        self.memory = [None]*self.capacity
        self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        #self.writer = SummaryWriter('./DQN/logs')

    def update_taget_net(self):
        self.target_net.load_state_dict(self.act_net.state_dict())

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        value = self.act_net(state)
        action_max_value, index = torch.max(value, 1)
        action = index.item()
        if np.random.rand(1) >= 0.9: # epslion greedy
            action = np.random.choice(range(5), 1).item()
        return action

    def store_transition(self,transition):
        index = self.memory_count % self.capacity
        self.memory[index] = transition
        self.memory_count += 1
        return self.memory_count >= self.capacity

    def update(self):
        if self.memory_count >= self.capacity:
            state = torch.tensor([t.state for t in self.memory]).float()
            action = torch.LongTensor([t.action for t in self.memory]).view(-1,1).long()
            reward = torch.tensor([t.reward for t in self.memory]).float()
            next_state = torch.tensor([t.next_state for t in self.memory]).float()

            reward = (reward - reward.mean()) / (reward.std() + 1e-7)
            with torch.no_grad():
                target_v = reward + self.gamma * self.target_net(next_state).max(1)[0]

            #Update...
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size, drop_last=False):
                v = (self.act_net(state).gather(1, action))[index]
                loss = self.loss_func(target_v[index].unsqueeze(1), (self.act_net(state).gather(1, action))[index])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #self.writer.add_scalar('loss/value_loss', loss, self.update_count)
                self.update_count +=1
                if self.update_count % 100 ==0:
                    self.target_net.load_state_dict(self.act_net.state_dict())

# Hyper-parameters
seed = 1
num_episodes = 2000
env = EnvFindGoals()
torch.manual_seed(seed)
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])
def main():

    agent = DQN_MODEL()
    for i_ep in range(num_episodes):
        total_reward = 0
        env.reset()
        state1 = env.get_agt1_obs()
        state2 = env.get_agt2_obs()
        #print(state.shape)
        for t in range(1000):
            #env.render()
            action1 = agent.select_action(state1)
            action2 = agent.select_action(state2)
            reward, done = env.step([action1,action2])
            next_state1 = env.get_agt1_obs()
            next_state2 = env.get_agt1_obs()
            agent.store_transition(Transition(state1, action1, reward[0], next_state1))
            agent.store_transition(Transition(state2, action2, reward[0], next_state2))
            state1 = next_state1
            state2 = next_state2
            total_reward = total_reward + reward[0] + reward[1]
            #agent.writer.add_scalar('live/finish_step', t+1, global_step=i_ep)
            if t%20==0:
                agent.update()
            if done:
                break
        agent.update_taget_net()
        print("episodes {}, total_reward is {} ".format(i_ep, total_reward))



if __name__ == '__main__':
    main()