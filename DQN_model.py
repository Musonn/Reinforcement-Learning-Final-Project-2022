import matplotlib.pyplot as plt
from torch import nn
import torch
from collections import deque
import itertools
import numpy as np
import random

class Network(nn.Module):
    def __init__(self,env):
        super().__init__()
        
        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(env.action_space.n, bias = False))
        
    def forward(self,x):
        return self.net(x)

    def act(self,obs, device):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        obs_t = torch.permute(obs_t, (2,0,1)).to(device)
        q_values = self(obs_t.unsqueeze(0)) # unsqueeze to give a dimension for batch

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action

class DQN():
    def __init__(self, BUFFER_SIZE, env, use_cuda=True):
        self.env = env
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.rew_buffer = deque([0.0],maxlen=100)
        self.n_set = 40
        self.set_size = 10
        self.step_list = deque([0.0],maxlen=self.set_size)

        self.episode = 0
        self.episode_reward = 0.0
        self.episode_step = 0.0
        
        self.result = np.zeros(self.n_set)
        self.counter = 0

        if use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print('use', self.device)

        self.online_net = Network(env).to(self.device)

    def set_hyperparameters(self, LR, MIN_REPLY_SIZE, EPSILON_DECAY, 
                            EPSILON_START, EPSILON_END, BATCH_SIZE, GAMMA):
        self.learning_rate = LR    
        self.MIN_REPLY_SIZE = MIN_REPLY_SIZE
        self.EPSILON_DECAY = EPSILON_DECAY
        self.EPSILON_START = EPSILON_START
        self.EPSILON_END = EPSILON_END
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA

    def init_buffer(self):
        #initiallize replay buffer
        obs = self.env.reset()
        for _ in range(self.MIN_REPLY_SIZE):
            action = self.env.action_space.sample()
            new_obs, rew, done, *_ = self.env.step(action)
            transition = (obs, action, rew, done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs

            if done:
                obs= self.env.reset()
        
        return
    
    def train(self, verbose=True) -> Network:
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.online_net.train()
        self.init_buffer()

        # Main Training Loop
        obs = self.env.reset()

        for step in itertools.count():
            epsilon = np.interp(step, [0, self.EPSILON_DECAY], [self.EPSILON_START, self.EPSILON_END])
            rnd_sample = random.random()
            if rnd_sample <= epsilon:
                action = self.env.action_space.sample()
            else:
                action = self.online_net.act(obs, self.device)

            new_obs, rew, done, *_ = self.env.step(action)
            transition = (obs, action, rew, done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs

            self.episode_reward += rew
            self.episode_step += 1

            if done:
                self.episode += 1
                
                obs= self.env.reset()
                self.step_list.append(self.episode_step)
                self.rew_buffer.append(self.episode_reward)
                self.episode_reward = 0.0
                self.episode_step = 0.0

            #Start Gradient Step
            transitions = random.sample(self.replay_buffer, self.BATCH_SIZE)

            obses = np.asarray([t[0] for t in transitions])
            actions = np.asarray([t[1] for t in transitions])
            rews = np.asarray([t[2] for t in transitions])
            dones = np.asarray([t[3] for t in transitions])
            new_obses = np.asarray([t[4] for t in transitions])

            obses_t = torch.as_tensor(obses, dtype=torch.float32)
            actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
            rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
            dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
            new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

            obses_t = torch.permute(obses_t, (0, 3, 1, 2)).to(self.device)
            actions_t = actions_t.to(self.device)
            rews_t = rews_t.to(self.device)
            dones_t = dones_t.to(self.device)
            new_obses_t = torch.permute(new_obses_t, (0, 3, 1, 2)).to(self.device)

            # Compute Targets
            target_q_values = self.online_net(new_obses_t)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
            targets = rews_t + self.GAMMA*(1 - dones_t)*max_target_q_values

            # Compute Loss
            q_values = self.online_net(obses_t)

            action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

            loss = nn.functional.smooth_l1_loss(action_q_values, targets)

            #Gradient Descent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #Logging
            if verbose:
                if self.episode % (self.set_size+1) == 0:
                    self.episode += 1
                    self.result[self.counter] = np.mean(self.step_list)
                    print("Finishing set {}/{}... Mean score {}".format(self.counter+1,self.n_set, self.result[self.counter]))
                    if self.counter == self.n_set-1:
                        break
                    self.counter+=1
        
        self.env.close()
        return self.online_net

# def customized_rew(new_obs, reward):
#     angle = new_obs[2]
#     bin_list = [-0.418, -0.2, -0.1, 0.1, 0.2, 0.418]
#     angle_idx = np.digitize(angle, bins=bin_list)
#     if angle_idx in [0, 7]:
#       return -1
#     elif angle_idx in [1, 6]:
#       return -1
#     elif angle_idx in [2, 5]:
#       return 1
#     else:
#       return 10