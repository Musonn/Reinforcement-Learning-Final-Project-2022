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
            nn.Linear(in_features,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,env.action_space.n))
        
    def forward(self,x):
        return self.net(x)

    def act(self,obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.reshape(-1).unsqueeze(0))

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

        self.online_net = Network(env)

        if use_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print('use', device)

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
        obs, *_ = self.env.reset()
        for _ in range(self.MIN_REPLY_SIZE):
            action = self.env.action_space.sample()
            new_obs, rew, done, *_ = self.env.step(action)
            transition = (obs, action, rew, done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs

            if done:
                obs, *_ = self.env.reset()
        
        return
    
    def train(self, verbose=True) -> Network:
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.init_buffer()

        # Main Training Loop
        obs, *_ = self.env.reset()

        for step in itertools.count():
            epsilon = np.interp(step, [0, self.EPSILON_DECAY], [self.EPSILON_START, self.EPSILON_END])
            rnd_sample = random.random()
            if rnd_sample <= epsilon:
                action = self.env.action_space.sample()
            else:
                action = self.online_net.act(obs)

            new_obs, rew, done, *_ = self.env.step(action)
            transition = (obs, action, rew, done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs

            self.episode_reward += rew
            self.episode_step += 1

            if done:
                self.episode += 1
                
                obs, *_ = self.env.reset()
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

            obses_t = torch.as_tensor(obses, dtype=torch.float32).reshape(self.BATCH_SIZE, -1)
            actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
            rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
            dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
            new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32).reshape(self.BATCH_SIZE, -1)


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