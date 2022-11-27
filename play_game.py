import DQN_model
import gym

env = gym.make('CartPole-v1')

# hyper parameters
GAMMA = 0.99
BATCH_SIZE = 512
BUFFER_SIZE = 50000 #50000 
MIN_REPLY_SIZE = 1000
EPSILON_START = 1.0 #1.0
EPSILON_END = 0.001 #0.02
EPSILON_DECAY = 10000
LR = 5e-4   

CartPoleDQN = DQN_model.DQN(BUFFER_SIZE, env)
CartPoleDQN.set_hyperparameters(LR, MIN_REPLY_SIZE, EPSILON_DECAY,
                            EPSILON_START, EPSILON_END, BATCH_SIZE, GAMMA)
model = CartPoleDQN.train()
