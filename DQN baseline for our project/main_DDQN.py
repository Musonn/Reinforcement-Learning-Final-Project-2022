import matplotlib.pyplot as plt
from torch import nn
import torch
import gym
import numpy as np
import random
from collections import deque
from network import DQN_NETWORK
import torch.nn.functional as F

from wrappers import *
import argparse


def env_wrapper(game):
    # create and update the environment
    env = gym.make(game)
    env = NoopResetEnv(env, noop_max=30)
    # original observation space 210 * 160 return max_frame in the _obs_buffer with frame skip 4
    env = MaxAndSkipEnv(env, skip=frame_skip_length)
    # calculate lives and return true done for specific game
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    # important add function for warp frame to 84*84*1
    env = WarpFrame(env)
    # print(env.observation_space.shape)
    # normlize and change to 1*84*84 shape
    env = PyTorchFrame(env)
    # print(env.observation_space.shape)
    # add reward function to clip reward to -1,0,1
    env = ClipRewardEnv(env)
    # print(env.observation_space.shape)
    # action = 0
    # ob, reward, done, info = env.step(action)
    # print(ob.shape)
    env = FrameStack(env, stack_length)
    # env = gym.wrappers.Monitor(env, './video/', video_callable=lambda episode_id: episode_id % 100 == 0, force=True)
    return env


if __name__ == '__main__':

    # game name 'PongNoFrameskip-v4' 'Breakout-v0'
    # game = 'PongNoFrameskip-v4'
    # game = 'BreakoutNoFrameskip-v4'
    # game = "ALE/Enduro-v5"
    game = 'SeaquestNoFrameskip-v4'
    print('=======================\n' + 'Now Playing' + game + '\n=======================')
    # parameters and device settings
    PARAM = True
    use_DDQN = True
    if PARAM:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('using', device)
        EPSILON_START = 1.0
        EPSILON_END = 0.01
        EPSILON_DECAY = 300000
        total_num_steps = int(1e8)
        replay_size = 50000
        replay_init_size = 1000
        frame_skip_length = 4
        # for replay_buffer initialization
        MIN_REPLY_SIZE = 1000
        gamma = 0.99
        minibatch_size = 64
        stack_length = 4
        counter = 0
        counter2 = 0
    set_size = 50
    n_set = 100
    network_save_count = int(set_size *n_set)        # save the checkpoint at # th episode
    results_save_count = int(set_size *n_set)          # save rewards and steps at # th episodes
    target_network_update_freq = 3000
    # episode number simulated
    episode = 0
    episode_reward = 0.0
    episode_step = 0
    # 
    replay_buffer = deque(maxlen=replay_size)
    rew_buffer = deque([0.0], maxlen=set_size)
    step_list = deque([0.0], maxlen=set_size)
    result = np.zeros(n_set)


    # ------------------Load passed number of steps, episodes and last epsilon------------------
    try:
        a = np.loadtxt('results/steps.csv', delimiter=',')
        finished_episodes = int(a[0])
        passed_steps = int(a[1])
        last_epsilon = a[2]
        EPSILON_DECAY -= passed_steps
        EPSILON_START = last_epsilon
        print('Load previous training...\nEpisode = {}\nSteps = {}\nEpsilon = {}'.format(a[0],a[1],a[2]))

    except:
        print('Starting new training...\nStep = 0\nReward = 0\nEpsilon = 1.0')

        finished_episodes = 0
        passed_steps = 0
        last_epsilon = 1

    # ------------------load previous results------------------
    try:
        step_every_ep = np.loadtxt('results/rews_and_steps/steps_per_episode'+passed_steps+'.csv', delimiter=',')
        rew_every_ep = np.loadtxt('results/rews_and_steps/rewards_per_episode' + passed_steps + '.csv', delimiter=',')
        qvalue_evert_ep = np.loadtxt('results/rews_and_steps/qvalues_per_episode' + passed_steps + '.csv', delimiter=',')

    except:
        print('Starting new training...')
        rew_every_ep = []
        step_every_ep = []
        qvalue_evert_ep = []

    # ------------------create environment wrapper------------------
    env = env_wrapper(game)
    state_stack = env.reset()
    # ------------------initialize the replay buffer------------------
    replay_buffer = deque(maxlen=replay_size)
    for _ in range(MIN_REPLY_SIZE):
        action = env.action_space.sample()
        new_state_stack, rew, done, _ = env.step(action)
        transition = (state_stack, action, rew, done, new_state_stack)
        replay_buffer.append(transition)
        state_stack = new_state_stack
        if done:
            state_stack = env.reset()


    # ------------------initialize target and main network same------------------
    if use_DDQN == True:
      target_network = DQN_NETWORK(env.observation_space,env.action_space).to(device)
    main_network = DQN_NETWORK(env.observation_space,env.action_space).to(device)
    
    if finished_episodes != 0:
        main_network.load_state_dict(torch.load(f'results/checkpoints/breakout_checkpoint' + str(
            int(finished_episodes)) + f'.pth'))
        print(f'load save network at results/checkpoints/breakout_checkpoint' + str(
            int(finished_episodes)) + f'.pth')
    
    if use_DDQN == True:
      target_network.load_state_dict(main_network.state_dict())
    # ------------------initialize optimizer------------------
    optimizer = torch.optim.Adam(main_network.parameters(), lr=1e-4)

    ##optimizer = torch.optim.RMSprop(main_network.parameters(), lr=0.00025, momentum=0.95)


    for t in range(total_num_steps):

        # fraction method for sample code
        # use easier decay epsilon in lab
        if EPSILON_DECAY <=0:
            current_epsilon = EPSILON_END
        else:
            current_epsilon = np.interp(t, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

        # ------------------perform epsilon greedy action selection
        ran_num = random.random()
        if ran_num < current_epsilon:
            action = env.action_space.sample()
        else:
            #4*84*84
            state_stack = np.array(state_stack)
            #normlize
            state_stack_norm = np.array(state_stack) / 255.0
            state_stack_norm = torch.from_numpy(state_stack_norm).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = main_network(state_stack_norm)
                #print(q_values.shape)
                #break
                _, action = q_values.max(1)
            action = action.item()


        # ------------------perform one action
        next_state_stack, reward, done, info = env.step(action)

        # ------------------store one transition
        transition = (state_stack, action, reward, done, next_state_stack)
        # ------------------append to replay buffer
        replay_buffer.append(transition)
        # ------------------update state
        state_stack = next_state_stack
        episode_step += 1
        episode_reward += reward
        if done:
            print('epsode {}: '.format(
                finished_episodes + episode + 1) + "reward: {} ".format(
                episode_reward) + '|| q_value: {:2f} ||'.format(torch.mean(main_q_values))+" Passed steps: {}  || Total steps:{} || Epsilon: {}".format(episode_step, passed_steps + t + 1,current_epsilon))
            if (((episode + 1) % network_save_count) == 0) & (episode != 0):
                counter += 1
                torch.save(main_network.state_dict(), f'results/checkpoints/breakout_checkpoint' + str(
                    int(finished_episodes + counter * network_save_count)) + f'.pth')
                print(
                    '|| network at {}th episode saved ||'.format(int(finished_episodes + counter * network_save_count)))
            episode += 1

            rew_every_ep.append(episode_reward)
            step_every_ep.append(episode_step)
            qvalue_evert_ep.append(torch.mean(main_q_values).cpu().detach().numpy())
            state_stack = env.reset()
            step_list.append(episode_step)
            rew_buffer.append(episode_reward)

            if (((episode) % results_save_count) == 0) & (episode != 0):
                counter2 += 1
                c = finished_episodes + results_save_count * counter2
                np.savetxt('results/rews_and_steps/rewards_per_episode' + '{}'.format(c) + '.csv', rew_every_ep,
                           delimiter=',', fmt='%1.3f')
                np.savetxt('results/rews_and_steps/steps_per_episode' + '{}'.format(c) + '.csv', step_every_ep,
                           delimiter=',', fmt='%1.3f')
                np.savetxt('results/rews_and_steps/qvalues_per_episode' + '{}'.format(c) + '.csv', qvalue_evert_ep,
                           delimiter=',', fmt='%1.3f')
                print('|| rewards & steps up to {}th episode saved ||'.format(c))

            episode_reward = 0
            episode_step = 0

        # ------------------Start Gradient Step sample minibatch------------------
        transitions_minibatch = random.sample(replay_buffer, minibatch_size)


        state_stack_batch = np.asarray([t[0] for t in transitions_minibatch])
        action_batch = np.asarray([t[1] for t in transitions_minibatch])
        reward_batch = np.asarray([t[2] for t in transitions_minibatch])
        done_batch = np.asarray([t[3] for t in transitions_minibatch])
        next_state_stack_batch = np.asarray([t[4] for t in transitions_minibatch])

        # ------------------convert to tensor and forward all the data to device
        state_stack_batch = np.array(state_stack_batch) / 255.0
        next_state_stack_batch = np.array(next_state_stack_batch) / 255.0
        # tensor(32,4,84,84)
        state_stack_batch_dev = torch.from_numpy(state_stack_batch).float().to(device)
        action_batch_dev = torch.from_numpy(action_batch).long().to(device)
        reward_batch_dev = torch.from_numpy(reward_batch).float().to(device)
        next_state_stack_batch_dev = torch.from_numpy(next_state_stack_batch).float().to(device)
        done_batch_dev = torch.from_numpy(done_batch).float().to(device)


        # ------------------reshape action reward done batch minibatch_size*1
        action_batch_dev = torch.reshape(action_batch_dev,(minibatch_size,1))
        reward_batch_dev = torch.reshape(reward_batch_dev,(minibatch_size,1))
        done_batch_dev = torch.reshape(done_batch_dev,(minibatch_size,1))


        # ------------------compute targets
        with torch.no_grad():
          if use_DDQN == True:
            target_q_values = target_network(next_state_stack_batch_dev)
            #print(target_q_values)

            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
            #print(max_target_q_values.shape)
            #break
            #compute yj in the Algorithm 1
            target_y_batch = reward_batch_dev + gamma * (1 - done_batch_dev) * max_target_q_values
          else:
            main_q_values_next = main_network(next_state_stack_batch_dev)
            max_main_q_values_next = main_q_values_next.max(dim=1, keepdim=True)[0]
            target_y_batch = reward_batch_dev + gamma * (1 - done_batch_dev) * max_main_q_values_next

        # ------------------compute loss
        #calculate q values for main network
        main_q_values = main_network(state_stack_batch_dev)

        main_q_values_action = torch.gather(input = main_q_values,dim = 1, index = action_batch_dev)

        loss = nn.functional.smooth_l1_loss(main_q_values_action, target_y_batch)


        # ------------------Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ------------------update target network
        if use_DDQN == True:
          if t % target_network_update_freq == 0:
              target_network.load_state_dict(main_network.state_dict())

        if episode == (set_size * n_set ):
            # print("Finishing set {}.".format(counter + 1))
            finished_episodes += (episode)
            a = [finished_episodes]
            a.append(passed_steps + t + 1)
            a.append(current_epsilon)
            np.savetxt('results/steps.csv', a, delimiter=',', fmt='%1.3f')
            print(
                '\n=====================================\ndone {}th episode,totally {} episodes , finish and exit'.format(set_size * n_set,finished_episodes))
            break












