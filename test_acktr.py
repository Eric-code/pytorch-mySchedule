import copy
import glob
import os
import time
from collections import deque

from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
from arguments import get_args
from envs import make_vec_envs
from model import Policy
from storage import RolloutStorage
from utils import get_vec_normalize
from visualize import visdom_plot
from main_ppo import Net
import random


num_updates = int(40000000) // 8000 // 1
num_processes = 1
num_steps = 8000
torch.manual_seed(1)

log_dir = "log"
try:
    os.makedirs("log")
except OSError:
    files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    torch.set_num_threads(1)
    device = torch.device("cpu")

    # if args.vis:
    #     from visdom import Visdom
    #     viz = Visdom(port=args.port)
    #     win = None

    # envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                     args.gamma, args.log_dir, args.add_timestep, device, False)

    observation_space = Box(low=0, high=10000, shape=(26,), dtype=np.float32)  # Box(84,84,4)
    action_space = Discrete(7)  # Discrete(4)

    actor_critic = Policy(observation_space.shape, action_space, base_kwargs={'recurrent': None})
    actor_critic.to(device)

    # if args.algo == 'a2c':
    #     agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
    #                            args.entropy_coef, lr=args.lr,
    #                            eps=args.eps, alpha=args.alpha,
    #                            max_grad_norm=args.max_grad_norm)
    # elif args.algo == 'ppo':
    #     agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
    #                      args.value_loss_coef, args.entropy_coef, lr=args.lr,
    #                            eps=args.eps,
    #                            max_grad_norm=args.max_grad_norm)
    # elif args.algo == 'acktr':
    agent = algo.A2C_ACKTR(actor_critic, value_loss_coef=0.1,
                           entropy_coef=0.01, acktr=True)

    rollouts = RolloutStorage(8000, 1, observation_space.shape, action_space, actor_critic.recurrent_hidden_state_size)

    obs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    rollouts.obs[0].copy_(torch.Tensor(obs))
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    f = open('poktr_rtmdp_20_2.txt', 'w')
    f.write("\noriginal loss(schedule 6 packets):")
    start = time.time()
    for j in range(num_updates):  # num_updates
        net = Net()
        node_list, path_list = net.read_graph(net.node_list, net.path_list)
        startnode = node_list[0]  # 起始节点
        net.get_data(startnode)
        count = 0
        remove_count = 0  # 记录丢弃的数据包的值
        end_time = startnode.messages[0].end_time
        pre_action_item = random.randint(0, 6)
        pre_action_item_oh = convert_one_hot(pre_action_item, 7)
        s = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, end_time, pre_action_item_oh]
        states = [[0], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]  # 用来存储所有节点状态
        ep_r = 0
        ep_acc_r = 0
        obs[:] = s
        reward_ten = torch.Tensor(1, 1)

        pre_value = torch.FloatTensor([[0.1]])
        pre_action = torch.Tensor([[random.randint(0, 6)]])
        pre_action_log_prob = torch.FloatTensor([[-1.]])
        pre_recurrent_hidden_states = torch.FloatTensor([[0.]])
        pre_masks = torch.FloatTensor([[0.]])
        for step in range(8000):
            # Sample actions
            count += 1
            old_action_log_prob = torch.Tensor([[0]])
            # print(rollouts, rollouts.obs[step], rollouts.recurrent_hidden_states[step], rollouts.masks[step])
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
                action_item = action.item()  # 将Tensor类型的数据转化为Int型
                action_item_oh = convert_one_hot(action_item, 7)

            # Obser reward and next obs
            obs, reward, done, states, remove_count, acc_r, su_packets = net.schedule(pre_action_item, count, states, node_list, path_list,
                                                                            remove_count)

            ep_r += reward
            ep_acc_r += acc_r
            reward_ten[[0]] = reward
            # for info in infos:
            #     if 'episode' in info.keys():
            #         episode_rewards.append(info['episode']['r'])
            obs.extend(pre_action_item_oh)
            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done else [1.0]])
            # print((obs), recurrent_hidden_states, torch.Tensor(action), type(action_log_prob), type(value), type(reward), type(masks))
            rollouts.insert(torch.Tensor(obs), recurrent_hidden_states, action, action_log_prob, value, reward_ten, masks)
            # rollouts.insert(torch.Tensor(obs), pre_recurrent_hidden_states, pre_action, pre_action_log_prob, pre_value, reward_ten, pre_masks)

            pre_action = action
            pre_action_item = action_item
            pre_action_log_prob = action_log_prob
            pre_recurrent_hidden_states = recurrent_hidden_states
            pre_value = value
            pre_action_item_oh = convert_one_hot(pre_action_item, 7)

        f.write("\ntime:"+str(time.strftime('%H:%M:%S', time.localtime(time.time())))+"|"+str(j)+"|ep_r:"+str(ep_r)+"|pakcets:"+str(su_packets)+"|remove:"+str(remove_count)+"|ep_acc_r:"+str(ep_acc_r / 8000))
        f.flush()
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, False, 0.99, 0.95)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        print("time:", time.strftime('%H:%M:%S', time.localtime(time.time())), "|", j, "|ep_r:", ep_r, "|pakcets:",
              su_packets, "|remove:", remove_count, "|ep_acc_r:", ep_acc_r / 8000, "|value_loss:", value_loss,
              "|action_loss:", action_loss, "|entropy:", dist_entropy)
        rollouts.after_update()


def convert_one_hot(x, num):
    a = [0 for _ in range(num)]
    a[x] = 1
    return a


if __name__ == "__main__":
    main()