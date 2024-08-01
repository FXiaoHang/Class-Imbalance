import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score, roc_auc_score
from collections import defaultdict
from sklearn import manifold
import matplotlib.pyplot as plt

from model import Encoder, MeanAggregator, SupervisedGraphSage, MLP

from eval_policy import eval_policy
import sys
from env import Env
from arguments import get_args

import scipy.sparse as sp


def data_spilit(labels, num_cls):
    """
    将数据集划分为训练集、验证集和测试集，并确保每个类别的样本数量平衡。

    参数:
        labels - 数据集的标签
        num_cls - 类别数量

    返回:
        train_set - 训练集样本索引
        train_y - 训练集标签
        val - 验证集样本索引
        val_y - 验证集标签
        test - 测试集样本索引
        test_y - 测试集标签
        unlable - 未标记样本索引
    """ 
    num_nodes = labels.shape[0]  # 获取数据集中样本的数量
    rand_indices = np.random.permutation(num_nodes) # 生成一个随机排列的索引数组，用于随机划分数据集
    test = rand_indices[:1500]
    val = rand_indices[1500:2000]
    train_set = list(rand_indices[2000:])
    # 将随机排列的索引数组划分为测试集，验证集，训练集
    # train = random.sample(train, 100)

    tr_ratio = [] # 存储训练集样本索引
    count_tr = np.zeros(num_cls) # 每个类别已选择的样本数量
    # count_tr_ratio = np.array([20, 6, 20, 6, 20, 20, 6])
    count_tr_ratio = np.array([20, 20, 20, 20, 6, 6, 6])  # 每个类别的样本数量限制


    for i in train_set:
        for j in range(num_cls):
            if labels[i] == j:
                count_tr[j] += 1
                break
        # if count_tr[labels[i]] <= 20:
        #     tr_balanced.append(i)
        # count_tr[labels[i]] += 1
        if count_tr[labels[i]] <= count_tr_ratio[labels[i]]:
            tr_ratio.append(i)
    train_set = tr_ratio  # 根据每个类别的样本数量限制，筛选出平衡的训练集样本索引

    test_balanced = [] # 存储平衡的测试集样本索引
    count_test = np.zeros(num_cls)  # 每个类别已选择的样本数量
    for i in test:
        for j in range(num_cls):
            if labels[i] == j:
                count_test[j] += 1
                break
        if count_test[labels[i]] <= 100:
            test_balanced.append(i)
    test = test_balanced  # 根据每个类别的样本数量限制，筛选出平衡的测试集样本索引

    val_bal = [] # 存储平衡的验证集样本索引
    count_val = np.zeros(num_cls)
    for i in val:
        for j in range(num_cls):
            if labels[i] == j:
                count_val[j] += 1
                break
        if count_val[labels[i]] <= 30:
            val_bal.append(i)
    val = val_bal # 根据每个类别的样本数量限制，筛选出平衡的验证集样本索引

    index = np.arange(0, num_nodes)  # 所有样本的索引
    unlable = np.setdiff1d(index, train_set)  # 未标记样本的索引
    unlable = np.setdiff1d(unlable, val)
    unlable = np.setdiff1d(unlable, test)
    # train_x = train
    train_y = []  # 训练集标签
    for i in train_set:
        train_y.append(int(labels[i]))
    # print(train_y)
    val_y = [] # 验证集标签
    for i in val:
        val_y.append(int(labels[i]))
    test_y = [] # 测试集标签
    for i in test:
        test_y.append(int(labels[i]))

    return train_set, train_y, val, val_y, test, test_y, unlable



def load_cora():
    """
    加载Cora数据集，包括特征数据、标签和邻接列表。

    返回:
        feat_data - 特征数据
        labels - 标签
        adj_lists - 邻接列表
    """
    num_nodes = 2708 # 节点数量
    num_feats = 1433  # 特征数量
    feat_data = np.zeros((num_nodes, num_feats)) # 创建一个全零矩阵作为特征数据
    labels = np.empty((num_nodes,1), dtype=np.int64) # 创建一个空的标签数组
    node_map = {} # 创建一个空的节点映射字典
    label_map = {} # 创建一个空的标签映射字典

    with open("cora/cora.content") as fp: # 打开cora.content文件
        for i, line in enumerate(fp): # 遍历文件中的每一行
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]  # 获取论文1的索引
            paper2 = node_map[info[1]]  # 获取论文2的索引
            adj_lists[paper1].add(paper2)  # 将论文2添加到论文1的邻接列表中
            adj_lists[paper2].add(paper1)  # 将论文1添加到论文2的邻接列表中
    return feat_data, labels, adj_lists  # 返回特征数据、标签和邻接列表


def run_cora():
    """
    运行Cora数据集的预处理，包括数据划分和特征转换。

    返回:
        train_x - 训练集样本索引
        train_y - 训练集标签
        val_x - 验证集样本索引
        val_y - 验证集标签
        test_x - 测试集样本索引
        test_y - 测试集标签
        unlable - 未标记样本索引
        features - 特征数据
        adj_lists - 邻接列表
        labels - 标签
    """
    np.random.seed(1) 
    random.seed(1)
    num_cls = 7
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()

    train_x, train_y, val_x, val_y, test_x, test_y, unlable = data_spilit(labels, num_cls)

    train_x = torch.LongTensor(train_x)
    train_y = torch.LongTensor(train_y)
    val_x = torch.LongTensor(val_x)
    val_y = torch.LongTensor(val_y)
    test_x = torch.LongTensor(test_x)
    test_y = torch.LongTensor(test_y)
    unlable = torch.LongTensor(unlable)

    return train_x, train_y, val_x, val_y, test_x, test_y, unlable, features, adj_lists, labels


def test_model(env, actor_model, test_x, test_y, features, adj_lists, labels):
    """
    测试模型。

    参数:
        env - 测试策略的环境
        actor_model - 要加载的actor模型
        test_x - 测试集样本索引
        test_y - 测试集标签
        features - 特征数据
        adj_lists - 邻接列表
        labels - 标签

    返回:
        None
    """
    print(f"Testing {actor_model}", flush=True)

    # If the actor model is not specified, then exit
    if actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    obs_dim = 384  # env.observation_space.shape[0]# 观察空间的维度
    act_dim = 2  # env.action_space.shape[0] # 动作空间的维度

    # Build our policy the same way we build our actor model in PPO
    policy = MLP(obs_dim, act_dim)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))

    eval_policy(policy=policy, env=env, test_x=test_x, test_y=test_y, features=features, adj_lists=adj_lists, labels=labels)


def main(args):
    """
        The main function to run.

        Parameters:
            args - the arguments parsed from command line

        Return:
            None
    """

    train_x, train_y, val_x, val_y, test_x, test_y, unlable, features, adj_lists, labels = run_cora()

    env = Env(train_x, train_y, unlable, val_x, val_y, features, adj_lists, test_x, test_y)

    if args.mode == 'test':
        test_model(env=env, actor_model='ppo_actor.pth', test_x=test_x, test_y=test_y, features=features, adj_lists=adj_lists, labels=labels)


if __name__ == "__main__":
    args = get_args()  # Parse arguments from command line
    args.mode = "test"
    main(args)


