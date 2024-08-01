import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np


class Encoder(nn.Module):       
    """
    使用'卷积' GraphSage 方法对节点进行编码
    """
    def __init__(self, features, feature_dim,
            embed_dim, adj_lists, aggregator,
            num_sample=10,
            base_model=None, gcn=False, cuda=False,
            feature_transform=False):
        super(Encoder, self).__init__()

        self.features = features  # 节点特征git
        self.feat_dim = feature_dim  # 特征维度
        self.adj_lists = adj_lists  # 邻接列表
        self.aggregator = aggregator  # 聚合器
        self.num_sample = num_sample  # 邻居采样数
        if base_model != None:  # 如果存在基础模型
            self.base_model = base_model  # 基础模型

        self.gcn = gcn    # 是否使用GCN
        self.embed_dim = embed_dim # 嵌入维度
        self.cuda = cuda # 是否使用GPU
        self.aggregator.cuda = cuda # 将聚合器设置为使用CUDA
        # 初始化权重
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform_(self.weight) # # 使用Xavier均匀分布初始化权重

    def forward(self, nodes):
        """
         为一批节点生成嵌入。
        nodes     -- 节点列表
        """
         # 获取邻居特征
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                self.num_sample)
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda()) # 获取节点特征
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1) # 合并特征
        else:
            combined = neigh_feats # 仅使用邻居特征
        combined = F.relu(self.weight.mm(combined.t())) # 计算嵌入
        return combined


"""
Set of modules for aggregating embeddings of neighbors. # 邻居嵌入的聚合器
"""


class MeanAggregator(nn.Module):
    """
    使用邻居嵌入的均值聚合节点的嵌入
    """

    def __init__(self, features, cuda=False, gcn=False):
        """
        初始化特定图的聚合器。
        features -- 将 LongTensor 的节点 ID 映射到 FloatTensor 的特征值的函数。
        cuda -- 是否使用 GPU
        gcn --- 是否执行连接 GraphSAGE 风格，或添加自环 GCN 风格
        """

        super(MeanAggregator, self).__init__() # 调用父类的构造函数

        self.features = features # 特征映射函数
        self.cuda = cuda # 是否使用CUDA
        self.gcn = gcn # 是否使用GCN

    def forward(self, nodes, to_neighs, num_sample=10):
        """
         nodes --- 一批节点列表
        to_neighs --- 每个节点的邻居集合列表
        num_sample --- 采样的邻居数量。如果为 None，则不进行采样。
        """
        # 本地指针函数（速度优化）
        _set = set #
        if not num_sample is None:
            _sample = random.sample # 随机采样函数
            # 采样邻居
            samp_neighs = [_set(_sample(to_neigh,
                                        num_sample,
                                        )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs # 不采样，直接使用邻居

        if self.gcn:
            # 如果使用GCN，添加自环
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs)) # 获取唯一节点列表
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)} # 创建节点索引映射
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes))) # 创建掩码矩阵
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh] # 列索引
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))] # 行索引
        mask[row_indices, column_indices] = 1 # 填充掩码矩阵
        if self.cuda:
            mask = mask.cuda() # 如果使用CUDA，将掩码移动到GPU

        num_neigh = mask.sum(1, keepdim=True) # 计算每个节点的邻居数量
        mask = mask.div(num_neigh) # 归一化掩码

        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda()) # 获取特征矩阵
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))

        to_feats = mask.mm(embed_matrix) # 计算聚合特征
        return to_feats


class SupervisedGraphSage(nn.Module): # 基于图的监督学习

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc # 编码器
        self.xent = nn.CrossEntropyLoss() # 交叉熵损失

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim)) # 类别权重
        init.xavier_uniform_(self.weight) # 初始化权重

    def forward(self, nodes):
        embeds = self.enc(nodes) # 获取嵌入
        scores = self.weight.mm(embeds) # 计算分数
        return scores.t(), embeds # 返回分数和嵌入

    def loss(self, nodes, labels):  # 计算损失
        labels = torch.LongTensor(labels) # 转换标签为LongTensor
        scores, _ = self.forward(nodes) # 计算分数
        return self.xent(scores, labels)   # 计算损失


def cls_model(features, adj_lists, fea_size, hidden): # 定义分类模型
    agg1 = MeanAggregator(features, cuda=True) # 创建第一个聚合器
    enc1 = Encoder(features, fea_size, hidden, adj_lists, agg1, gcn=True, cuda=False) # 创建第一个编码器
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)  # 创建第二个聚合器
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, hidden, adj_lists, agg2,
                       base_model=enc1, gcn=True, cuda=False) # 创建第二个编码器
    enc1.num_samples = 5 # 设置样本数量
    enc2.num_samples = 5
    graphsage = SupervisedGraphSage(7, enc2) # 创建监督GraphSAGE模型

    return graphsage


def cls_train(graphsage, train_x, train_y):
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.5) # 创建优化器
    # train_y = labels[np.array(train)].squeeze()
    for batch in range(100):
        batch_nodes = train_x[:128] # 获取批次节点
        batch_y = train_y[:128] # 获取批次标签
        # random.shuffle(train) # 打乱训练集
        c = list(zip(train_x, train_y)) # 打包训练数据
        random.shuffle(c)  # 随机打乱
        train_x, train_y = zip(*c) # 解包

        optimizer.zero_grad() # 清零梯度
        loss = graphsage.loss(batch_nodes, batch_y) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数

    return graphsage


class MLP(nn.Module):
    """
        标准的 in_dim-64-64-out_dim 前馈神经网络。
    """
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__() # 调用父类的构造函数
        """
            初始化网络并设置层。
        Parameters:
            in_dim - 输入维度
            out_dim - 输出维度
        Return:
            None
        """

        self.layer1 = nn.Linear(in_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        """
           在神经网络上运行前向传播。
        Parameters:
            obs - 输入观察
        Return:
            output - 前向传播的输出
        """
        # Convert observation to tensor if it's a numpy array # 如果输入是numpy数组，则转换为tensor
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.layer1(obs)) # 第一层激活
        activation1 = F.dropout(activation1, p=0.7, training=True) # 第一层dropout
        activation2 = F.relu(self.layer2(activation1)) # 第二层激活
        # activation2 = F.dropout(activation2, p=0.5, training=True)
        output = self.layer3(activation2) # 输出层

        return output



