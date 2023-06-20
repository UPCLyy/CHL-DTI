# -*- coding: utf-8 -*-
import math
import torch
from GCN import *
import numpy as np
import torch.nn as nn
from loaddata_utils import *
import torch.nn.functional as F
from torch.autograd import Variable
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class GCN(nn.Module):
    def __init__(self, nfeat, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, 128)
        self.gc2 = GraphConvolution(128, 128)
        self.gc3 = GraphConvolution(128, 128)
        self.dropout = dropout

    def forward(self, x, adj):
        x = x.to(device)
        adj = adj.to(device)
        x11 = F.relu(self.gc1(x, adj), inplace=True)  # 试下tanh
        x1 = F.dropout(x11, self.dropout)
        x22 = self.gc2(x1, adj)
        x2 = F.dropout(x22, self.dropout)
        x3 = self.gc3(x2, adj)

        return x11, x22, x3


class DPP_GCN(nn.Module):
    def __init__(self, nfeat, dropout, alpha=0.8):
        super(DPP_GCN, self).__init__()
        self.gcn1 = GCN(nfeat, dropout)
        self.gcn2 = GCN(nfeat, dropout)

        self.tau = 0.5
        self.alpha = alpha

    def forward(self, x1, adj1, x2, adj2):
        z1, z2, z3 = self.gcn1(x1, adj1)
        z4, z5, z6 = self.gcn2(x2, adj2)

        return z1, z2, z3, z4, z5, z6


class MLP(nn.Module):
    def __init__(self, nfeat):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(nfeat, 64, bias=False).apply(init),
            nn.ELU(),
            nn.Linear(64, 16, bias=False).apply(init),
            nn.ELU(),
            nn.Linear(16, 2, bias=False),
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        output = self.MLP(x)
        return output


class CHLDTI(nn.Module):
    def __init__(self, num_protein, num_drug, num_hidden1, num_hidden2, num_out, dropout=0.5):  # 1512, 708, 512, 128
        super(CHLDTI, self).__init__()
        self.DPP_GCN = DPP_GCN(256, dropout)
        self.MLP = MLP(256)
        self.alpha = 0.8

        self.linear_d1 = nn.Linear(num_drug, num_hidden1)
        self.linear_d2 = nn.Linear(num_hidden1, num_hidden2)
        self.linear_d3 = nn.Linear(num_hidden2, num_out)

        self.linear_p1 = nn.Linear(num_protein, num_hidden1)
        self.linear_p2 = nn.Linear(num_hidden1, num_hidden2)
        self.linear_p3 = nn.Linear(num_hidden2, num_out)

        self.c = nn.Parameter(torch.Tensor((3), 1, 1))
        self.d = nn.Parameter(torch.Tensor((3), 1, 1))

        nn.init.constant_(self.c, 1)
        nn.init.constant_(self.d, 1)

    def SGC(self, feature, adj):

        adj = adj + (torch.eye(adj.shape[0]).cuda()) * 2
        deg = torch.sum(adj, dim=1)
        deg[deg <= 1e-10] = 1
        deg_inv = deg.pow(-0.5)
        deg_inv = deg_inv * torch.eye(adj.shape[0]).type(torch.FloatTensor).cuda()
        adj = torch.mm(deg_inv, adj)
        adj = torch.mm(adj, deg_inv).type(torch.FloatTensor)

        output = torch.mm(adj.cuda(), feature.cuda())

        return output

    def forward(self, node_num, drug_vec, protein_vec, protein_drug, drug_protein, HyGraph_Drug, HyGraph_protein, dateset_index, data, HyGraph_Structure_DPP, iftrain=True, d=None, p=None):

        drug1 = F.relu(self.linear_d1(drug_vec))
        drug2 = F.relu(self.linear_d2(drug1))
        drug3 = F.relu(self.linear_d3(drug2))

        protein1 = F.relu(self.linear_p1(protein_vec))
        protein2 = F.relu(self.linear_p2(protein1))
        protein3 = F.relu(self.linear_p3(protein2))
        feature = torch.cat((drug3, protein3), dim=0)

        HyGraph_ToGraph = torch.zeros((node_num[0] + node_num[1], node_num[0] + node_num[1])).cuda()
        HyGraph_ToGraph[:node_num[0], node_num[0]:] = drug_protein
        HyGraph_ToGraph[node_num[0]:, :node_num[0]] = protein_drug
        HyGraph_ToGraph[:node_num[0], :node_num[0]] = HyGraph_Drug
        HyGraph_ToGraph[node_num[0]:, node_num[0]:] = HyGraph_protein


        adj = HyGraph_ToGraph
        X_conv1 = self.SGC(feature, adj)
        X_conv2 = self.SGC(X_conv1, adj)
        X_conv3 = self.SGC(X_conv2, adj)
        X_conv4 = self.SGC(X_conv3, adj)
        X_conv5 = self.SGC(X_conv4, adj)
        conv_sum = self.SGC(X_conv5, adj)
        conv_sumx = self.SGC(conv_sum, adj)
        drug_feature_ht = conv_sumx[:drug_vec.shape[0]]
        protein_feature_ht = conv_sumx[drug_vec.shape[0]:]

        if iftrain:
            d, p = drug_feature_ht, protein_feature_ht

        feature_dp = torch.cat((d[data[:, :1]], p[data[:, 1:2]]), dim=2)

        feature_dp = feature_dp.squeeze(1)

        f_edge, f_feature = constructure_knngraph(data, d, p)

        z1, z2, z3, z4, z5, z6 = self.DPP_GCN(feature_dp, HyGraph_Structure_DPP, f_feature, f_edge)

        att3 = F.softmax(self.c, dim=0)
        feature_hg = torch.stack((z1, z2, z3), dim=0)
        feature_hg = torch.sum((att3 * feature_hg), dim=0)

        att4 = F.softmax(self.d, dim=0)
        feature_knn = torch.stack((z4, z5, z6), dim=0)
        feature_knn = torch.sum((att4 * feature_knn), dim=0)

        feature_stack = torch.cat((feature_hg, feature_knn), dim=1)
        pred = self.MLP(feature_stack[dateset_index])

        if iftrain:
            return pred, d, p
        return pred


def init(i):
    if isinstance(i, nn.Linear):
        torch.nn.init.xavier_uniform_(i.weight)
