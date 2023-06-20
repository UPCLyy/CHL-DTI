import datetime
import errno
import numpy as np
import os
import time
import pickle
import random
import torch
from pprint import pprint
from scipy import sparse
from scipy import io as sio
import scipy.spatial.distance as dist
from sklearn.metrics import auc as auc3
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve



def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


default_configure = {
    'batch_size': 20
}


def setup(args, seed):
    args.update(default_configure)
    set_random_seed(seed)
    return args


def comp_jaccard(M):
    matV = np.mat(M)
    x = dist.pdist(matV, 'jaccard')

    k = np.eye(matV.shape[0])
    count = 0
    for i in range(k.shape[0]):
        for j in range(i + 1, k.shape[1]):
            k[i][j] = x[count]
            k[j][i] = x[count]
            count += 1
    return k


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def load_luo(network_path):
    """
    Loading data
    """
    drug_drug = np.loadtxt(network_path + 'mat_drug_drug.txt')
    drug_protein = np.loadtxt(network_path + 'mat_drug_protein.txt')
    protein_protein = np.loadtxt(network_path + 'mat_protein_protein.txt')

    num_drug = drug_drug.shape[0]
    num_protein = protein_protein.shape[0]
    node_num = [num_drug, num_protein]

    drug_protein = torch.Tensor(drug_protein)
    protein__drug = drug_protein.t()

    dti_o = np.loadtxt(network_path + 'mat_drug_protein.txt')
    train_positive_index = []
    whole_negative_index = []

    for i in range(np.shape(dti_o)[0]):
        for j in range(np.shape(dti_o)[1]):
            if int(dti_o[i][j]) == 1:
                train_positive_index.append([i, j])

            else:
                whole_negative_index.append([i, j])

    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size = 10 * len(train_positive_index),
                                             replace=False)

    data_set = np.zeros((len(negative_sample_index) + len(train_positive_index), 3), dtype=int)
    count = 0

    for i in train_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1


    for i in range(len(negative_sample_index)):
        data_set[count][0] = whole_negative_index[negative_sample_index[i]][0]
        data_set[count][1] = whole_negative_index[negative_sample_index[i]][1]
        data_set[count][2] = 0
        count += 1
    f = open(f"dti_index.txt", "w", encoding="utf-8")
    for i in data_set:
        f.write(f"{i[0]}\t{i[1]}\t{i[2]}\n")

    dateset = data_set
    f = open("dtiedge.txt", "w", encoding="utf-8")
    for i in range(dateset.shape[0]):
        for j in range(i, dateset.shape[0]):
            if dateset[i][0] == dateset[j][0] or dateset[i][1] == dateset[j][1]:
                f.write(f"{i}\t{j}\n")
    f.close()

    return node_num, drug_protein, protein__drug, dateset


def load_Yamanishi(network_path):

    drug_protein = np.loadtxt(network_path + 'd_p_i.txt')

    dti_o = np.loadtxt(network_path + 'd_p_i.txt')
    num_drug = drug_protein.shape[0]
    num_protein = drug_protein.shape[1]
    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(dti_o)[0]):
        for j in range(np.shape(dti_o)[1]):
            if int(dti_o[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(dti_o[i][j]) == 0:
                whole_negative_index.append([i, j])

    positive_shuffle_index = np.random.choice(np.arange(len(whole_positive_index)),
                                              size=1 * len(whole_positive_index), replace=False)
    whole_positive_index = np.array(whole_positive_index)
    whole_positive_index = whole_positive_index[positive_shuffle_index]

    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=1 * len(whole_positive_index), replace=False)

    data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
    count = 0
    for ind, i in enumerate(whole_positive_index):
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1

    for ind, i in enumerate(negative_sample_index):
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1
    f = open(f"dti_index.txt", "w", encoding="utf-8")
    for i in data_set:
        f.write(f"{i[0]}\t{i[1]}\t{i[2]}\n")
    f.close()

    dateset = data_set
    f = open("dtiedge.txt", "w", encoding="utf-8")
    for i in range(dateset.shape[0]):
        for j in range(i, dateset.shape[0]):
            if dateset[i][0] == dateset[j][0] or dateset[i][1] == dateset[j][1]:
                f.write(f"{i}\t{j}\n")
    f.close()
    node_num = [num_drug, num_protein]

    drug_protein = torch.Tensor(drug_protein)
    protein_drug = drug_protein.t()

    return node_num,  drug_protein, protein_drug, dateset


def load_graph(feature_edges, n):
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sparse.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(n, n),
                             dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = fadj + sparse.eye(fadj.shape[0])
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    return nfadj


def load_zheng(network_path):

    drug_protein = np.loadtxt(network_path + 'mat_drug_target_1.txt')
    num_drug = drug_protein.shape[0]
    num_protein = drug_protein.shape[1]

    dti_o = np.loadtxt(network_path + 'mat_drug_target_train.txt')
    dti_test = np.loadtxt(network_path + 'mat_drug_target_test.txt')
    train_positive_index = []
    test_positive_index = []
    whole_negative_index = []

    for i in range(np.shape(dti_o)[0]):
        for j in range(np.shape(dti_o)[1]):
            if int(dti_o[i][j]) == 1:
                train_positive_index.append([i, j])

            elif int(dti_test[i][j]) == 1:
                test_positive_index.append([i, j])
            else:
                whole_negative_index.append([i, j])

    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=len(test_positive_index) + len(train_positive_index),
                                             replace=False)
    data_set = np.zeros((len(negative_sample_index) + len(test_positive_index) + len(train_positive_index), 3),
                        dtype=int)
    count = 0
    train_index = []
    test_index = []
    for i in train_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        train_index.append(count)
        count += 1
    for i in test_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        test_index.append(count)
        count += 1

    for i in range(len(negative_sample_index)):
        data_set[count][0] = whole_negative_index[negative_sample_index[i]][0]
        data_set[count][1] = whole_negative_index[negative_sample_index[i]][1]
        data_set[count][2] = 0
        if i < 4000:
            train_index.append(count)
        else:
            test_index.append(count)
        count += 1
    f = open(f"dti_index.txt", "w", encoding="utf-8")
    for i in data_set:
        f.write(f"{i[0]}\t{i[1]}\t{i[2]}\n")

    dateset = data_set
    f = open("dtiedge.txt", "w", encoding="utf-8")
    for i in range(dateset.shape[0]):
        for j in range(i, dateset.shape[0]):
            if dateset[i][0] == dateset[j][0] or dateset[i][1] == dateset[j][1]:
                f.write(f"{i}\t{j}\n")

    f.close()
    node_num = [num_drug, num_protein]

    drug_protein = torch.Tensor(drug_protein)  # 药物靶标的关联矩阵，训练数据9/10
    protein_drug = drug_protein.t()

    return node_num, drug_protein, protein_drug, dateset


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def construct_fgraph(features, topk):
    # Cosine similarity
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    edge = []
    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                edge.append([i, vv])
    return edge


def generate_knn(data):
    topk = 3

    edge = construct_fgraph(data, topk)
    res = []

    for line in edge:
        start, end = line[0], line[1]
        if int(start) < int(end):
            res.append([start, end])
    return res


def constructure_knngraph(dateset, h1, h2, aug=False):
    feature = torch.cat((h1[dateset[:, :1]], h2[dateset[:, 1:2]]), dim=2)

    feature = feature.squeeze(1)
    fedge = np.array(generate_knn(feature.cpu().detach().numpy()))
    fedge = load_graph(np.array(fedge), dateset.shape[0])
    edg = torch.Tensor.to_dense(fedge)
    edge = edg.numpy()


    return fedge, feature


def get_set(data, split=5):
    """
    :param data: dataset and label
    :return:
    testset index and trainset index
    """
    set1 = []
    set2 = []
    skf = StratifiedKFold(n_splits=split, shuffle=True)
    for train_index, test_index in skf.split(data[:, :2], data[:, 2:3]):
        set1.append(train_index)
        set2.append(test_index)
    return set1[0].reshape(-1), set2[0].reshape(-1)


def get_cross(data, split=5):
    """
    :param data: dataset and label
    :return:
    testset index and trainset index
    """
    set1 = []
    set2 = []
    skf = StratifiedKFold(n_splits=split, shuffle=True)
    for train_index, test_index in skf.split(data[:, :2], data[:, 2:3]):
        set1.append(train_index)
        set2.append(test_index)
    return set1, set2


def get_roc(out, label):
    return np.nan_to_num(roc_auc_score(label.cpu(), out[:, 1:].cpu().detach().numpy()))


def get_pr(out, label):
    precision, recall, thresholds = precision_recall_curve(label.cpu(), out[:, 1:].cpu().detach().numpy())
    return auc3(recall, precision)


def get_f1score(out, label):
    return f1_score(label.cpu(), out.argmax(dim=1).cpu().detach().numpy())


def get_L2reg(parameters):
    reg = 0
    for param in parameters:
        reg += 0.5 * (param ** 2).sum()
    return reg


def load_dataset(dateName):
    if dateName == "Luo":
        return load_luo("../data/Luo/")
    elif dateName == "Zheng":
        return load_zheng("../data/Zheng/")
    else:
        return load_Yamanishi(f"../data/Yamanishi/{dateName}/")
