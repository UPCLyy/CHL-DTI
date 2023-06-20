import os
import torch
import numpy as np
from hypergraph_utils import generate_G_from_H


def HyGraph_Matrix_DPP_Structure(dateset, num_drug, num_protein):
    f = open("HyGraph_Structure_DPP_drug.txt", "w", encoding="utf-8")
    Graph_1 = np.zeros((dateset.shape[0], num_drug))
    for k in range(0, num_drug):
        a = 0
        for i in range(dateset.shape[0]):
            if dateset[i][0] == k:
                if a == 0:
                    j = i
                    a = a + 1
                else:
                    a = a + 1
                if a == 2:
                    f.write(f"{j}\t{i}\t")
                    Graph_1[j][k] = 1
                    Graph_1[i][k] = 1
                elif a > 2:
                    f.write(f"{i}\t")
                    Graph_1[i][k] = 1
            if i == dateset.shape[0] - 1:
                f.write(f"\n")
    f.close()
    # print(Graph_1.shape)

    f = open("HyGraph_Structure_DPP_protein.txt", "w", encoding="utf-8")
    Graph_2 = np.zeros((dateset.shape[0], num_protein))
    for k in range(0, num_protein):
        a = 0
        for i in range(dateset.shape[0]):
            if dateset[i][1] == k:
                if a == 0:
                    j = i
                    a = a + 1
                else:
                    a = a + 1
                if a == 2:
                    f.write(f"{j}\t{i}\t")
                    Graph_2[j][k] = 1
                    Graph_2[i][k] = 1
                elif a > 2:
                    f.write(f"{i}\t")
                    Graph_2[i][k] = 1
            if i == dateset.shape[0] - 1:
                f.write(f"\n")
    f.close()
    HyGraph_Structure_DPP = np.concatenate([Graph_1, Graph_2], axis=1)
    HyGraph_Structure_DPP = HyGraph_Structure_DPP[:, HyGraph_Structure_DPP.sum(axis=0) != 0]
    HyGraph_Structure_DPP = generate_G_from_H(HyGraph_Structure_DPP)

    return HyGraph_Structure_DPP
