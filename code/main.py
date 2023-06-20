import os
import torch
import warnings
import numpy as np
from model import *
from tqdm import tqdm
import torch.nn as nn
from loaddata_utils import *
import torch.nn.functional as F
from hypergraph_utils import generate_G_from_H
from hypergraph_utils import construct_H_with_KNN
from sklearn.metrics import roc_auc_score, f1_score
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity as cos
from Structe_DPP_HyperGraph import HyGraph_Matrix_DPP_Structure

warnings.filterwarnings("ignore")
seed = 47
args = setup(default_configure, seed)
in_size = 512
hidden_size = 256
out_size = 128
dropout = 0.5
lr = 0.0003
weight_decay = 1e-10
epochs = 800
reg_loss_co = 0.0002
fold = 0
dir = "../modelSave"

args['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"
for name in ["Luo"]:
    # for name in ["Luo","Es","GPCRs","ICs","NRs", "Zheng"]:
    # dataName: Luo Es GPCRs ICs NRs Zheng
    node_num, drug_protein, protein_drug, dtidata = load_dataset(name)
    dti_label = torch.tensor(dtidata[:, 2:3]).to(args['device'])

    hd = torch.randn((node_num[0], node_num[0]))
    hp = torch.randn((node_num[1], node_num[1]))

    drug_protein_eye = torch.cat((drug_protein, torch.eye(node_num[0])), dim=1)
    protein_drug_eye = torch.cat((protein_drug, torch.eye(node_num[1])), dim=1)

    HyGraph_Drug = generate_G_from_H(drug_protein_eye).to(args['device'])
    HyGraph_protein = generate_G_from_H(protein_drug_eye).to(args['device'])

    drug_protein = drug_protein.to(args['device'])
    protein_drug = protein_drug.to(args['device'])

    features_d = hd.to(args['device'])
    features_p = hp.to(args['device'])

    data = dtidata
    label = dti_label

    HyGraph_Structure_DPP = HyGraph_Matrix_DPP_Structure(data, node_num[0], node_num[1])
    HyGraph_Structure_DPP = HyGraph_Structure_DPP.to(args['device'])


    def main(tr, te, seed):
        all_acc = []
        all_roc = []
        all_pr = []
        all_f1 = []
        for i in range(len(tr)):
            f = open(f"{i}foldtrain.txt", "w", encoding="utf-8")
            train_index = tr[i]
            for train_index_one in train_index:
                f.write(f"{train_index_one}\n")
            test_index = te[i]
            f = open(f"{i}foldtest.txt", "w", encoding="utf-8")
            for train_index_one in test_index:
                f.write(f"{train_index_one}\n")
            #
            # if not os.path.isdir(f"{dir}"):
            #     os.makedirs(f"{dir}")

            model = CHLDTI(
                num_protein=node_num[1],
                num_drug=node_num[0],
                num_hidden1=512,
                num_hidden2=256,
                num_out=128,
            ).to(args['device'])
            # model.load_state_dict(torch.load(f"{dir}/net{i}.pth"))
            optim = torch.optim.Adam(lr=lr, weight_decay=weight_decay, params=model.parameters())
            best_acc = 0
            best_pr = 0
            best_roc = 0
            best_f1 = 0
            for epoch in tqdm(range(epochs)):
                loss, train_acc, task1_roc, acc, task1_roc1, task1_pr, task1_f1 = train(model, optim, train_index, test_index, epoch, i)
                if acc > best_acc:
                    best_acc = acc
                if task1_pr > best_pr:
                    best_pr = task1_pr
                if task1_roc1 > best_roc:
                    best_roc = task1_roc1
                if task1_f1 > best_f1:
                    best_f1 = task1_f1
            all_acc.append(best_acc)
            all_roc.append(best_roc)
            all_pr.append(best_pr)
            all_f1.append(best_f1)

        print(f"{name},aver Acc is:{sum(all_acc) / len(all_acc):.4f},  Aver roc is:{sum(all_roc) / len(all_roc):.4f}, "
              f"Aver Pr is:{sum(all_pr) / len(all_pr):.4f} ,Aver f1 is:{sum(all_f1) / len(all_f1):.4f}")


    def train(model, optim, train_index, test_index, epoch, fold):
        model.train()
        out, d, p = model(node_num, features_d, features_p, protein_drug, drug_protein, HyGraph_Drug, HyGraph_protein, train_index, data, HyGraph_Structure_DPP)

        train_acc = (out.argmax(dim=1) == label[train_index].reshape(-1).long()).sum(dtype=float) / torch.tensor(len(train_index), dtype=float)
        task1_roc = get_roc(out, label[train_index])
        reg = get_L2reg(model.parameters())
        loss = F.nll_loss(out, label[train_index].reshape(-1).long()) + reg_loss_co * reg

        optim.zero_grad()
        loss.backward()
        optim.step()
        # print(f"{epoch} epoch loss  {loss:.4f} train is acc  {train_acc:.4f}, task1 roc is {task1_roc:.4f},")
        te_acc, te_task1_roc1, te_task1_pr, te_task1_f1 = main_test(model, d, p, test_index, epoch, fold)

        return loss.item(), train_acc, task1_roc, te_acc, te_task1_roc1, te_task1_pr, te_task1_f1


    def main_test(model, d, p, test_index, epoch, fold):
        model.eval()
        out = model(node_num, features_d, features_p, protein_drug, drug_protein, HyGraph_Drug, HyGraph_protein, test_index, data, HyGraph_Structure_DPP, iftrain=False, d=d, p=p)

        acc1 = (out.argmax(dim=1) == label[test_index].reshape(-1).long()).sum(dtype=float) / torch.tensor(len(test_index), dtype=float)
        task_roc = get_roc(out, label[test_index])
        task_pr = get_pr(out, label[test_index])
        task_f1 = get_f1score(out, label[test_index])
        # if epoch == 799:
        #     f = open(f"{fold}out.txt","w",encoding="utf-8")
        #     for o in  (out.argmax(dim=1) == label[test_index].reshape(-1)):
        #         f.write(f"{o}\n")
        #     f.close()
        return acc1, task_roc, task_pr, task_f1


    train_indeces, test_indeces = get_cross(dtidata)
    main(train_indeces, test_indeces, seed)
