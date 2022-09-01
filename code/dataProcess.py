import scipy.io as scio
import os.path
import pandas as pd
import torch
from sklearn.model_selection import KFold
import random
import numpy as np


class ProcessAndData():

    def __init__(self, datasetName, seed=999, KFold_NUM=5):
        random.seed(seed)
        assert datasetName in ['Fdataset', 'Cdataset']
        root_dir = "./data"
        # root_dir = "/root/Finally/data"

        tensors_dir = os.path.join(root_dir, datasetName, "tensors")
        self.drug_sim_dir = os.path.join(tensors_dir, "drug_sim.pt")
        self.dis_sim_dir = os.path.join(tensors_dir, "dis_sim.pt")
        self.drug_sim_origin_dir = os.path.join(tensors_dir, "drug_sim_origin.pt")
        self.dis_sim_origin_dir = os.path.join(tensors_dir, "dis_sim_origin.pt")
        
        self.r_d_edge_dir = os.path.join(tensors_dir, "r_d_edge.pt")
        self.labels_dir = os.path.join(tensors_dir, "labels.pt")
        self.train_index_dir = os.path.join(tensors_dir, "train_index")
        self.test_index_dir = os.path.join(tensors_dir, "test_index")
        self.drug_disease_sim_dir = os.path.join(tensors_dir, "drug_disease_sim")
        os.makedirs(name=tensors_dir, exist_ok=True)

        if datasetName in ['Fdataset', 'Cdataset']:
            dataset = scio.loadmat(os.path.join(root_dir, datasetName + ".mat"))
            drug_ChemS = dataset["drug_ChemS"]
            drug_DDIS = dataset["drug_DDIS"]
            drug_SideS = dataset["drug_SideS"]
            drug_TargetS = dataset["drug_TargetS"]

            disease_PhS = dataset["disease_PhS"]
            disease_DoS = dataset["disease_DoS"]
            
            didr = dataset["didr"].T  
            drug_num = didr.shape[0]
            disease_num = didr.shape[1]
            
            kth = 10
            drug_ChemS2 = np.zeros((drug_num, drug_num))
            neighbor = np.argpartition(-drug_ChemS, kth=kth, axis=1)[:, :kth]
            for i, value in enumerate(neighbor):
                for j in value:
                    drug_ChemS2[i][j] = drug_ChemS[i][j]
                    
            drug_SideS2 = np.zeros((drug_num, drug_num))
            neighbor = np.argpartition(-drug_SideS, kth=kth, axis=1)[:, :kth]
            for i, value in enumerate(neighbor):
                for j in value:
                    drug_SideS2[i][j] = drug_SideS[i][j]
                    
            drug_DDIS2 = np.zeros((drug_num, drug_num))
            neighbor = np.argpartition(-drug_DDIS, kth=kth, axis=1)[:, :kth]
            for i, value in enumerate(neighbor):
                for j in value:
                    drug_DDIS2[i][j] = drug_DDIS[i][j]
      
            drug_TargetS2 = np.zeros((drug_num, drug_num))
            neighbor = np.argpartition(-drug_TargetS, kth=kth, axis=1)[:, :kth]
            for i, value in enumerate(neighbor):
                for j in value:
                    drug_TargetS2[i][j] = drug_TargetS[i][j]
                    
            disease_PhS2 = np.zeros((disease_num, disease_num))
            neighbor = np.argpartition(-disease_PhS, kth=kth, axis=1)[:, :kth]
            for i, value in enumerate(neighbor):
                for j in value:
                    disease_PhS2[i][j] = disease_PhS[i][j]
            
            disease_DoS2 = np.zeros((disease_num, disease_num))
            neighbor = np.argpartition(-disease_DoS, kth=kth, axis=1)[:, :kth]
            for i, value in enumerate(neighbor):
                for j in value:
                    disease_DoS2[i][j] = disease_DoS[i][j]
                    
            drug_sim = torch.from_numpy(np.array([drug_ChemS2, drug_SideS2, drug_DDIS2, drug_TargetS2]))
            disease_sim = torch.from_numpy(np.array([disease_PhS2, disease_DoS2]))
            drug_sim_origin = torch.from_numpy(np.array([drug_ChemS, drug_SideS, drug_DDIS, drug_TargetS]))
            disease_sim_origin = torch.from_numpy(np.array([disease_PhS, disease_DoS]))
            
        drug_num = didr.shape[0]
        disease_num = didr.shape[1]
        self.drug_num = drug_num
        self.dis_num = disease_num

        pos_row, _ = np.nonzero(didr)
        pos_weight = len(np.where(didr == 0)[0]) / len(pos_row)
        self.pos_weight = pos_weight

        torch.save(drug_sim, self.drug_sim_dir)
        torch.save(disease_sim, self.dis_sim_dir)
        torch.save(drug_sim_origin, self.drug_sim_origin_dir)
        torch.save(disease_sim_origin, self.dis_sim_origin_dir)
        
        pos_row, _ = np.nonzero(didr)
        r_d_pos_edge = torch.from_numpy(np.array(np.nonzero(didr)))
        r_d_neg_edge = torch.from_numpy(np.array(np.where(didr == 0)))  # sxd

        labels = torch.cat((torch.ones((r_d_pos_edge.size(1))), torch.zeros((r_d_neg_edge.size(1)))), dim=0)
        torch.save(labels, self.labels_dir)

        r_d_edge = torch.cat((r_d_pos_edge.long(), r_d_neg_edge.long()), dim=1)
        torch.save(r_d_edge, self.r_d_edge_dir)


        KFload_split = KFold(n_splits=KFold_NUM, shuffle=True, random_state=seed)
        pos_index_gen = KFload_split.split(r_d_pos_edge.t())  # pos_train_index, pos_test_index
        neg_index_gen = KFload_split.split(r_d_neg_edge.t())  # neg_train_index, nef_test_index

        for split_i in range(KFold_NUM):
            pos_train_index, pos_test_index = pos_index_gen.__next__()
            neg_train_index, neg_test_index = neg_index_gen.__next__()

            pos_train_index_copy = list(pos_train_index)
            pos_train_index, pos_test_index = list(pos_train_index), list(pos_test_index)

            neg_train_index, neg_test_index = list(neg_train_index), list(neg_test_index)

            pos_train_index.extend(list(np.array(neg_train_index) + r_d_pos_edge.size(1)))
            train_index = pos_train_index
            train_index = torch.from_numpy(np.array(train_index)).long()
            torch.save(train_index, self.train_index_dir + "_{}.pt".format(split_i))

            pos_test_index.extend(list(np.array(neg_test_index) + r_d_pos_edge.size(1)))
            test_index = pos_test_index
            test_index = torch.from_numpy(np.array(test_index)).long()
            torch.save(test_index, self.test_index_dir + "_{}.pt".format(split_i))

            drug_disease_sim = torch.zeros((drug_num + disease_num, drug_num + disease_num))
            for i in pos_train_index_copy:
                row, col = r_d_edge[:, i]
                drug_disease_sim[row][col + drug_num] = 1
                drug_disease_sim[drug_num + col][row] = 1
            for i in range((drug_num + disease_num)):
                drug_disease_sim[i][i] = 1
            torch.save(drug_disease_sim, self.drug_disease_sim_dir + "_{}.pt".format(split_i))



if __name__ == '__main__':
    ProcessAndData("Fdataset")
