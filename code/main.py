import torch
import random
import numpy as np
from model import Model
from dataProcess import ProcessAndData
from evaluation_metrics import auroc, auprc

seed = 999
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

class Data():
    def __init__(self, i=0):
        self.drug_sim = torch.load(processAndData.drug_sim_dir).double().to(device)
        self.dis_sim = torch.load(processAndData.dis_sim_dir).double().to(device)
        self.drug_feature = torch.cat((self.drug_sim, torch.zeros((4, self.drug_sim.shape[1], self.dis_sim.shape[2]),
                                                                  dtype=torch.float64).to(device)), dim=2).double().to(device)
        self.dis_feature = torch.cat((torch.zeros((2, self.dis_sim.shape[1], self.drug_sim.shape[2]),
                                                  dtype=torch.float64).to(device), self.dis_sim), dim=2).double().to(device)
        self.test_index = torch.load(processAndData.test_index_dir + "_{}.pt".format(i)).to(device)
        self.train_index = torch.load(processAndData.train_index_dir + "_{}.pt".format(i)).to(device)
        self.r_d_edge = torch.load(processAndData.r_d_edge_dir).to(device)
        self.labels = torch.load(processAndData.labels_dir).to(device)
        self.drug_disease_sim = torch.load(processAndData.drug_disease_sim_dir + "_{}.pt".format(i)).to(device)

epoch_num = 1000

def train_val_test(phase, epoch, model, data):
    logging_info = {}
    logging_info.update({'%s Epoch' % phase: epoch})
    if phase == 'Train':
        model.train()
        optimizer.zero_grad()
        prob = model(data, train=True)
        labels = data.labels[data.train_index].float()
    else:
        model.eval()
        prob = model(data, train=False)
        labels = data.labels[data.test_index].float()


    weight = (pos_weight * labels + 1 - labels)

    loss = torch.nn.functional.binary_cross_entropy(input=prob.reshape(-1), target=labels, weight=weight)

    if phase == 'Train':
        loss.backward()
        optimizer.step()

    now_roc = auroc(prob, labels)
    now_prc = auprc(prob, labels)


    logging_info.update({'loss': '%.04f' % loss.data.item()})
    logging_info.update({'auc': '%.04f' % now_roc})
    logging_info.update({'auprc': '%.04f' % now_prc})
    print(logging_info)
    return logging_info



if __name__ == '__main__':
    KFold_NUM = 10
    processAndData = ProcessAndData('Fdataset', seed=seed, KFold_NUM=KFold_NUM)

    pos_weight = processAndData.pos_weight
    drug_num = processAndData.drug_num
    dis_num = processAndData.dis_num

    for i in range(0, KFold_NUM):
        data = Data(i)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        model = Model(drug_num, dis_num).to(device)

        lr = 1e-3
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1 * lr, max_lr=lr,
                                                      gamma=0.995, mode="exp_range", step_size_up=15,
                                                      cycle_momentum=False)
        for epoch in range(1, epoch_num + 1):
            train_val_test('Train', epoch, model, data)
            train_val_test('Test ', epoch, model, data)






