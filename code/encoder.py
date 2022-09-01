import torch
import torch.nn as nn
from Extraction import Extraction
from Fusion import Fusion

class Encoder(nn.Module):
    def __init__(self, drug_num, dis_num):
        super().__init__()
        self.drug_num = drug_num
        self.drug_extraction = Extraction(hidden_dim=drug_num + dis_num, view_size=4)
        self.dis_extraction = Extraction(hidden_dim=drug_num + dis_num, view_size=2)
        self.interaction = Fusion(hidden_dim=drug_num+dis_num)

    def forward(self, data):
        drug = self.drug_extraction(data.drug_sim, data.drug_feature)

        dis = self.dis_extraction(data.dis_sim, data.dis_feature)

        drug_dis_f = torch.cat((drug, dis), dim=0)
        
        output = self.interaction(data.drug_disease_sim, drug_dis_f)

        drug, dis = output[:self.drug_num, :], output[self.drug_num:, :]

        return drug, dis