import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Model(nn.Module):
    def __init__(self, drug_num, dis_num):
        super().__init__()
        self.encoder = Encoder(drug_num, dis_num)
        self.decoder = Decoder(drug_num, dis_num)

    def forward(self, data, train):

        drug, dis = self.encoder(data)

        if train:
            row, col = data.r_d_edge[0][data.train_index], data.r_d_edge[1][data.train_index]

        else:
            row, col = data.r_d_edge[0][data.test_index], data.r_d_edge[1][data.test_index]

        prob = self.decoder(drug[row, : ], dis[col, :])

        return prob
