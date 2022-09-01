import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, drug_num, dis_num):
        super(Decoder, self).__init__()
        input_dim = drug_num + dis_num
        self.mlp_1 = nn.Sequential(nn.Linear(int(input_dim*2), int(input_dim)),
                                   nn.ReLU())
        self.mlp_2 = nn.Sequential(nn.Linear(int(input_dim), int(input_dim // 2)),
                                   nn.ReLU())
        self.mlp_3 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(int(input_dim // 2), 1),
                                   nn.Sigmoid())

    def forward(self, drug_feature, disease_feature):
        pair_feature = torch.cat([drug_feature, disease_feature], dim=1)

        embedding_1 = self.mlp_1(pair_feature)

        embedding_2 = self.mlp_2(embedding_1)

        outputs = self.mlp_3(embedding_2)

        return outputs