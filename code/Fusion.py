import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np

class SelfAttention(nn.Module):

    def __init__(self, hidden_size, weights_dropout):
        super(SelfAttention, self).__init__()
        self.att_size = hidden_size
        self.scale = self.att_size ** -0.5
        
        self.linear_k = nn.Linear(hidden_size, self.att_size)
        self.linear_q = nn.Linear(hidden_size, self.att_size)
        self.linear_v = nn.Linear(hidden_size, self.att_size)
        
        self.weights_dropout = nn.Dropout(weights_dropout)


    def forward(self, q, k, v, bias=None):

        q = self.linear_q(q)

        k = self.linear_k(k)

        v = self.linear_v(v)

        k = k.transpose(0, 1) 

        q = q * self.scale

        x = torch.matmul(q, k)
        
        x = torch.softmax(x, dim=-1)
        
        if bias is not None:
            x = x + bias
       
        x = self.weights_dropout(x).to(torch.float32)
        
        x = x.matmul(v)

        x = x.view(-1, self.att_size)

        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size, weights_dropout):
        super(Encoder, self).__init__()
        self.self_attention = SelfAttention(hidden_size, weights_dropout)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, bias=None):

        y = self.norm(x)

        SA = self.self_attention(y, y, y, bias)

        return x + SA

class Fusion(nn.Module):
    def __init__(self,
                 hidden_dim,
                 weights_dropout=0.1,
                ):
        super().__init__()
        self.encoder = Encoder(hidden_dim, weights_dropout)
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, sim, x):

        expanded_sim, x = sim, x.to(torch.float32)

        bias = expanded_sim.clone()

        output = self.encoder(x, bias)

        output = self.final_ln(output)

        return output




