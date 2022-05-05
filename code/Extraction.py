import torch
import torch.nn as nn


class MultiViewSelfAttention(nn.Module):

    def __init__(self, hidden_size, weights_dropout, view_size):
        super(MultiViewSelfAttention, self).__init__()

        self.view_size = view_size
        self.att_size = hidden_size // view_size
        self.scale = self.att_size ** -0.5
        self.linear_q = nn.Linear(hidden_size, self.att_size)
        self.linear_k = nn.Linear(hidden_size, self.att_size)
        self.linear_v = nn.Linear(hidden_size, self.att_size)
        self.weights_dropout = nn.Dropout(weights_dropout)
        self.output_layer = nn.Linear(view_size * self.att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        d_v = self.att_size

        q = self.linear_q(q)

        k = self.linear_k(k)

        v = self.linear_v(v)

        k = k.transpose(1, 2)

        q = q * self.scale

        x = torch.matmul(q, k)

        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=2)

        x = self.weights_dropout(x).to(torch.float32)

        x = x.matmul(v)

        x = x.transpose(0, 1).contiguous()

        x = x.view(-1, d_v * self.view_size)

        x = self.output_layer(x)

        return x


class Encoder(nn.Module):

    def __init__(self, hidden_size, output_dropout, weights_dropout, view_size):
        super(Encoder, self).__init__()
        self.self_norm = nn.LayerNorm(hidden_size)
        self.MVSA = MultiViewSelfAttention(hidden_size, weights_dropout, view_size)
        self.output_dropout = nn.Dropout(output_dropout)


    def forward(self, x, bias=None):

        y = self.self_norm(x)

        _MVSA = self.MVSA(y, y, y, bias)

        _MVSA = self.output_dropout(_MVSA)

        out = torch.mean(x, dim=0) + _MVSA

        return out



class Extraction(nn.Module):

    def __init__(self,
                 hidden_dim,
                 view_size=4,
                 input_dropout=0.1,
                 weights_dropout=0.1,
                 output_dropout=0.1,
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.view_size = view_size
        self.a = 16  # α of feature fusion module
        self.input_dropout = nn.Dropout(input_dropout)
        self.encoder = Encoder(hidden_dim, output_dropout, weights_dropout, view_size)
        self.final_ln = nn.LayerNorm(hidden_dim)

    def forward(self, sim, x):
        expanded_sim, x = sim*self.a, x.to(torch.float32)

        bias = expanded_sim.clone()

        x = self.input_dropout(x)

        output = self.encoder(x, bias)

        output = self.final_ln(output)

        return output

