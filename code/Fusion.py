import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, hidden_size, weights_dropout):
        super(SelfAttention, self).__init__()
        self.att_size = hidden_size
        self.scale = self.att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.weights_dropout = nn.Dropout(weights_dropout)

        self.output_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v, bias=None):

        q = self.linear_q(q).view(-1, 1, self.att_size)

        k = self.linear_k(k).view( -1, 1, self.att_size)

        v = self.linear_v(v).view(-1, 1, self.att_size)

        q = q.transpose(0, 1)

        v = v.transpose(0, 1)

        k = k.transpose(0, 1) .transpose(1, 2)

        q = q * self.scale

        x = torch.matmul(q, k)

        x = torch.softmax(x, dim=2)
        
        if bias is not None:
            x = x + bias
        
        x = self.weights_dropout(x).to(torch.float32)

        x = x.matmul(v)

        x = x.view(-1, self.att_size)

        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size, output_dropout, weights_dropout):
        super(Encoder, self).__init__()
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attention = SelfAttention(hidden_size, weights_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, x, bias=None):

        y = self.self_norm(x)

        SA = self.self_attention(y, y, y, bias)

        SA = self.output_dropout(SA)

        return x + SA


class Fusion(nn.Module):
    def __init__(self,
                 hidden_dim,
                 input_dropout=0.1,
                 weights_dropout=0.1,
                 output_dropout=0.1):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout)
        self.encoder = Encoder(hidden_dim, output_dropout, weights_dropout)
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, sim, x):

        expanded_sim, x = sim, x.to(torch.float32)

        bias = expanded_sim.clone()

        x = self.input_dropout(x)

        output = self.encoder(x, bias)

        output = self.final_ln(output)

        return output


if __name__ == '__main__':
    model = Fusion(6)
    x = torch.tensor([[1, 2, 3, 4, 5, 6],
                      [1, 2, 3, 4, 5, 6]])
    sim = torch.zeros((2, 2))
    print(model(sim, x))


