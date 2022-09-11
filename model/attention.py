import math
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weight


class Attention(nn.Module):
    def __init__(self, num_heads, emb_dim, drop):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        assert emb_dim % self.num_heads == 0, "can not divide attention dim by number of heads"

        self.d_h = emb_dim // self.num_heads

        self.Q = nn.Linear(emb_dim, emb_dim)
        self.K = nn.Linear(emb_dim, emb_dim)
        self.V = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(drop)

        init_weight(self.Q)
        init_weight(self.K)
        init_weight(self.V)

        self.fc = nn.Linear(emb_dim, emb_dim)

        init_weight(self.fc)

    def forward(self, q_, k_, v_, mask):
        batch_size = q_.shape[0]
        # q, k, v: (batch_size, max_seq_len, lstm_out_dim)
        # mask: (batch_size, max_seq_len, max_seq_len)

        q = self.Q(q_)
        k = self.K(k_)
        v = self.V(v_)

        q = q.view(batch_size, -1, self.num_heads, self.d_h).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_heads, self.d_h).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.num_heads, self.d_h).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-1, -2)
        attn = attn / math.sqrt(self.d_h)

        mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)               # (32, 256)
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, k_.shape[1])  # (32, 256, 406)

        if mask is not None:
            attn.masked_fill_(~mask, -1e12)

        attn_score = F.softmax(attn, dim=-1)
        # for exploding
        attn_score = self.dropout(attn_score)
        out = (attn_score @ v).permute(0, 2, 1, 3)

        out = out.reshape(batch_size, -1, self.d_h * self.num_heads)

        # residual connection
        out = self.fc(out + q_)

        return out