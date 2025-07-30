import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, emb_dim, qkv_bias=False, att_dropout=0.0, proj_drop=0.0):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5
        self.q_map = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.k_map = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.v_map = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(att_dropout)

        self.proj = nn.Linear(emb_dim, emb_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, Q, K, pad_mask=None):
        '''

        :param x: [batch_size, 512]
        :param context: [batch_szie,  emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        # B, N = K.shape
        V = K

        Q = self.q_map(Q)
        K = self.k_map(K)
        V = self.v_map(V)

        # [batch_size, h*w, seq_len]
        att_weights = torch.matmul(Q, K.transpose(-2, -1))
        att_weights = att_weights * self.scale


        if pad_mask is not None:
            # [batch_size, h*w, seq_len]
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = self.attn_drop(att_weights)

        out = torch.matmul(att_weights, V)   # [batch_size, h*w, emb_dim]
        out = self.proj(out)
        out = self.proj_drop(out)
        out = Q+out

        return out
