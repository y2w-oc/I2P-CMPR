import torch
import torch.nn as nn
import numpy as np
import math


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, num_head=16, dropout=0.0):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = 1e-6

        self.dim = embed_dim // num_head
        self.nhead = num_head

        # multi-head attention
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.merge = nn.Linear(embed_dim, embed_dim, bias=False)
        self.merge_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        bs = query.size(0)

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        # message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]

        Q = self.feature_map(query)
        K = self.feature_map(key)

        v_length = value.size(1)
        value = value / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, value)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        message = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        message = message.contiguous()
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]

        message = self.merge_dropout(message)

        return message


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, num_head=16, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = num_head
        self.attention_head_size = int(embed_dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v):
        """
        Input:
            wkv: embeddings, [B, N, C]
        Output:
            attention_output: embeddings after self-attention, [B, N, C]
        """
        mixed_query_layer = self.query(q)
        mixed_key_layer = self.key(k)
        mixed_value_layer = self.value(v)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, embed_dim=512, dropout=0.0):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(embed_dim, 512)
        self.fc2 = nn.Linear(512, embed_dim)
        self.act_fn = nn.functional.gelu
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, embed_dim=512, num_head=16, dropout=0.0, att_mode='linear'):
        super(ResidualAttentionBlock, self).__init__()
        self.ffn = Mlp(embed_dim=embed_dim, dropout=dropout)
        if att_mode == 'linear':
            self.attention = LinearMultiHeadAttention(embed_dim=embed_dim, num_head=num_head, dropout=dropout)
        elif att_mode == 'vanilla':
            self.attention = MultiHeadAttention(embed_dim=embed_dim, num_head=num_head, dropout=dropout)
        else:
            assert att_mode == 'linear' or att_mode == 'vanilla', "att_mode should be 'linear' or 'vanilla'!"
        self.att_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        """
        Input:
            x: image patch embedding, [B, Ne, Ce]
        """
        residual = x
        x = self.att_norm(x)
        x = self.attention(x, x, x)
        x = x + residual

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + residual
        return x
