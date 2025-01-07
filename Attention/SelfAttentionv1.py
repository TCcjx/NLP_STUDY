"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: SelfAttentionv1.py
 @DateTime: 2025-01-07 10:24
 @SoftWare: PyCharm
"""

import math
import torch.nn as nn
import torch

import warnings
warnings.filterwarnings(action="ignore")

class SelfAttentionV1(nn.Module):
    def __init__(self,hidden_dim):
        super(SelfAttentionV1, self).__init__()
        self.hidden_dim = hidden_dim

        self.query_proj = nn.Linear(hidden_dim,hidden_dim)
        self.key_proj = nn.Linear(hidden_dim,hidden_dim)
        self.value_proj = nn.Linear(hidden_dim,hidden_dim)

    def forward(self,x): # X shape: (batch_size,seq_len,hidden_dim)
        Q  = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)  # Q\K\V shape: (batch_size, seq_len, hidden_dim)
        print(f"Q:\n{Q.shape}\nK:\n{K.shape}\nV:\n{V.shape}")

        attention_value = torch.matmul(Q,K.transpose(-1,-2)) # attention_value : (batch_size,seq_len,seq_len)
        attention_weight = torch.softmax(  # a.防止过拟合 b.为了让Q,K内积分布保持和输入一样
            attention_value / math.sqrt(self.hidden_dim),
            dim = -1
        )
        output = torch.matmul(attention_weight,V) # output: (batch_size, seq_len, hidden_dim)
        return output

x = torch.randn(3,2,4)
net = SelfAttentionV1(4)
print(net(x).shape)