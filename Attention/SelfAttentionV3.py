"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: SelfAttentionV3.py
 @DateTime: 2025-01-07 10:58
 @SoftWare: PyCharm
"""
import math
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings(action='ignore')

# 1.Dropout位置
# 2.Attention_mask
# 3.output矩阵映射

class SelfAttentionV3(nn.Module):

    def __init__(self, dim, dropout_rate = 0.1)->None:
        super(SelfAttentionV3, self).__init__()
        self.dim = dim

        self.proj = nn.Linear(dim, dim*3)
        self.attention_dropout = nn.Dropout(dropout_rate) # dropout层

        self.output = nn.Linear(dim, dim)

    def forward(self, X, attention_mask = None):

        QKV = self.proj(X)
        Q,K,V = torch.split(QKV,self.dim,dim=-1)
        attention_weight = Q @ K.transpose(-1,-2) / math.sqrt(self.dim)

        if attention_mask is not None:  # (bs,sq,sq) 注意力掩码矩阵
            attention_weight =  attention_weight.masked_fill(
                attention_mask == 0,float('-inf')
            )
        attention_weight = torch.softmax(
            attention_weight,
            dim = -1
        )
        attention_weight = self.attention_dropout(attention_weight)
        attention_result = attention_weight @ V

        output = self.output(attention_result)
        return output

X = torch.rand(3, 4, 2)
b = torch.tensor(
    [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ]
)
print(b.shape)
mask = b.unsqueeze(dim=1).repeat(1, 4, 1)
print(mask)
net = SelfAttentionV3(2)
print(net(X, mask).shape)
