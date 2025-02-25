"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: SelfAttentionV4.py
 @DateTime: 2025-01-07 11:27
 @SoftWare: PyCharm
"""
"""
面试版本
"""
import math
import torch
import torch.nn as nn

"""
chaofa 版本
"""
# class SelfAttentionV4(nn.Module):
#
#     def __init__(self, dim, dropout_rate = 0.1):
#         super(SelfAttentionV4, self).__init__()
#
#         self.dim = dim
#
#         self.query = nn.Linear(dim,dim)
#         self.key = nn.Linear(dim,dim)
#         self.value = nn.Linear(dim,dim)
#
#         self.dropout = nn.Dropout(dropout_rate)
#         self.output_layer = nn.Linear(dim,dim) # 输出映射层
#
#     def forward(self, X, attention_mask = None): # X -> (bs,sq,dim)
#         Q = self.query(X)
#         K = self.key(X)
#         V = self.value(X) # Q,K,V -> (bs,sq,dim)
#
#         attention_weight = torch.matmul(Q,K.transpose(-1,-2)) / math.sqrt(self.dim) # (bs,sq,sq)
#         if attention_mask is not None:
#             print('attention_mask:',attention_mask)
#             attention_weight = attention_weight.masked_fill(
#                 attention_mask == 0, float('-inf')
#             )
#         attention_weight = torch.softmax(attention_weight,dim=-1)
#         print('attention_weight:',attention_weight)
#         attention_weight = self.dropout(attention_weight)
#         attention_result = attention_weight @ V # (bs,sq,dim)
#         output = self.output_layer(attention_result)
#
#         return output


class SelfAttentionV4(nn.Module):

    def __init__(self,dim,dropout_rate = 0.1):
        super(SelfAttentionV4, self).__init__()

        self.dim = dim
        self.query_proj = nn.Linear(dim,dim)
        self.key_proj = nn.Linear(dim,dim)
        self.value_proj = nn.Linear(dim,dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(dim,dim)

    def forward(self,x,mask=None):
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        attention_weight = torch.matmul(Q,K.transpose(-1,-2)) / math.sqrt(self.dim)
        if mask is not None:
            attention_weight = attention_weight.masked_fill(
                mask == 0,float('-inf')
            )
        attention_weight = torch.softmax(
            attention_weight,dim=-1
        )
        attention_weight = self.dropout(attention_weight)
        output = attention_weight @ V
        output = self.output_layer(output)
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
net = SelfAttentionV4(2)
print(net(X, mask).shape)