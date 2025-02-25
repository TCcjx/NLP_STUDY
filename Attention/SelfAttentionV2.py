"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: SelfAttentionV2.py
 @DateTime: 2025-01-07 10:49
 @SoftWare: PyCharm
"""
import math
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings(action='ignore')

'''
chaofa 版本
'''
# class SelfAttentionV2(nn.Module):
#     def __init__(self,hidden_dim):
#         super(SelfAttentionV2, self).__init__()
#
#         self.hidden_dim = hidden_dim
#
#         self.QKV_proj = nn.Linear(hidden_dim,hidden_dim*3)
#
#     def forward(self,X):
#         QKV = self.QKV_proj(X) # 相较于V1版本最大的改进就是将V1的Q、K、V合并在一起，进行矩阵加速运算
#
#         Q,K,V = torch.split(QKV,self.hidden_dim,dim=-1)
#
#         atten_weight = torch.softmax(
#             Q @ K.transpose(-1,-2) / math.sqrt(self.hidden_dim),
#             dim = -1
#         )
#         output = atten_weight @ V
#         return output

class SelfAttentionV2(nn.Module):
    def __init__(self,dim):

        super(SelfAttentionV2, self).__init__()
        self.dim = dim
        self.QKV_proj = nn.Linear(dim,dim*3)
        self.output_layer = nn.Linear(dim,dim)

    def forward(self,x):
        QKV = self.QKV_proj(x)
        Q,K,V = torch.split(QKV,self.dim,dim=-1)

        attention_weight = torch.matmul(Q,K.traspose(-1,-2)) / math.sqrt(self.dim)
        attention_weight = torch.softmax(
            attention_weight,dim=-1
        )
        attention_score = attention_weight @ V
        output = self.output_layer(attention_score)
        return output


x = torch.randn(3,2,4)
net = SelfAttentionV2(4)
print(net(x).shape)
