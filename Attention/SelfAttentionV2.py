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

class SelfAttentionV2(nn.Module):
    def __init__(self,hidden_dim):
        super(SelfAttentionV2, self).__init__()

        self.hidden_dim = hidden_dim

        self.QKV_proj = nn.Linear(hidden_dim,hidden_dim*3)

    def forward(self,X):
        QKV = self.QKV_proj(X) # 相较于V1版本最大的改进就是将V1的Q、K、V合并在一起，进行矩阵加速运算

        Q,K,V = torch.split(QKV,self.hidden_dim,dim=-1)

        atten_weight = torch.softmax(
            Q @ K.transpose(-1,-2) / math.sqrt(self.hidden_dim),
            dim = -1
        )
        output = atten_weight @ V
        return output


x = torch.randn(3,2,4)
net = SelfAttentionV2(4)
print(net(x).shape)
