"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: 线性回归房价预测.py
 @DateTime: 2025-01-14 17:33
 @SoftWare: PyCharm
"""
import random

start_year = 2000
end_year = 2022
years = [i for i in range(start_year,end_year+1)]
prices = [i for i in range(23)]
assert len(years) == len(prices)


# y = kx + b
# loss = (kx + b - label) ** 2
# delta_k = 2(kx + b - label) * x
# delta_b = 2(kx + b - label)
k = 99
b = -1
lr = 0.0000001
epochs = 10

for i in range(epochs):
    for x,y in zip(years,prices):
        pred = k * x + b
        loss = (y - pred) ** 2
        k -= lr * 2 * (pred - y) * x
        b -= lr * 2 * (pred - y)
    print(f'epoch:{i+1} , loss: {loss}')
print(f'k:{k},b:{b}')

