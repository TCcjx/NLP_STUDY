"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: 梯度下降.py
 @DateTime: 2025-01-14 16:48
 @SoftWare: PyCharm
"""
"""
sqrt(3) = ?
x ** 2 = 3 , x = ?
"""

lr = 0.01
label = 3
init_x = 5
epoch = 100

for e in range(epoch):
    pred = init_x ** 2
    loss = (label - pred) ** 2
    delta_x =  2*(label-pred) * (-2) * init_x

    init_x -= delta_x * lr
    if epoch % 5 == 0:
        print("init_x:",init_x)

print('final init_x:',init_x)
