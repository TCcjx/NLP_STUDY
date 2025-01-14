"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: dataset2.py
 @DateTime: 2024-12-30 15:42
 @SoftWare: PyCharm
"""
import random
import numpy as np

def func1(list1):
    for i in list1:
        res = yield i
        # print('res:',res)

def func2(list1):
    for i in list1:
        # return i
        yield i
if __name__ == '__main__':


    list1 = [1,2,3,4]
    print('-' * 25)
    x = func2(list1)
    for i in range(4):
        print(next(x))
    print('-' * 25)

    r = func1(list1) # 返回的是生成器
    for i in r:
        print(i)

    # print(r)
    # print(next(r))
    # print(next(r))
    # print(next(r))
    # print(next(r))