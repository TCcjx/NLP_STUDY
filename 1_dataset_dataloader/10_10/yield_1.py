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
        yield i

if __name__ == '__main__':
    list1 = [1,2,3,4]
    r = func1(list1)
    print(r)
    print(next(r))
    print(next(r))
    print(next(r))
    print(next(r))