"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: dataset2.py
 @DateTime: 2024-12-30 15:42
 @SoftWare: PyCharm
"""
import os
def read_data(file):
    all_text = []
    all_label = []
    with open(file,'r',encoding='utf-8') as f:
        all_data = f.read().split('\n')
    for data in all_data:
        data = data.split(" ")
        if len(data) != 2:
            continue
        else:
            text, label = data
        all_text.append(text)
        all_label.append(label)
    return all_text, all_label

if __name__ == '__main__':
    train_text, train_label = read_data(os.path.join('..','data','train.txt'))
    print(train_text,train_label)