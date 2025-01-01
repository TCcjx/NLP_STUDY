"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: dataset2.py
 @DateTime: 2024-12-30 15:42
 @SoftWare: PyCharm
"""
import random
import os
import numpy as np

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

def build_word_2_index(train_text):  # 建立 word2index 唯一索引
    word_2_index = {'<PAD>':0}
    for text in train_text:
        for w in text:
            if w not in word_2_index:
               word_2_index[w] = len(word_2_index)

    return word_2_index

def get_dataset(all_text, all_label, batch_size, shuffle):
    batch_time = int(np.ceil(len(all_text) / batch_size))
    for i in range(batch_time):
        batch_text = all_text[i * batch_size:(i+1) * batch_size]
        batch_label = all_label[i * batch_size:(i+1) * batch_size]
        yield batch_text,batch_label


if __name__ == '__main__':
    train_text, train_label = read_data(os.path.join('..', 'data', 'train.txt'))
    word_2_index = build_word_2_index(train_text)  # 将对应字符转换成数字索引
    # print(word_2_index)
    assert len(train_text) == len(train_label)
    print(f'数据加载成功，数据长度为{len(train_text)}')

    epochs = 3
    batch_size = 2
    max_len = 4

    for epoch in range(epochs):
        dataset = get_dataset(train_text, train_label, batch_size, shuffle=True) # 生成器，返回一个可迭代对象
        print(dataset)
        for batch_idx,batch_label in dataset:
            print(batch_idx,batch_label)