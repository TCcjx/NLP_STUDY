"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: dataset2.py
 @DateTime: 2024-12-30 15:42
 @SoftWare: PyCharm
"""
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

class MyDataset:
    def __init__(self,all_text,all_label,batch_size,shuffle = True):
        self.all_text = all_text
        self.all_label = all_label
        self.batch_size = batch_size
        self.shuffle = shuffle

        assert  len(all_text) == len(all_label)

    def __iter__(self):
        return DataLoader(self)

    def __len__(self):
        return len(self.all_text)

class DataLoader(object):
    def __init__(self,dataset):
        self.dataset = dataset
        self.cursor = 0 # 光标
        self.shuffle_index = np.arange(len(self.dataset)

        if self.dataset.shuffle == True:
            np.random.shuffle(self.shuffle_index)

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration

        index = self.shuffle_index[self.cursor : self.cursor + self.dataset.batch_size]
        text = self.dataset.all_text[index]
        label = self.dataset.all_label[index]

        self.cursor += self.dataset.batch_size
        return text,label

if __name__ == '__main__':
    train_text, train_label = read_data(os.path.join('..','data','train.txt'))
    assert len(train_text) == len(train_label)
    print(f'数据加载成功，数据长度为{len(train_text)}')