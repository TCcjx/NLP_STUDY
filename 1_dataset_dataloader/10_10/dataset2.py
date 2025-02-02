"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: dataset2.py
 @DateTime: 2024-12-30 15:42
 @SoftWare: PyCharm
"""
import random
import numpy as np

class MyDataset:
    def __init__(self, all_text, all_label,batch_size , shuffle = True):
        self.all_text = np.array(all_text)
        self.all_label = np.array(all_label)
        self.batch_size = batch_size
        self.shuffle = shuffle

        assert len(self.all_text) == len(self.all_label),'数据和标签长度不相等！'


    def __iter__(self):
        return DataLoader(self)  # 迭代器，返回一个一个具有__next__的对象

    def __len__(self):
        return len(self.all_text)

class DataLoader(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.cursor = 0
        self.shuffle_index = np.arange(len(self.dataset))

        if self.dataset.shuffle == True:
            np.random.shuffle(self.shuffle_index)

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration

        index = self.shuffle_index[self.cursor : self.cursor + self.dataset.batch_size] # 切片操作，不用担心越界
        text  = self.dataset.all_text[index]
        label = self.dataset.all_label[index]

        self.cursor += self.dataset.batch_size # 光标后移batch_size
        return text, label



def get_data():
    all_text = ['你很丑','你很好看','你太漂亮了','这个很难吃','这个还不错']
    all_label = [0,1,1,0,1]
    return all_text,all_label

if __name__ == '__main__':
    all_text, all_label = get_data()
    epochs = 10
    batch_size = 3

    dataset = MyDataset(all_text,all_label,batch_size,shuffle=True)

    for epoch in range(epochs):
        print('---' * 20)
        for batch_text,batch_label in dataset:
            print(epoch+1,batch_text,batch_label)