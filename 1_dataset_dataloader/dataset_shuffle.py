"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: dataset_shuffle.py
 @DateTime: 2024-12-28 21:18
 @SoftWare: PyCharm
"""
import numpy as np


class MyDataset:
    def __init__(self, all_text, all_label, batch_size, shuffle):
        self.all_text  = all_text
        self.all_label = all_label
        self.batch_size = batch_size
        self.shuffle = shuffle

        assert len(self.all_text) == len(all_label)

    def __getitem__(self, index):
        if index < len(self):
            text = self.all_text[index]
            label = self.all_label[index]
            return text,label
        else:
            return None, None

    def __iter__(self): # 初始执行__iter__ ,返回一个__next__()对象
        self.cursor = 0 # 记录光标
        print('执行__iter__函数')
        if self.shuffle == True:
            self.shuffle_index = [i for i in range(len(self))]
            np.random.shuffle(self.shuffle_index)

        return self

    def __next__(self):
        if self.cursor >= len(self):
            raise StopIteration

        batch_text = []
        batch_label = []
        for i in range(self.batch_size):
            if self.cursor < len(self.shuffle_index):
                index = self.shuffle_index[self.cursor]
                text, label = self[index]
                # if text != None:
                batch_text.append(text)
                batch_label.append(label)
                self.cursor += 1
        return batch_text, batch_label

    def __len__(self,):
        return len(self.all_text)

def get_data():
    all_text = ['你很丑','你很好看','你太漂亮了','这个很难吃','这个还不错']
    all_label = [0,1,1,0,1]
    return all_text,all_label

if __name__ == "__main__":
    all_text , all_label = get_data()
    batch_size = 2
    epochs = 10
    shuffle = True

    dataset = MyDataset(all_text,all_label,batch_size,shuffle)
    for epoch in range(epochs):
        print('-' * 25)
        for batch_text, batch_label in dataset:
            print(batch_text, batch_label)

