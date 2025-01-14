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

def build_word_2_index(train_text):  # 建立 word2index 唯一索引
    word_2_index = {'<PAD>':0}
    for text in train_text:
        for w in text:
            if w not in word_2_index:
               word_2_index[w] = len(word_2_index)

    return word_2_index

class MyModel:
    def __init__(self):
        self.model = np.random.normal(0,1,(4,1))

    def forward(self,batch_idx):
        pre = batch_idx @ self.model
        return pre

class MyDataset:
    def __init__(self,all_text,all_label,batch_size,shuffle = True):
        self.all_text = np.array(all_text)
        self.all_label = np.array(all_label)
        self.batch_size = batch_size
        self.shuffle = shuffle

        assert  len(all_text) == len(all_label)

    def __iter__(self): # iterator: 迭代器,返回一个具有__next__的对象
        return DataLoader(self)

    def __len__(self):
        return len(self.all_text)

class DataLoader(object):
    def __init__(self,dataset):
        self.dataset = dataset
        self.cursor = 0 # 光标
        self.shuffle_index = np.arange(len(self.dataset))

        if self.dataset.shuffle == True:
            np.random.shuffle(self.shuffle_index)

    def __getitem__(self, index):
        global max_len , word_2_index # 声明全局变量
        # 先裁剪， word --> index , 填充
        text = self.dataset.all_text[index][:max_len] # 先进行裁剪，如果长度超了不会报错
        text_idx = [word_2_index[i] for i in text]  # 转换成index
        if len(text_idx) < max_len: # 填充<PAD>
            print('进行填充......')
            text_idx  += [0] * (max_len - len(text_idx))
        label = self.dataset.all_label[index]
        return text,text_idx,label

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration
        index = self.shuffle_index[self.cursor : self.cursor + self.dataset.batch_size]
        batch_text = []
        batch_text_idx = []
        batch_label = []
        for i in index:
            text ,text_idx, label = self[i]
            batch_text.append(text)
            batch_text_idx.append(text_idx)
            batch_label.append(label)
        self.cursor += self.dataset.batch_size
        return batch_text,np.array(batch_text_idx),batch_label

if __name__ == '__main__':
    train_text, train_label = read_data(os.path.join('..','data','train.txt'))
    word_2_index = build_word_2_index(train_text) # 将对应字符转换成数字索引
    print(word_2_index)
    assert len(train_text) == len(train_label)
    print(f'数据加载成功，数据长度为{len(train_text)}')

    epochs = 3
    batch_size = 2
    max_len = 4
    model = MyModel()

    train_dataset = MyDataset(train_text,train_label,batch_size,shuffle=True)
    for epoch in range(epochs):
        print('*' * 20)
        for batch_text,batch_text_idx,batch_label in train_dataset:
            print('batch_text:')
            print(batch_text,batch_text_idx.shape,batch_label)
            pre = model.forward(batch_text_idx)
            print(pre)