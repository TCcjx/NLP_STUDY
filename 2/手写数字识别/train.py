"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: train.py
 @DateTime: 2025-02-17 10:23
 @SoftWare: PyCharm
"""
import torch
import numpy as np
import struct
import os
import matplotlib.pyplot as plt

def load_labels(file): # 加载数据
    with open(file,'rb') as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]))

def load_image(file):  # 加载数据
    with open(file,"rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">iiii",data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items,-1)

class Dataset(object):
    def __init__(self, all_text, all_label, batch_size, shuffle):
        self.all_text = np.array(all_text)
        self.all_label = np.array(all_label)
        self.batch_size = batch_size
        self.shuffle = shuffle

        assert len(all_text) == len(all_label)

    def __iter__(self):
        return DataLoader(self)

    def __len__(self):
        return len(self.all_text)

class DataLoader(object):
    def __init__(self, dataset):
        self.cursor = 0
        self.dataset = dataset
        self.shuffle_index = np.arange(len(dataset))

        if self.dataset.shuffle == True:
            np.random.shuffle(self.shuffle_index)
            # print(self.shuffle_index)

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration

        batch_index = self.shuffle_index[self.cursor:self.cursor+self.dataset.batch_size]

        batch_text = self.dataset.all_text[batch_index]
        batch_label = self.dataset.all_label[batch_index]
        self.cursor += self.dataset.batch_size

        batch_label = label_to_onehot(batch_label)
        return batch_text, batch_label

def label_to_onehot(labels,class_num = 10):
    batch_num = labels.shape[0]
    result = np.zeros((batch_num,class_num))

    for i,j in enumerate(labels):
        result[i][j] = 1
    return result

def softmax(x):
    ex = np.exp(x)
    # ex_sum = np.sum(ex,axis=1).reshape(-1,1)
    ex_sum = np.sum(ex,axis=1,keepdims=True)
    return ex / ex_sum



if __name__ == '__main__':
    train_data = load_image(os.path.join('./','data','MNIST','train-images-idx3-ubyte','train-images.idx3-ubyte')) / 255
    train_labels = load_labels(os.path.join('./','data','MNIST','train-labels-idx1-ubyte','train-labels.idx1-ubyte'))

    dev_data = load_image(os.path.join('./','data','MNIST','t10k-images-idx3-ubyte','t10k-images.idx3-ubyte')) / 255
    dev_labels = load_labels(os.path.join('./','data','MNIST','t10k-labels-idx1-ubyte','t10k-labels.idx1-ubyte'))
    # data = train_data[0].reshape(28,28)
    W = np.random.normal(0,1,size=(784,10))
    b = 0

    lr = 0.01
    epoch = 20
    batch_size = 20
    shuffle = True


    dataset = Dataset(train_data,train_labels,batch_size=batch_size,shuffle=True)
    for e in range(epoch):
        print(f"{e+1}_{'*'*100}")
        for bi,(batch_data,batch_label) in enumerate(dataset):
            pre = batch_data @ W + b
            soft_pre = softmax(pre)
            loss = -np.sum(batch_label * np.log(soft_pre))

            G = soft_pre - batch_label
            delta_W = batch_data.T @ G
            delta_b = np.mean(G)

            W -= lr * delta_W
            b -= lr * delta_b
            #if bi % 50 == 0:
            #   print(loss)

        # test
        right_num = 0
        p = dev_data @ W + b
        p = np.argmax(p,axis=1)

        for pl,tl in zip(p,dev_labels):
            if pl == tl:
                right_num += 1

        acc = right_num / len(dev_labels) * 100
        print(f"acc:{acc}%")



    # plt.imshow(data,cmap='gray')
    # plt.show()
