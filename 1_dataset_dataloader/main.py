import  random
import numpy as np

class MyDataset(object):
    def __init__(self, all_text, all_label, batch_size, shuffle):
        self.all_text = np.array(all_text)
        self.all_label = np.array(all_label)
        self.batch_size = batch_size
        self.shuffle = shuffle

        assert len(self.all_text) == len(self.all_label)   # 预先assert,断言

    # def __getitem__(self,index):
    #     # print('len(self)')
    #     if index < len(self):
    #         text = self.all_text[index]
    #         label = self.all_label[index]
    #         return text,label
    #     else:
    #         return None,None

    def __iter__(self): # iterator: 迭代器,返回一个具有__next__的对象
        # print('iter')
        return MyDataLoader(self) # 这里的self是MyDataset的一个实例

    def __len__(self):
        return len(self.all_label) # 定义了返回self对象实例的label长度


class MyDataLoader(object):
    def __init__(self,dataset):
        # print(dataset)
        self.cursor = 0
        self.dataset = dataset
        self.shuffle_index = np.arange(len(dataset))

        if self.dataset.shuffle == True:
            np.random.shuffle(self.shuffle_index)
            print(self.shuffle_index)

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration # 中止迭代

        batch_index = self.shuffle_index[self.cursor:self.cursor+self.dataset.batch_size] # numpy 批量索引的方法

        batch_text = self.dataset.all_text[batch_index]
        batch_label = self.dataset.all_label[batch_index]
        self.cursor += self.dataset.batch_size # 后移batch_size

        # for i in range(self.batch_size):
        #     if self.cursor < len(self.shuffle_index):
        #         index = self.shuffle_index[self.cursor]
        #         text,label = self[index]
        #
        #         batch_text.append(text)
        #         batch_label.append(label)
        #     # 光标后移
        #     self.cursor += 1
        return batch_text,batch_label


def get_data():
    all_text = ['你很丑','你很好看','你太漂亮了','这个很难吃','这个还不错']
    all_label = [0,1,1,0,1]
    return all_text,all_label

if __name__ == "__main__":
    all_text, all_label = get_data()  # 迭代器 、 生成器 、 可迭代对象 、 yield
    batch_size = 2
    epochs = 5
    dataset = MyDataset(all_text,all_label,batch_size,True)

    for epoch in range(epochs):
        for batch_text, batch_labels in dataset:
            print(batch_text,batch_labels)
        print("----------------------------")