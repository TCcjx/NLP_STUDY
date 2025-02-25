"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: dataset1.py
 @DateTime: 2024-12-29 21:38
 @SoftWare: PyCharm
"""
class MyDataset:
    def __init__(self, all_data, all_label, batch_size):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass


def get_data():
    all_text = ['你很丑','你很好看','你太漂亮了','这个很难吃','这个还不错']
    all_label = [0,1,1,0,1]
    return all_text,all_label

if __name__ == '__main__':
    all_text, all_label = get_data()

    epochs = 10
    batch_size = 2
    dataset = MyDataset(all_text, all_label, batch_size)
    for epoch in range(epochs):
        # for t in dataset:
            pass

    list1 = [1,2,3,4]
    print(list1[0:8])
    list1_t = list1.__iter__()
    while True:
        try:
            print(next(list1_t))
        except:
            print('break')
            break

    for i in list1:
        print(i)

