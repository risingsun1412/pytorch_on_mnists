import torch as t

def to_categorical(data):
    res = []
    for i in range(data.shape[0]):
        a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        a[data[i]] = 1.0
        res.append(a)
    return t.tensor(res)