from time import sleep
import torchvision as tv
import numpy as np
from torchvision import transforms
import torch as t
from torch.utils.data import DataLoader
from torch.nn import BCELoss
from torch.optim import Adam
from net import Net
from visdom import Visdom
from options import *
from utils import to_categorical

transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.ToTensor(),
])
trainDataset = tv.datasets.MNIST(
    root="./data", # 下载数据，并且存放在data文件夹中
    train=True,
    transform=transform,
    download=True
)
trainLoader = DataLoader(dataset=trainDataset, shuffle=True, batch_size=batch_size)
testDataset = tv.datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)
testLoader = DataLoader(dataset=testDataset, shuffle=True, batch_size=batch_size)
#---------------------
device = 'cpu'
if t.cuda.is_available():
    device = 'cuda'

net = Net().to(device)
criterion = BCELoss().to(device)
optimizer = Adam(net.parameters(), lr=lr, betas=(0.5, 0.99))
viz = Visdom()

#---------------------

if is_train:#train
    print('[*]Start to Train !')
    for i in range(epoch):
        for x_train, y_train in trainLoader:

            x_train_gpu = x_train.to(device)
            y_train_gpu = y_train.to(device)
            y_train_gpu_cate = to_categorical(y_train).to(device)


            correct_num = 0
            tot_num = y_train_gpu.shape[0]

            output = net(x_train_gpu)
            loss = criterion(output, y_train_gpu_cate)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct_num = (t.eq(output.topk(1)[1].squeeze(dim=1), y_train_gpu)).sum().item()
            print('[train] [%d, %d] acc on this batch: [%d/%d] = %f %%' % (i+1, epoch, correct_num, tot_num, correct_num/tot_num*100), end=', ')
            print('loss = %f' % loss)

        #save every epoch
        t.save(net.state_dict(), net_path)
else:#test
    print('[!]Start to Test !')
    net.load_state_dict(t.load(net_path))
    correct_num = 0
    tot_num = 0
    is_first = True
    for x_test, y_test in testLoader:

        x_test_gpu = x_test.to(device)
        y_test_gpu = y_test.to(device)
        y_test_gpu_cate = to_categorical(y_test).to(device)

        tot_num += y_test_gpu.shape[0]

        output = net(x_test_gpu)
        if is_first:
            is_first = False
            viz.images(x_test)
            print(y_test)
            print('to show prediction of first batch, sleep for 20s.')
            sleep(20)

        loss = criterion(output, y_test_gpu_cate)

        correct_num += (t.eq(output.topk(1)[1].squeeze(dim=1), y_test_gpu)).sum().item()
        print('[test] acc(on whole test data) after this batch: (%d/%d) = %f %%' % (correct_num, tot_num, correct_num/tot_num*100), end=', ')
        print('loss = %f' % loss)
