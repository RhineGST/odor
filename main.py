import model
import dataset
import config as cfg

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

data = dataset.DataSet(root='Dataset', batch=10)
data_loader = torch.utils.data.DataLoader(data, batch_size=cfg.batchSize,
                                                     shuffle=True, num_workers=cfg.workers)

net = model.NET()
net.train()

optimizer = optim.Adadelta(net.parameters(), lr=0.1, weight_decay=1e-4)

epoch_size = len(data_loader)

if __name__ == '__main__':
    for epoch in range(50):
        data_iter = iter(data_loader)
        correct = float(0)
        total = float(0)
        for i in range(epoch_size):
            val, cls = next(data_iter)
            # print(val[0, :])
            val = Variable(val.to(torch.float32))
            cls = cls.to(torch.long)
            # print(val.shape)
            pre = net(val)
            loss = F.cross_entropy(pre, cls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lab = torch.argmax(pre, 1)
            correct += (lab == cls).sum().float()
            total += len(cls)

            print("epoch:%d loss:%f, acc:%f" % (epoch, loss.data, correct/total))