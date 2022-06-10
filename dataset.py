import torch
import torch.utils.data as data
import numpy as np
import os
import os.path

class DataSet(data.Dataset):
    def __init__(self, root, batch):
        self.batch = batch
        self.root = root
        self.val = []
        self.cls = []
        for i in range(1, batch + 1):
            with open(os.path.join(root, 'batch' + str(i) + '.dat')) as f:
                for line in f:
                    line = line.split()
                    v = []
                    self.cls.append(int(line[0]) - 1)
                    for j in range(1, 129):
                        v.append(float(line[j].split(':')[1]))
                    self.val.append(v)
        self.val = np.asarray(self.val)
        self.cls = np.asarray(self.cls)


    def __len__(self):
        return self.cls.shape[0]

    def __getitem__(self, item):
        return self.val[item], self.cls[item]
