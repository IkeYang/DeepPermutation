import numpy as np

from torch.utils.data import Dataset

import torch


def getDataSet():
    '''
    load your own dataset with the shape of [n,wt,p],
    where n is the number of total cases sorted with chronological order
    wt is the number of wind turbine
    p is the number of considered parameters and the first parameter is wind speed
    :return:
    '''
    pass



class SCADADataset(Dataset):
    def __init__(self,windL ,predL ,wtnum):
        '''

        :param windL:
        :param predL:
        :param wtnum:
        '''
        self.data =getDataSet()
        self.predL =predL
        self.windL =windL
        self.wtnum =wtnum
        self.datashape = list(self.data.shape)

    def __len__(self):
        return self.data.shape[0] - self.windL - self.predL

    def __getitem__(self, idx):

        x = np.copy(self.data[idx:idx + self.windL, :, :])
        x = torch.from_numpy(x).float()

        y = np.copy(self.data[idx + self.windL:idx + self.windL + self.predL, self.wtnum, 0])
        y = torch.from_numpy(y).float()
        return x, y
