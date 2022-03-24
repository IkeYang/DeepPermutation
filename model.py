#Author:ike yang

import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import copy
from einops.layers.torch import Rearrange
from einops import rearrange, reduce, repeat
import pandas as pd
import matplotlib.pyplot as plt
import pickle
def watchV(tensor):
    return tensor.cpu().detach().numpy()



class CNN(nn.Module):
    def __init__(self, wt, p,outD,dropout=0,inputC=1,hiddenC=32):
        super(CNN, self).__init__()
        self.cnn=nn.Sequential(
            nn.Conv2d(inputC,hiddenC,3,1,padding=1),
            # wt,p
            nn.ReLU(),
            nn.MaxPool2d((2,2), 1),
            #wt-1,p-1
            nn.Conv2d(hiddenC, hiddenC, (1,1), 1),
            # wt-1-4,p-1
            nn.ReLU(),
        )
        self.fc=nn.Sequential(nn.Linear((wt-1)*(p-1)*hiddenC,outD),
                              nn.Dropout(dropout),)

    def forward(self,input):
        #input= bs*t,wt,p
        if len(input.shape)==4:
            bsT, c, wt, p = input.shape
            input = input.view(bsT, c, wt, p)
        else:
            bsT,  wt, p = input.shape
            input = input.view(bsT, 1, wt, p)
        out=self.cnn(input)
        out=out.view(bsT,-1)
        out=self.fc(out)
        return out

class CNNAndLSTM(nn.Module):
    def __init__(self,wt,p,CNNoutD,RNNHidden,RNNOutD,hiddenC=32,permu=False,loadWeight=False,name=None,dimWT=-1,dimP=-1):
        super(CNNAndLSTM, self).__init__()

        self.cnn = CNN(wt, p, CNNoutD,hiddenC=hiddenC)
        self.permu = permu
        self.loadWeight = loadWeight

        self.lstm = nn.LSTM(CNNoutD,RNNHidden,batch_first=True)

        self.outfc=nn.Sequential(
            nn.Linear(RNNHidden, 2*RNNHidden),
            nn.ReLU(),
            nn.Linear(2*RNNHidden, RNNOutD)
        )

        if self.permu:
            if loadWeight:
                with open(name, 'rb') as f:
                    wL, wp = pickle.load(f)
                    ind = np.where(wL > 0.5)
                    ind2 = np.where(wL < 0.5)
                    wL[ind] = 1
                    wL[ind2] = 0

                    ind = np.where(wp > 0.5)
                    ind2 = np.where(wp < 0.5)
                    wp[ind] = 1
                    wp[ind2] = 0
                self.weightPML = torch.tensor(wL)
                self.weightPMR = torch.tensor(wp)
            else:
                self.weightPML = Parameter(
                    torch.ones(wt, wt))
                self.weightPMR = Parameter(
                    torch.ones(p, p))
        self.tou=1
        self.softwt= nn.Softmax(dim=dimWT)
        self.softp= nn.Softmax(dim=dimP)

    def forward(self, x):
        # input size= ( bs , wl, wt,params)

        if self.permu:
            if self.loadWeight:
                weightL = self.weightPML.to(x.device)
                weightR = self.weightPMR.to(x.device)
            else:
                weightL = self.softwt(self.weightPML/self.tou)
                weightR = self.softp(self.weightPMR/self.tou)

            x = torch.matmul(weightL, x)
            x = torch.matmul(x, weightR)  # bs,wl,wt,p

        bs, wl, wt, params=x.shape
        # x = x.permute(0, 1, 3, 2)

        x = x.view(bs * wl, wt, params)
        x = self.cnn(x)
        ##shape(bs*t,CNNoutD)

        x = x.view(bs, wl, -1)

        x,_ = self.lstm(x)
        x = self.outfc(x[:, -1, :])
        if self.permu and not self.loadWeight:
            return x, weightL, weightR
        return x

       

if __name__=='__main__':
    # x=torch.rand([16,25])
    # model=RSDAE(25,16,3)
    # print(model(x,prediction=False).shape)
    # print(model(x,prediction=True).shape)

    # x=torch.rand([16,50,4,3])
    # model=RDCNN(wt=4,HL=2,VL=2,covNumber=2,channel=2,ks=4,dropout=0.5,hiddenLayerNumber=2,predL=6,windL=50)
    # print(model(x).shape)

    # x = torch.rand([16, 50])
    # model=RRBMRealV( n_vis=50,n_hid=500,k=5)
    # res=model(x)
    # print(res[0].shape)
    # print(model(x)[1].shape)
    # print(model.free_energy(res[1],res[2],res[3]))

    x = torch.rand([16, 4])
    model=ANIFS(n_in=4, n_cluster=3,n_out=6,cluster_init=None)

    res = model(x)
    print(res.shape)






