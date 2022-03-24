#Author:ike yang

import torch
from torch import nn



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
    def __init__(self,wt,p,CNNoutD,RNNHidden,RNNOutD,hiddenC=32):
        super(CNNAndLSTM, self).__init__()

        self.cnn = CNN(wt, p, CNNoutD,hiddenC=hiddenC)
        self.lstm = nn.LSTM(CNNoutD,RNNHidden,batch_first=True)
        self.outfc=nn.Sequential(
            nn.Linear(RNNHidden, 2*RNNHidden),
            nn.ReLU(),
            nn.Linear(2*RNNHidden, RNNOutD)
        )


        self.tou=1
        self.softwt= nn.Softmax(dim=-1)
        self.softp= nn.Softmax(dim=0)

    def forward(self, x):
        # input size= ( bs , wl, wt,params)


        ###ROC
        weightL = self.softwt(self.weightPML/self.tou)
        weightR = self.softp(self.weightPMR/self.tou)
        x = torch.matmul(weightL, x)
        x = torch.matmul(x, weightR)  # bs,wl,wt,p
        ##CNNLSTM
        bs, wl, wt, params=x.shape
        x = x.view(bs * wl, wt, params)
        x = self.cnn(x)
        x = x.view(bs, wl, -1)
        x,_ = self.lstm(x)
        x = self.outfc(x[:, -1, :])
        return x, weightL, weightR

