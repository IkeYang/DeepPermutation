#Author:ike yang
#Author:ike yang
from model import *

import sys
sys.path.append(r"..\..")
from loadData import SCADADataset
import torch

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.distributions as D
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np

# outf=r'/home/daisy001/mountdir/LuoxiaoYANG/Model/WindPrediction'



def evalCNNLSTM(name,dataName,wtnum,model=None,dataloader=None,server=False,selectData='WindPower',device=None):
    
    if server:
        outf = r'/home/daisy001/mountdir/LuoxiaoYANG/Model/WindPrediction/Permutation'
    else:
        outf = r'D:\YANG Luoxiao\Model\WindSpeed\Permutation'

    batch_size=128
    nameConfig = '%s/%s_%s_WT%d_Config' % (outf, name, dataName, wtnum)
    with open(nameConfig, 'rb') as f:
        parameterDict=pickle.load( f)

    wt=parameterDict['wt']
    p=parameterDict['p']
    predL=parameterDict['predL']
    windL=parameterDict['windL']
    CNNoutD=parameterDict['CNNoutD']
    RNNHidden=parameterDict['RNNHidden']
    hiddenC=parameterDict['hiddenC']
    if device is None:
        device = parameterDict['device']

    permu = parameterDict['permu']
    nameWeight = parameterDict['nameWeight']
    loadWeight = parameterDict['loadWeight']
    dimP = parameterDict['dimP']
    dimWT = parameterDict['dimWT']

    if model is None:
        model =CNNAndLSTM(wt, p, CNNoutD, RNNHidden, predL,dimP=dimP,dimWT=dimWT,
                          name=nameWeight,hiddenC=hiddenC,permu=permu,loadWeight=loadWeight).to(device)
        checkpoint = torch.load(  '%s/%s_%s_WT%d' % (outf, name, dataName, wtnum), map_location=device)
        model.load_state_dict(checkpoint['model'])
    if dataloader is None:
        scadaValDataset = SCADADataset(dataName,prediction=selectData, typeData='Test', windL=windL, predL=predL, wtnum=wtnum,server=server)
        dataloader = torch.utils.data.DataLoader(scadaValDataset, batch_size=batch_size,
                                                    shuffle=False, num_workers=int(0))
    # model.loadWeight=loadWeight
    model.eval()
    yNp = np.zeros([1, predL])
    yReal = np.zeros([1, predL])

    with torch.no_grad():
        c=0
        for i, (x, y) in enumerate(dataloader):
            c += 1
            x = x.to(device)
            y = y.to(device)
            y2 = np.copy(y.cpu().detach().numpy()) * 20.76
            if permu and not loadWeight:
                ypred = model(x)[0] * 20.76
            else:
                ypred = model(x)* 20.76

            yNp = np.vstack((yNp, ypred .cpu().detach().numpy()))
            yReal = np.vstack((yReal, y2))


    # if dataName=='Borne':
    #     return yNp[1:, :]/ 20.76 * 24.27 , yReal[1:, :]/ 20.76 * 24.27
    # elif dataName=='10s':
    #     return yNp[1:, :] / 20.76 * 25.5+0.4, yReal[1:, :] / 20.76 * 25.5+0.4
    # elif 'ZM' in dataName:
    #     return yNp[1:, :] , yReal[1:, :]
    # else:
    return yNp[1:, :], yReal[1:, :]


def evalPM(dataName,wtnum,model=None,predL=6,windL=50,server=False,noise=False):
    scadaValDataset = SCADADataset(dataName, prediction='WindPower', typeData='Test', windL=windL, predL=predL,
                                   wtnum=wtnum, server=server)
    dataloader = torch.utils.data.DataLoader(scadaValDataset, batch_size=512,
                                             shuffle=False, num_workers=int(0))

    yNp = np.zeros([1, predL])
    yReal = np.zeros([1, predL])

    with torch.no_grad():
        c=0
        for i, (x, y) in enumerate(dataloader):
            c += 1
            yp=np.repeat(x[:,-1,wtnum,0].reshape((-1,1)), predL, axis=1)


            yNp = np.vstack((yNp,yp* 20.76))
            yReal = np.vstack((yReal, y* 20.76))


    if dataName=='Borne':
        return yNp[1:, :]/ 20.76 * 24.27 , yReal[1:, :]/ 20.76 * 24.27
    elif dataName=='10s':
        return yNp[1:, :] / 20.76 * 25.5+0.4, yReal[1:, :] / 20.76 * 25.5+0.4
    elif 'ZM' in dataName:
        return yNp[1:, :] , yReal[1:, :]
    else:
        return yNp[1:, :], yReal[1:, :]



if __name__=='__main__':
    # yp, yt =  evalPM('Borne',0,model=None,predL=6,windL=50,server=False)
    # print('Testloss:  ', np.sqrt((np.mean((yp - yt) ** 2))))
    
    #PM ZM3-5 Testloss:   1.1815953599374274
    #PM 10s Testloss:  0.997900654014453
    #PM Borne Testloss:   1.3457613183974069
    #PM k1 Testloss:   1.97

    #1.91
    for windL in [5,6]:
        print(windL)
        for _ in range(10):
            yp, yt =evalDTSF('ZM3-5', m=[100,300,500,200,50], wtnum=0, model=None, predL=6, windL=windL, server=False)
            print('Testloss:  ', np.sqrt((np.mean((yp - yt) ** 2))))