from train import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import copy

if __name__ == '__main__':

    WTs = {'Borne': 4,
                'Tie': 25,
                'K1': 33,
                'K2': 33,
                'ZM': 11,
                }

    hiddenC=32
    RNNHidden=128
    deviceNum=1
    epoch = 100
    server = True
    selectData = 'WindPower'#WindSpeed
    hyperparaPMTrain1={
    'hiddenC':hiddenC,
    'RNNHidden':RNNHidden,
    'permu':True,
    'loadWeight':False,
    'server':server,
    'deviceNum':deviceNum,
    'epoch':epoch,
    'lossWT':2,
    'lossP':2,
    'selectData':selectData,
    }
    hyperparaPMTrain2={
    'hiddenC':hiddenC,
    'RNNHidden':RNNHidden,
    'permu':True,
    'loadWeight':False,
    'server':server,
    'deviceNum':deviceNum,
    'epoch':epoch,
    'lossWT':0,
    'lossP':0,
    'selectData':selectData,
    }

    hyperparaPMTrain3={
    'hiddenC':hiddenC,
    'RNNHidden':RNNHidden,
    'permu':True,
    'loadWeight':False,
    'server':server,
    'deviceNum':deviceNum,
    'epoch':epoch,
    'lossWT':1,
    'lossP':1,
    'selectData':selectData,
    }
    hyperparaNormal = {
        'hiddenC': hiddenC,
        'RNNHidden': RNNHidden,
        'permu': False,
        'loadWeight': False,
        'server': server,
        'deviceNum': deviceNum,
        'epoch': epoch,
        'selectData':selectData,
    }

    import copy

    methodTotal = 2 * 6 + 1
    ilist=[0,0.2,0.4,0.6,0.8,1]
    hpL = []
    hpL.append(copy.deepcopy(hyperparaNormal))
    for i in range(methodTotal - 1):
        hyperparaNormal['permu'] = True
        hyperparaNormal['lossWT'] = 2
        hyperparaNormal['lossP'] = 2

        if i > 5:
            hyperparaNormal['tou'] = False
            hyperparaNormal['thresP'] = ilist[i-6]
            hyperparaNormal['thresWt'] = ilist[i-6]
        else:
            hyperparaNormal['tou'] = True
            hyperparaNormal['thresP'] = ilist[i]
            hyperparaNormal['thresWt'] = ilist[i]
        hpL.append(copy.deepcopy(hyperparaNormal))
        # if i%2==0:
        #     hyperparaNormal['lossWT']=3
        #     hyperparaNormal['lossP']=3
        # else:
        #     hyperparaNormal['lossWT'] = 2
        #     hyperparaNormal['lossP'] = 2



    nummberRepeat=3

    resT={}
    # for wf in ['Borne','Tie','K1','K2','ZM']:
    for wf in ['K2','ZM']:
        wts=WTs[wf]
        resT[wf]=np.zeros([wts,methodTotal,nummberRepeat])
        for wt in range(wts):
            for v in range(nummberRepeat):
                for j in range(methodTotal):
                    if j==0:
                        savename='K3CNNLSTMWPV%d'%(v)
                    else:
                        savename='K3Permu%dCNNLSTMWPV%d'%(j,v)
                    print(hpL[j])
                    ypT, ytT=trainMain('CNNLSTM',wf,wt,savename,hpL[j])

                    resT[wf][wt, j, v] =np.sqrt((np.mean((ypT - ytT) ** 2)))

                with open('resSoftFinalK2', 'wb') as f:
                    pickle.dump(resT, f)