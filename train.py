#Author:ike yang
from model import *
from eval import *
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

import numpy as np
from sklearn.cluster import KMeans

def trainCNNLSTM(epochs=6006, wtnum=0, deviceNum=0, printOut=True, dataName='K1',lr = 2e-4,weight_decay = 0,
                 wp=(50, 6), name='DNN', lossNeed=True,server=False,hiddenC=32,RNNHidden=128,permu=False,
                 loadWeight=False,lossWT=-1,lossP=2,thresWt=0,thresP=0,selectData='WindPower',tou=True):
    torch.cuda.set_device(deviceNum)
    if server:
        outf = r'/home/daisy001/mountdir/LuoxiaoYANG/Model/WindPrediction/Permutation'
    else:
        outf = r'D:\YANG Luoxiao\Model\WindSpeed\Permutation'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    windL, predL = wp

    batch_size = 64
    CNNoutD, RNNHidden, RNNOutD = ( RNNHidden, RNNHidden, predL)




    start_epoch = 0
    loadModel = False

    if lossP==2:
        dimP=0
    elif lossP==1:
        dimP = 0
    elif lossP == 0:
        dimP = -1
    elif lossP == 3:
        dimP = -1

    if lossWT==2:
        dimWT=-1
    elif lossWT==1:
        dimWT = -1
    elif lossWT == 0:
        dimWT = 0
    elif lossWT == 3:
        dimWT = 0
    scadaTrainDataset = SCADADataset(dataName,prediction=selectData, typeData='Train', windL=windL, predL=predL, wtnum=wtnum,server=server)
    dataloader = torch.utils.data.DataLoader(scadaTrainDataset, batch_size=batch_size,
                                             shuffle=True, num_workers=int(0))
    scadaValDataset = SCADADataset(dataName,prediction=selectData, typeData='Val', windL=windL, predL=predL, wtnum=wtnum,server=server)
    dataloaderVAl = torch.utils.data.DataLoader(scadaValDataset, batch_size=batch_size,
                                                shuffle=True, num_workers=int(0))
    _, wt, p = scadaValDataset.data.shape
    nameWeight = '%s/%s_WT%d_Weight' % (outf, dataName, wtnum)
    parameterDict = {
        'wtnum': wtnum, 'lr': lr, 'weight_decay': weight_decay,'device': device,
        'windL': windL, 'predL': predL, 'outf': outf
        , 'wt': wt, 'p': p, 'CNNoutD': CNNoutD, 'RNNHidden': RNNHidden, 'hiddenC': hiddenC,
        'permu': permu, 'loadWeight': loadWeight,'nameWeight': nameWeight, 'dimWT': dimWT,'dimP': dimP,
    }
    nameConfig = '%s/%s_%s_WT%d_Config' % (outf, name, dataName, wtnum)

    with open(nameConfig, 'wb') as f:
        pickle.dump(parameterDict, f)
    print(nameConfig,'wtnum: ', wtnum, ' lr: ', lr, ' weight_decay: ', weight_decay, ' windL: ', windL, ' predL: ', predL,
          ' batch_size: ', 64, )




    model = CNNAndLSTM(wt, p, CNNoutD, RNNHidden, predL,dimP=dimP,dimWT=dimWT,
                       hiddenC=hiddenC,permu=permu,loadWeight=loadWeight,name=nameWeight).to(device)

    if loadWeight:
        for namePM, param in model.named_parameters():
            if "weightPM" in namePM:
                param.requires_grad = False

    if permu and not loadWeight:
        paras_new = []
        for k, v in dict(model.named_parameters()).items():
            if k == 'weightPML':
                print(k)
                paras_new += [{'params': [v], 'lr': lr * 50}]
            elif k == 'weightPMR':
                print(k)
                paras_new += [{'params': [v], 'lr': lr * 50}]
            else:
                paras_new += [{'params': [v], 'lr': lr}]

        # optimizer_G = torch.optim.Adam(generator.parameters(), lr=lrG, weight_decay=weight_decayG)
        optimizer = torch.optim.Adam(paras_new, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(list(model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20, verbose=False)
    minloss = 100
    model.tou = 1
    for epoch in range(start_epoch, start_epoch + epochs):
        # model.loadWeight=loadWeight
        model.train()
        if tou:
            model.tou *= 0.9
        for i, (x, y) in enumerate(dataloader):

            optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)
            ypred = model(x)
            if permu and not loadWeight:
                wtT = torch.ones(wt).to(device)
                Wtp = torch.ones(p).to(device)
                yp, wL, wr=ypred
                wL1 = torch.sum(wL, dim=0)
                wL2 = torch.sum(wL, dim=1)

                wR1 = torch.sum(wr, dim=0)
                wR2 = torch.sum(wr, dim=1)
                if lossNeed:
                    loss=0
                    if lossP == 2:
                        loss+= F.relu(F.mse_loss(wR2, Wtp)-thresP*p)
                    if lossWT == 2:
                        loss+= F.relu(F.mse_loss(wL1, wtT)-thresWt*wt)
                    if lossP == 3:
                        loss+= F.relu(F.mse_loss(wR1, Wtp)-thresP*p)
                    if lossWT == 3:
                        loss+= F.relu(F.mse_loss(wL2, wtT)-thresWt*wt)
                    loss += F.mse_loss(y, yp)
                    # loss += F.mse_loss(y, yp) + F.mse_loss(wL1, wtT) + \
                    #        F.mse_loss(wL2, wtT) + F.mse_loss(wR1, Wtp) + F.mse_loss(wR2, Wtp)
                else:
                    loss = F.mse_loss(y, yp)

            else:
                loss = F.mse_loss(y, ypred)
            loss.backward()

            optimizer.step()

            if printOut:
                if (i) % 2000 == 0:
                    print('[%d/%d][%d/%d]\tLoss: %.4f\t '
                          % (epoch, start_epoch + epochs, i, len(dataloader), loss))
        if permu and not loadWeight:
            with open(nameWeight, 'wb') as f:
                pickle.dump((wL.cpu().detach().numpy(), wr.cpu().detach().numpy()), f)
        yp, yt = evalCNNLSTM(name, dataName, wtnum, model=model, dataloader=dataloaderVAl,server=server,selectData=selectData)
        lossRMSE = np.sqrt((np.mean((yp - yt) ** 2)))
        if printOut:
            print('VAL loss= ', lossRMSE, )

        scheduler.step(lossRMSE)
        if minloss > (lossRMSE):
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'epoch': epoch}
            modelname = '%s/%s_%s_WT%d' % (outf, name, dataName, wtnum)
            if server:
                torch.save(state, modelname, _use_new_zipfile_serialization=False)
            else:
                torch.save(state, modelname)

            minloss = lossRMSE
            ypT, ytT = evalCNNLSTM(name, dataName, wtnum, model=model, dataloader=None,server=server,selectData=selectData)
            print('Testloss:  ', np.sqrt((np.mean((ypT - ytT) ** 2))))
    if permu:
        yp, yt = evalCNNLSTM(name, dataName, wtnum, model=model, dataloader=None, server=server,selectData=selectData)
        print('Last Epoch Testloss:  ', np.sqrt((np.mean((yp - yt) ** 2))))
        # print(model.weightPML.cpu().detach().numpy())
        # print(model.weightPMR.cpu().detach().numpy())
    return ypT, ytT




def trainMain(model,dataset,wtnum,savename,hyperpara):
    server = hyperpara['server']
    deviceNum = hyperpara['deviceNum']

    try:
        selectData = hyperpara['selectData']
    except:
        selectData = 'WindPower'
    try:
        lossWT = hyperpara['lossWT']
    except:
        lossWT = 2
    try:
        lossP = hyperpara['lossP']
    except:
        lossP=2

    try:
        lossNeed = hyperpara['lossNeed']
    except:
        lossNeed =True
    try:
        printOut = hyperpara['printOut']
    except:
        printOut =False
    try:
        epoch = hyperpara['epoch']
    except:
        epoch = 50
    try:
        epoch= hyperpara['epoch']
        wp= hyperpara['wp']
        weight_decay= hyperpara['weight_decay']
        lr= hyperpara['lr']
    except:
        wp= (50,6)
        weight_decay=0
        lr= 2e-4
    print('model = ', model)
    if model=='PM':
        yp,yt=evalPM(dataset, wtnum, model=None, predL=wp[1], windL=wp[0], server=server)
        print('Testloss:  ', np.sqrt((np.mean((yp - yt) ** 2))))
        return yp,yt
    if model == 'DNN':
        yp, yt=trainDNN(epochs=epoch, wtnum=wtnum, deviceNum=deviceNum, printOut=printOut, lr=lr, weight_decay=weight_decay, dataName=dataset, wp=wp, name=savename,
                 server=server,hidden=hyperpara['hidden'])
        return yp, yt
    if model =='CNNLSTM':

        try:
            hiddenC = int(hyperpara['hiddenC'])
            RNNHidden = int(hyperpara['RNNHidden'])
            permu = hyperpara['permu']
            loadWeight = hyperpara['loadWeight']
        except:
            hiddenC = 64
            RNNHidden = 128
            loadWeight = False
            permu = False

        try:
            tou = hyperpara['tou']
        except:
            tou = True
        try:
            thresP = hyperpara['thresP']
        except:
            thresP = 0
        try:
            thresWt = hyperpara['thresWt']
        except:
            thresWt = 0
        yp, yt=trainCNNLSTM(epochs=epoch, wtnum=wtnum,lossNeed=lossNeed, deviceNum=deviceNum,printOut=printOut, lr=lr, weight_decay=weight_decay, dataName=dataset, wp=wp, name=savename,
                 server=server, hiddenC=hiddenC, RNNHidden=RNNHidden,permu=permu,loadWeight=loadWeight,lossWT=lossWT,lossP=lossP,
                            thresWt=thresWt,thresP=thresP,selectData=selectData,tou=tou)
        return yp, yt



def evalMain(model,dataset,wtnum,savename,hyperpara,noise=False):

    wp= hyperpara['wp']
    server= hyperpara['server']

    print('model = ', model)
    if model=='PM':
        yp,yt=evalPM(dataset, wtnum, model=None, predL=wp[1], windL=wp[0], server=server,noise=noise)
        print('Testloss:  ', np.sqrt((np.mean((yp - yt) ** 2))))
        return yp,yt
    if model == 'DNN':
        yp, yt=evalDNN(savename,dataset,wtnum,model=None,dataloader=None,server=server,noise=noise)
        return yp, yt
    if model =='CNNLSTM':
        yp, yt=evalCNNLSTM(savename, dataset, wtnum, model=None, dataloader=None, server=server, noise=noise)
        return yp, yt
    if model =='PDCNN':
        yp, yt=evalPDCNN(savename, dataset, wtnum, model=None, dataloader=None, server=server, noise=noise)
        return yp, yt
    if model =='MARMA':
        yp, yt = evalMARMA(savename, dataset, wtnum, model=None, dataloader=None, server=server, noise=noise)
        return yp, yt
    if model =='DACRN-KM-V1':
        yp, yt=evalKshape(savename,dataset,wtnum,model=None,dataloader=None,server=server,noise=noise)
        return yp, yt
    if model =='DACRN-KM-V2':
        yp, yt = evalKshape2(savename, dataset, wtnum, model=None, dataloader=None, server=server, noise=noise)
        return yp, yt
    if model=='DTSF':
        yNp = np.zeros([1, hyperpara['wp'][1]])
        yReal = np.zeros([1, hyperpara['wp'][1]])
        for i in range(10):
            yp, yt = evalDTSF(dataset, m=hyperpara['m'], wtnum=wtnum, model=None, predL=hyperpara['wp'][1], windL=hyperpara['w'],
                              server=server)
            yNp = np.vstack((yNp, yp ))
            yReal = np.vstack((yReal, yt))

        return yNp[1:, :], yReal[1:, :]
    if model =='RDCNN':
        yp, yt= evalRDCNN(savename, dataset, wtnum, model=None, dataloader=None, server=server, noise=noise)
        return yp, yt
    if model =='EWTLSTM':
        yp, yt=evalEWTLSTM(savename, dataset, wtnum, model=None, dataloader=None, server=server, noise=noise)
        return yp, yt
    if model == 'RSDAE':
        yp, yt=evalRSDAE(savename, dataset, wtnum, model=None, dataloader=None, server=server, noise=noise)
        return yp, yt
    if model == 'DeepHyp':
        yp, yt=evalDeepHyper(savename, dataset, wtnum, model=None, dataloader=None, server=server, noise=noise)
        return yp, yt

#,'ZM3-5','ZM7-9'


datasettoWt={

    'ZM3-5': 11,
    'ZM7-9':11,
    'ZM11-1': 11,
    '10s': 10,
    'Borne':4,
    'K1': 15,

}
import copy
def OutPutHyperPara(model,hyperpara,dataset='ZM'):
    if model=='PM':
        return [hyperpara]
    if model == 'DNN':
        hyperparaOut=[]
        for i,hidden in enumerate([32,64,128]):
            hyperparaOut.append(copy.deepcopy(hyperpara))
            hyperparaOut[i]['hidden']=int(hidden)
        return hyperparaOut
    if model =='CNNLSTM':
        hyperparaOut = []
        for i in range(6):
            hiddenC=np.random.choice([8,16,32,64])
            RNNHidden=np.random.choice([16,32,64,128])
            hyperparaOut.append(copy.deepcopy(hyperpara))
            hyperparaOut[i]['hiddenC'] = int(hiddenC)
            hyperparaOut[i]['RNNHidden'] = int(RNNHidden)
        return hyperparaOut
    if model =='PDCNN':
        hyperparaOut = []
        for i, hiddenC in enumerate([8,16,32,64]):
            hyperparaOut.append(copy.deepcopy(hyperpara))
            hyperparaOut[i]['hiddenC'] = int(hiddenC)
        return hyperparaOut

    if model =='MARMA':
        hyperparaOut = []
        for i, p in enumerate([8, 16, 32]):
            for j, q in enumerate([8, 16, 32]):
                hyperparaOut.append(copy.deepcopy(hyperpara))
                hyperparaOut[i*3+j]['pq'] = (p,q)

        return hyperparaOut

    if model =='DACRN-KM-V1':
        hyperparaOut = []
        for i in range(20):
            K = np.random.choice([3,6,9])
            mem_dim = np.random.choice([1000, 2000, 4000])
            RNNHidden = np.random.choice([16,32,64,128])
            hiddenC = np.random.choice([8, 16, 32, 64])
            weight_decay = np.random.choice([0,1e-4,1e-5])

            hyperparaOut.append(copy.deepcopy(hyperpara))
            hyperparaOut[i]['hiddenC'] = int(hiddenC)
            hyperparaOut[i]['RNNHidden'] = int(RNNHidden)
            hyperparaOut[i]['weight_decay'] = weight_decay
            hyperparaOut[i]['mem_dim'] = int(mem_dim)
            hyperparaOut[i]['K'] = int(K)
        return hyperparaOut

    if model =='DACRN-KM-V2':
        hyperparaOut = []
        for i in range(20):
            K = np.random.choice([3, 6, 9])
            mem_dim = np.random.choice([1000, 2000, 4000])
            RNNHidden = np.random.choice([16, 32, 64, 128])
            hiddenC = np.random.choice([8, 16, 32, 64])
            weight_decay = np.random.choice([0, 1e-4, 1e-5])

            hyperparaOut.append(copy.deepcopy(hyperpara))
            hyperparaOut[i]['hiddenC'] = int(hiddenC)
            hyperparaOut[i]['RNNHidden'] = int(RNNHidden)
            hyperparaOut[i]['weight_decay'] = weight_decay
            hyperparaOut[i]['mem_dim'] = int(mem_dim)
            hyperparaOut[i]['K'] = int(K)
        return hyperparaOut
    if model=='DTSF':
        hyperpara['m']=[100, 300, 500, 200, 50]
        hyperparaOut=[]
        for i, w in enumerate([4,5,6,7]):
            hyperparaOut.append(copy.deepcopy(hyperpara))
            hyperparaOut[i]['w'] = w
        return hyperparaOut
    if model =='RDCNN':

        hyperparaOut = []
        for i in range(20):
            HL = np.random.choice([1,2])
            VL = np.random.choice([1,2,3])
            covNumber = np.random.choice([1,2,3])
            channel = np.random.choice([1,2,3])
            if dataset=='Borne':
                ks = np.random.choice([2, 3])
            else:
                ks = np.random.choice([2,3,4])
            dropout = np.random.choice([0,0.2,0.5])
            hiddenLayerNumber = np.random.choice([1,2,3])

            hyperparaOut.append(copy.deepcopy(hyperpara))
            hyperparaOut[i]['HL'] = int(HL)
            hyperparaOut[i]['VL'] = int(VL)
            hyperparaOut[i]['covNumber'] = int(covNumber)
            hyperparaOut[i]['channel'] = int(channel)
            hyperparaOut[i]['ks'] = int(ks)
            hyperparaOut[i]['dropout'] =dropout
            hyperparaOut[i]['hiddenLayerNumber'] = int(hiddenLayerNumber)
        return hyperparaOut

    if model =='EWTLSTM':

        hyperparaOut = []
        for i, hidden in enumerate([16,32,64]):
            for j, layer in enumerate([1, 2, 3]):
                hyperparaOut.append(copy.deepcopy(hyperpara))
                hyperparaOut[i * 3 + j]['hidden'] = hidden
                hyperparaOut[i * 3 + j]['layer'] = layer
        return hyperparaOut

    if model == 'RSDAE':
        hyperparaOut = []
        for i, hidden in enumerate([16, 32,64]):
            for j, layer in enumerate([1, 2, 3]):
                hyperparaOut.append(copy.deepcopy(hyperpara))
                hyperparaOut[i * 3 + j]['hidden'] = hidden
                hyperparaOut[i * 3 + j]['layer'] = layer
        return hyperparaOut
    if model == 'DeepHyp':
        hyperpara['epoch']=150
        hyperparaOut = []
        for i, hidden in enumerate([16, 32,64]):
            for j, layer in enumerate([1, 2, 3]):
                hyperparaOut.append(copy.deepcopy(hyperpara))
                hyperparaOut[i * 3 + j]['n_hid'] = hidden
                hyperparaOut[i * 3 + j]['layer'] = layer
        return hyperparaOut
if __name__=='__main__':
    # trainRSDAE(epochs=100, wtnum=0, weight_decay=0, printOut=True,
    #            dataName='ZM3-5', wp=(25, 5), name='RSDAE', server=False, hideen=64, layer=3)
    #rmse K1 0 MARMA  1.845613
    #
    # trainRDCNN(epochs=100, wtnum=0, HL=1, VL=1, covNumber=1, channel=3, ks=4, dropout=0.5, hiddenLayerNumber=2,
    #            printOut=True, dataName='K1', wp=(47, 6), name='RDCNN', server=False)

    # trainEWTLSTM(epochs=200, wtnum=0, hidden=100, layer=2, numberewt=17,
    #              printOut=True, dataName='K1', wp=(50, 6), name='EWTLSTM', server=False)

    # trainRBM(epochs=6006, wtnum=0, n_hid=100, layer=2, numberewt=17,
    #          printOut=True, dataName='K1', wp=(50, 6), name='RBM', server=False)
    #
    # trainDeepHy(epochs=100, wtnum=0, n_hid=10,layer=2,n_out=4,num_clusters=3,weight_decay = 0,lr = 2e-4,
    #            printOut=True, dataName='K1', wp=(50, 6), name='DeepHy',server=False)

    trainKshape2(server=False, weight_decay=1e-6, alpha=2e-9, batch_size=64, lr=2e-4, mem=True,AL=True,
                 rNNHidden=64, printOut=False,
                 K=3, wp=(50, 6), name='KshapeV2', epochs=100, dataName=r'Borne', wtnum=0, modelV=2)
























































