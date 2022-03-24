#Author:ike yang
from model import CNNAndLSTM
from utlize import SCADADataset
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def trainCNNLSTM(epochs=200, wtnum=0, deviceNum=0,lr = 2e-4,weight_decay = 0,
                 hiddenC=32,RNNHidden=128,thresWt=0,thresP=0,tou=True):
    torch.cuda.set_device(deviceNum)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    windL=50#historical length
    predL=6#prediction length

    batch_size = 64

    CNNoutD, RNNHidden, RNNOutD = ( RNNHidden, RNNHidden, predL)

    scadaTrainDataset = SCADADataset(windL=windL, predL=predL, wtnum=wtnum)
    dataloader = torch.utils.data.DataLoader(scadaTrainDataset, batch_size=batch_size,
                                             shuffle=True, num_workers=int(16))

    _, wt, p = scadaTrainDataset.data.shape


    model = CNNAndLSTM(wt, p, CNNoutD, RNNHidden, predL,hiddenC=hiddenC).to(device)



    #speed up convergence
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

    optimizer = torch.optim.Adam(paras_new, weight_decay=weight_decay)


    for epoch in range(epochs):
        model.train()
        if tou:
            model.tou *= 0.9
        for i, (x, y) in enumerate(dataloader):

            optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)
            ypred = model(x)

            ###ROC loss
            wtT = torch.ones(wt).to(device)
            Wtp = torch.ones(p).to(device)
            yp, wL, wr=ypred
            wL1 = torch.sum(wL, dim=0)
            wR2 = torch.sum(wr, dim=1)
            loss =F.relu(F.mse_loss(wL1, wtT)-thresWt*wt)+F.relu(F.mse_loss(wR2, Wtp)-thresP*p)


            #task loss
            loss += F.mse_loss(y, yp)
            loss.backward()

            optimizer.step()



if __name__ =='__main__':
    trainCNNLSTM( thresWt=1, thresP=1, tou=True)

















































