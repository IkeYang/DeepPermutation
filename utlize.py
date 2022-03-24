# Author:ike yang
from torch.utils.data import Dataset
import pickle
import torch
import numpy as np
import numpy as np
import pickle
from  tslearn.clustering import KShape
from  tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.generators import random_walks
from matplotlib import pyplot as plt
import ewtpy

serverDict={

                'ZM3-5': r'/home/daisy001/mountdir/LuoxiaoYANG/Data/processedData/%s' % ('3-5'),
                'ZM7-9': r'/home/daisy001/mountdir/LuoxiaoYANG/Data/processedData/%s' % ('7-9'),
                'ZM11-1': r'/home/daisy001/mountdir/LuoxiaoYANG/Data/processedData/%s' % ('3-5'),
                '10s': r'/home/daisy001/mountdir/LuoxiaoYANG/Data/10sData2010/wt',
                'Borne': r'/home/daisy001/mountdir/LuoxiaoYANG/Data/processedData/Borne17_',
                'K1': '/home/daisy001/mountdir/LuoxiaoYANG/Data/processedData/wf1_data_',
            }


mydict={

                'ZM3-5': r'D:\YANG Luoxiao\Data\10台风机数据（20160205-20170605）\pickleF\%s' % ('3-5'),
                'ZM7-9': r'D:\YANG Luoxiao\Data\10台风机数据（20160205-20170605）\pickleF\%s' % ('7-9'),
                'ZM11-1': r'D:\YANG Luoxiao\Data\10台风机数据（20160205-20170605）\pickleF\%s' % ('3-5'),
                '10s': r'D:\YANG Luoxiao\Data\WT10s\pickle\wt',
                'Borne': r'D:\YANG Luoxiao\Data\basluona\Beslona17',
                'K1': 'D:\YANG Luoxiao\Data\wp_pred_review\wf1\wf1_data_',
            }


var=0.1

def channelArguX(data):
    #data [t,wt,p]
    t,wt,p=data.shape
    dataOut=np.zeros((t,2,wt,p))
    dataOut[:,0,:,:]=data

    dataOut[1:,1,:,:]=data[1:,:,:]-data[:t-1,:,:]

    # dataOut[1:,2,:,:]=data[1:,:,:]/(data[:t-1,:,:]+1e-6)
    # dataOut[0, 2, :, :]=1
    return dataOut
class SCADADataset(Dataset):
    # +jf33N train together 正常baseline的数据集 没有额外的数据预处理
    def __init__(self, dataName, Tpyename, windL, predL, wtnum,server=False,onlywindspee=False,noise=False):
        if server:
            dataFileDict =serverDict
        else:
            dataFileDict =mydict
        dataFile=dataFileDict[dataName]
        if '-' in dataName:
            self.dataType = 'ZM'
        elif 'NewYork' in dataFile:
            self.dataType = 'NY'
        elif 'WT10s' in dataFile:
            self.dataType = '10s'
        else:
            self.dataType = 'else'
        if self.dataType == 'NY':
            filename = dataFile + r'\\' + Tpyename
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
        elif self.dataType == '10s':
            filename = dataFile + Tpyename
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)[:,:,[13, 17, 30, 36, 59, 60]]
        elif self.dataType == 'ZM':
            filename = dataFile + Tpyename
            with open(filename, 'rb') as f:
                self.dataX, self.dataY = pickle.load(f)
                self.data = self.dataX[:,-1,:, :]
        else:
            filename = dataFile + Tpyename
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
        self.predL = predL
        self.onlywindspee = onlywindspee
        self.windL = windL
        self.wtnum = wtnum
        self.noise = noise
        if dataName=='K1':
            self.samlpeInterval=3
            self.data=self.data[:int(self.data.shape[0]/self.samlpeInterval)*self.samlpeInterval,:,:].reshape((-1,3,self.data.shape[1],self.data.shape[2]))
            self.data = np.mean(self.data,axis=1)[:,:,[1, 3, 8]]
    def __len__(self):
        if self.dataType == 'kkk':
            return self.dataX.shape[0]

        else:
            return self.data.shape[0] - self.windL - self.predL

    def __getitem__(self, idx):
        if self.dataType == 'kkk':

            x = self.dataX[idx, :, :, :]
            if self.onlywindspee:
                x=x[:,self.wtnum,0]
            if self.noise:
                x[:, self.wtnum, 0] = x[:, self.wtnum, 0]+np.random.randn(x.shape[0])*(var*x[:, self.wtnum, 0])
            # print(np.sum(np.isnan(x)))
            x = torch.from_numpy(x).float()

            y = self.dataY[idx, :, self.wtnum, 0]
            y = torch.from_numpy(y).float()
            return x, y
        elif self.dataType == '10s':
            x = self.data[idx:idx + self.windL, :, :]  # 15 wt 6 p
            if self.onlywindspee:
                x=x[:,self.wtnum,0]
            if self.noise:
                x[:, self.wtnum, 0] = x[:, self.wtnum, 0]+np.random.randn(x.shape[0])*(var*x[:, self.wtnum, 0])
            x = torch.from_numpy(x).float()

            y = np.copy(self.data[idx + self.windL:idx + self.windL + self.predL, self.wtnum, 0])
            y = torch.from_numpy(y).float()
            return x, y
        else:
            x = np.copy(self.data[idx:idx + self.windL, :, :])
            if self.onlywindspee:
                x=x[:,self.wtnum,0]
            if self.noise:
                x[:, self.wtnum, 0] = x[:, self.wtnum, 0]+np.random.randn(x.shape[0])*(var*x[:, self.wtnum, 0])
            # print(np.sum(np.isnan(x)))
            x = torch.from_numpy(x).float()

            y = np.copy(self.data[idx + self.windL:idx + self.windL + self.predL, self.wtnum, 0])
            y = torch.from_numpy(y).float()
            return x, y


# 这个方法注意要对时间戳
class MARMASCADADataset(Dataset):
    # +jf33N train together
    def __init__(self, dataName, Typename, p, q, predL, wtnum,server=False,noise=False):
        if server:
            dataFileDict = serverDict
        else:
            dataFileDict = mydict

        dataFile = dataFileDict[dataName]
        if '-' in dataName:
            self.dataType = 'ZM'
        elif 'NewYork' in dataFile:
            self.dataType = 'NY'
        elif 'WT10s' in dataFile:
            self.dataType = '10s'
        else:
            self.dataType = 'else'
        if self.dataType == 'ZM':
            filename = dataFile + Typename
            with open(filename, 'rb') as f:
                self.dataX, self.dataY = pickle.load(f)
                self.data = self.dataX[:, -1, :, :]
            self.predL = predL
            self._p = p
            self._q = q
            self.wtnum = wtnum
            self.maxpq = max(self._p, self._q)
        elif self.dataType == 'NY':
            filename = dataFile + r'\\' + Typename
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
            self.predL = predL
            self._p = p
            self._q = q
            self.wtnum = wtnum
            self.maxpq = max(self._p, self._q)
        else:
            filename = dataFile + Typename
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)

            self.predL = predL
            self._p = p
            self._q = q
            self.wtnum = wtnum

            self.maxpq = max(self._p, self._q)
        if dataName == 'K1':
            self.samlpeInterval = 3
            self.data = self.data[:int(self.data.shape[0] / self.samlpeInterval) * self.samlpeInterval, :, :].reshape(
                (-1, 3, self.data.shape[1], self.data.shape[2]))
            self.data = np.mean(self.data, axis=1)[:,:,[1, 3, 8]]
        self.noise = noise
    def __len__(self):
        if self.dataType == 'kkk':
            return self.dataX.shape[0]

        else:
            return self.data.shape[0] - self.predL - self.maxpq

    def __getitem__(self, idx):

        if self.dataType == 'kkk':
            xy = np.copy(self.dataX[idx, -self._p:, :, 0])
            if self.noise:
                xy[:, self.wtnum] = xy[:, self.wtnum]+np.random.randn(xy.shape[0])*(var*xy[:, self.wtnum])
            xs = np.copy(self.dataX[idx, - self._q:, :, 0])
            mu = np.mean(xs, axis=0)
            xs = xs - mu
            xs = xs.reshape((-1, 1))
            xy = xy.reshape((-1, 1))
            x = np.vstack((xy, xs)).flatten()
            # if x.shape[0]==40:
            #     pass
            # else:
            #     print(x.shape)
            # print(np.sum(np.isnan(x)))
            x = torch.from_numpy(x).float()

            y = np.copy(self.dataY[idx, :, self.wtnum, 0])
            y = torch.from_numpy(y).float()
            return x, y

        else:
            if self.dataType == '10s':
                wsattr = 13
            else:
                wsattr = 0
            # print(self._q)
            xy = np.copy(self.data[idx + self.maxpq - self._p:idx + self.maxpq, :, wsattr])
            if self.noise:
                xy[:, self.wtnum] = xy[:, self.wtnum] + np.random.randn(xy.shape[0]) * (var * xy[:, self.wtnum])
            xs = np.copy(self.data[idx + self.maxpq - self._q:idx + self.maxpq, :, wsattr])
            mu = np.mean(xs, axis=0)
            xs = xs - mu
            xs = xs.reshape((-1, 1))
            xy = xy.reshape((-1, 1))
            x = np.vstack((xy, xs)).flatten()
            # if x.shape[0]==40:
            #     pass
            # else:
            #     print(x.shape)
            # print(np.sum(np.isnan(x)))
            x = torch.from_numpy(x).float()

            y = np.copy(self.data[idx + self.maxpq:idx + self.maxpq + self.predL, self.wtnum, wsattr])
            y = torch.from_numpy(y).float()
            return x, y


class k0():
    def __init__(self):
        self.k=0
    def predict(self,x):
        n=x.shape[0]
        c=np.zeros((n))
        return c

def trainKshapeC(k,wl=50,wtnum=0,dataName='K1',server=False,noise=False):
    if server:
        dataFileDict = serverDict
    else:
        dataFileDict = mydict
    if server:
        outf = r'/home/daisy001/mountdir/LuoxiaoYANG/Model/WindPrediction'
    else:
        outf = r'D:\YANG Luoxiao\Model\WindSpeed'
    
    
    
    
    dataFile = dataFileDict[dataName]


    if k == 1:
        ks = k0()
        with open( outf + r'\\' + '%s_%d_%d_K%d' % (dataName,wl,wtnum,k),'wb') as f:
            pickle.dump(ks,f)
    else:
        if 'ZM' in dataName:
            with open(dataFile + 'Train', 'rb') as f:
                data, _ = pickle.load(f)
                data = data[:, -wl:, wtnum, 0].reshape((-1, wl, 1)) * 20.76
        else:
            with open(dataFile+'Train', 'rb') as f:
                data = pickle.load(f)
            if dataName == 'K1':
                samlpeInterval = 3
                data = data[:int(data.shape[0] / samlpeInterval) * samlpeInterval, :,
                            :].reshape((-1, 3, data.shape[1], data.shape[2]))
                data = np.mean(data, axis=1)[:, :, [1, 3, 8]]
            elif dataName == '10s':
                data=data[:, :, [13,17,30,36,59,60]]
            data=data[:int(data.shape[0]/wl)*wl,wtnum,0].reshape((-1,wl,1))*20.76

        np.random.shuffle(data)
        X = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(data)
        ks = KShape(n_clusters=k, n_init=2,max_iter=5000).fit(X)
        with open(outf + r'\\' + '%s_%d_%d_K%d' % (dataName, wl, wtnum, k), 'wb') as f:
            pickle.dump(ks,f)

    return ks





class KshapeSCADADataset(Dataset):
    # +jf33N train together
    def __init__(self, dataName, Typename,wtnum,windL=50,predL=6,K=3,server=False,noise=False):
        if server:
            dataFileDict = serverDict
        else:
            dataFileDict = mydict
        if server:
            outf = r'/home/daisy001/mountdir/LuoxiaoYANG/Model/WindPrediction'
        else:
            outf = r'D:\YANG Luoxiao\Model\WindSpeed'

        dataFile = dataFileDict[dataName]

        filename = dataFile + Typename
        with open(filename, 'rb') as f:
            self.data = pickle.load(f)

        self.windL=windL
        self.noise=noise
        self.predL=predL
        self.wtnum=wtnum
        self.dataName=dataName
        try:
            with open(outf + r'\\' + '%s_%d_%d_K%d' % (dataName, windL, wtnum, K), 'rb') as f:
                self.ks=pickle.load(f)
        except:
            self.ks=trainKshapeC(K, wl=windL, wtnum=wtnum, dataName=dataName,server=server)

        if dataName=='K1':
            self.samlpeInterval=3
            self.data=self.data[:int(self.data.shape[0]/self.samlpeInterval)*self.samlpeInterval,:,:].reshape((-1,self.samlpeInterval,self.data.shape[1],self.data.shape[2]))
            self.data = np.mean(self.data,axis=1)[:,:,[1, 3, 8]]

        if dataName=='10s':
            self.data=self.data[:,:,[13, 17, 30, 36, 59, 60]]

        if 'ZM' in dataName:
            self.dataX, self.dataY = self.data
            self.data = self.dataX[:, -1, :, :]

        self.length = self.data.shape[0] - self.windL - self.predL

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if 'kkk' in self.dataName:
            x = np.copy( self.dataX[idx, :, :, :])
            if self.noise:
                x[:, self.wtnum, 0]  = x[:, self.wtnum, 0] + np.random.randn(x.shape[0]) * (var  * x[:, self.wtnum, 0])
            # x [wl,15,6]
            x[:, :, 0] = x[:, :, 0] * 20.76
            mu = np.mean(x[:, self.wtnum, 0], axis=0)
            std = np.std(x[:, self.wtnum, 0], axis=0)
            if std == 0 and mu != 0:
                print(mu)
            x[:, :, 0] = (x[:, :, 0] - mu) / (std + 1e-20)

            c = self.ks.predict(x[:, self.wtnum, 0].reshape((1, -1, 1)))
            x = channelArguX(x)
            x = torch.from_numpy(x).float()
            mu = mu.reshape((1, 1))
            mu = torch.from_numpy(mu).float()
            std = std.reshape((1, 1))
            std = torch.from_numpy(std).float()
            y = np.copy(self.dataY[idx, :, self.wtnum, 0])
            y = torch.from_numpy(y).float()
            return x, y, mu, std, c.reshape((1, 1))

            pass
        else:

            x = np.copy(self.data[idx:idx+ self.windL, :, :])
            if self.noise:
                x[:, self.wtnum, 0]  = x[:, self.wtnum, 0] + np.random.randn(x.shape[0]) *(var * x[:, self.wtnum, 0])
            # x [wl,15,6]
            x[:, :, 0] = x[:, :, 0] * 20.76
            mu = np.mean(x[:, self.wtnum, 0], axis=0)
            std = np.std(x[:, self.wtnum, 0], axis=0)
            if std == 0 and mu != 0:
                print(mu)
            x[:, :, 0] = (x[:, :, 0] - mu) / (std + 1e-20)

            c = self.ks.predict(x[:, self.wtnum, 0].reshape((1, -1, 1)))
            x = channelArguX(x)
            x = torch.from_numpy(x).float()
            mu = mu.reshape((1, 1))
            mu = torch.from_numpy(mu).float()
            std = std.reshape((1, 1))
            std = torch.from_numpy(std).float()
            y = np.copy(self.data[idx+self.windL:idx+self.windL+self.predL, self.wtnum, 0])
            y = torch.from_numpy(y).float()
            return x, y, mu, std, c.reshape((1, 1))


class KshapeSCADADataset2(Dataset):
    # +jf33N train together
    def __init__(self, dataName, Typename, wtnum, windL=50, predL=6, K=3,server=False,noise=False):
        if server:
            dataFileDict = serverDict
        else:
            dataFileDict = mydict
        if server:
            outf = r'/home/daisy001/mountdir/LuoxiaoYANG/Model/WindPrediction'
        else:
            outf = r'D:\YANG Luoxiao\Model\WindSpeed'

        dataFile = dataFileDict[dataName]

        filename = dataFile + Typename
        with open(filename, 'rb') as f:
            self.data = pickle.load(f)

        self.windL = windL
        self.noise = noise
        self.predL = predL
        self.wtnum = wtnum
        self.dataName = dataName
        try:
            with open(outf + r'\\' + '%s_%d_%d_K%d' % (dataName, windL, wtnum, K), 'rb') as f:
                self.ks = pickle.load(f)
        except:
            self.ks = trainKshapeC(K, wl=windL, wtnum=wtnum, dataName=dataName,server=server)

        if dataName == 'K1':
            self.samlpeInterval = 3
            self.data = self.data[:int(self.data.shape[0] / self.samlpeInterval) * self.samlpeInterval, :, :].reshape(
                (-1, self.samlpeInterval, self.data.shape[1], self.data.shape[2]))
            self.data = np.mean(self.data, axis=1)[:, :, [1, 3, 8]]

        if dataName == '10s':
            self.data = self.data[:, :, [13, 17, 30, 36, 59, 60]]

        if 'ZM' in dataName:
            self.dataX, self.dataY = self.data
            self.data = self.dataX[:, -1, :, :]

        self.length = self.data.shape[0] - self.windL - self.predL
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if 'kkk' in self.dataName:
            x = np.copy(self.dataX[idx, :, :, :])
            if self.noise:
                x [:, self.wtnum, 0]= x[:, self.wtnum, 0] + np.random.randn(x.shape[0]) * (var * x[:, self.wtnum, 0])
            # x [wl,15,6]
            x[:, :, 0] = x[:, :, 0] * 20.76
            mu = np.mean(x[:, self.wtnum, 0], axis=0)
            std = np.std(x[:, self.wtnum, 0], axis=0)
            if std == 0 and mu != 0:
                print(mu)
            x[:, :, 0] = (x[:, :, 0] - mu) / (std + 1e-20)

            c = self.ks.predict(x[:, self.wtnum, 0].reshape((1, -1, 1)))

            x = torch.from_numpy(np.copy(self.dataX[idx, :, :, :])).float()
            y = np.copy(self.dataY[idx, :, self.wtnum, 0])
            y = torch.from_numpy(y).float()
            return x, y, c.reshape((1, 1))
        else:
            x = np.copy(self.data[idx:idx + self.windL, :, :])
            if self.noise:
                x [:, self.wtnum, 0]= x[:, self.wtnum, 0] + np.random.randn(x.shape[0]) * (var * x[:, self.wtnum, 0])
            # x [wl,15,6]
            x[:, :, 0] = x[:, :, 0] * 20.76
            mu = np.mean(x[:, self.wtnum, 0], axis=0)
            std = np.std(x[:, self.wtnum, 0], axis=0)
            if std == 0 and mu != 0:
                print(mu)
            x[:, :, 0] = (x[:, :, 0] - mu) / (std + 1e-20)

            c = self.ks.predict(x[:, self.wtnum, 0].reshape((1, -1, 1)))

            x = torch.from_numpy(np.copy(self.data[idx:idx + self.windL, :, :])).float()  # wl, wt ,p
            y = np.copy(self.data[idx + self.windL:idx + self.windL + self.predL, self.wtnum, 0])
            y = torch.from_numpy(y).float()
            return x, y, c.reshape((1, 1))

#
class RSDAEDataset(Dataset):
    # +jf33N train together 正常baseline的数据集 没有额外的数据预处理
    def __init__(self, dataName, Tpyename, windL, predL, wtnum,server=False,prediction=True,noise=False):
        if server:
            dataFileDict = serverDict
        else:
            dataFileDict = mydict
        if server:
            outf = r'/home/daisy001/mountdir/LuoxiaoYANG/Model/WindPrediction'
        else:
            outf = r'D:\YANG Luoxiao\Model\WindSpeed'

        dataFile=dataFileDict[dataName]
        if '-' in dataName:
            self.dataType = 'ZM'
        elif 'NewYork' in dataFile:
            self.dataType = 'NY'
        elif 'WT10s' in dataFile:
            self.dataType = '10s'
        else:
            self.dataType = 'else'
        if self.dataType == 'NY':
            filename = dataFile + r'\\' + Tpyename
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
        elif self.dataType == '10s':
            filename = dataFile + Tpyename
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)[:,:,[13, 17, 30, 36, 59, 60]]
        elif self.dataType == 'ZM':
            filename = dataFile + Tpyename
            with open(filename, 'rb') as f:
                self.dataX, self.dataY = pickle.load(f)
                self.data = self.dataX[:, -1, :, :]
        else:
            filename = dataFile + Tpyename
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
        self.predL = predL
        self.windL = windL
        self.wtnum = wtnum
        self.prediction = prediction
        self.noise = noise
        if dataName=='K1':
            self.samlpeInterval=3
            self.data=self.data[:int(self.data.shape[0]/self.samlpeInterval)*self.samlpeInterval,:,:].reshape((-1,3,self.data.shape[1],self.data.shape[2]))
            self.data = np.mean(self.data,axis=1)[:,:,[1, 3, 8]]
        if self.noise:
            if self.dataType == 'kkk':
                self.dataX[:,:,self.wtnum,0]+=np.random.randn(self.dataX.shape[0],self.dataX.shape[1]) * (var * self.dataX[:,:,self.wtnum,0])

            else:
                self.data[ :, self.wtnum, 0] += np.random.randn(self.data.shape[0]) * (
                            var * self.data[:, self.wtnum, 0])

    def __len__(self):
        if self.dataType == 'kkk':
            return self.dataX.shape[0]

        else:
            return self.data.shape[0] - self.windL - self.predL

    def __getitem__(self, idx):
        if self.dataType == 'kkk':
            x=np.zeros([self.windL+self.windL-1,])
            x[:self.windL] = np.copy(self.dataX[idx, - self.windL:, self.wtnum, 0])
            noise=np.random.randn(self.windL+self.windL-1)/10e-2
            x[self.windL:]= np.copy(self.dataX[idx, - self.windL+1:, self.wtnum, 0])-np.copy(self.dataX[idx, - self.windL:-1, self.wtnum, 0])
            if not self.prediction:
                x+=noise
                x = torch.from_numpy(x).float()
                return x

            x = torch.from_numpy(x).float()

            y = self.dataY[idx, :, self.wtnum, 0]
            y = torch.from_numpy(y).float()
            return x, y
        else:
            x = np.zeros([self.windL + self.windL - 1, ])
            x[:self.windL] = np.copy(self.data[idx:idx + self.windL, self.wtnum, 0])
            noise = np.random.randn(self.windL + self.windL - 1) / 10e-2
            x[self.windL:] = np.copy(self.data[idx+1:idx + self.windL, self.wtnum, 0]) - np.copy(
                self.data[idx:idx + self.windL-1, self.wtnum, 0])
            if not self.prediction:
                x += noise
                x = torch.from_numpy(x).float()
                return x
            # print(np.sum(np.isnan(x)))
            x = torch.from_numpy(x).float()

            y = np.copy(self.data[idx + self.windL:idx + self.windL + self.predL, self.wtnum, 0])
            y = torch.from_numpy(y).float()
            return x, y


from sklearn.linear_model import LinearRegression
class DTSF():
    def __init__(self,dataName, Tpyename, windL, predL, wtnum,m,server=False):

        if server:
            dataFileDict = serverDict
        else:
            dataFileDict = mydict
        if server:
            outf = r'/home/daisy001/mountdir/LuoxiaoYANG/Model/WindPrediction'
        else:
            outf = r'D:\YANG Luoxiao\Model\WindSpeed'

        dataFile=dataFileDict[dataName]
        if '-' in dataName:
            self.dataType = 'ZM'
        elif 'NewYork' in dataFile:
            self.dataType = 'NY'
        elif 'WT10s' in dataFile:
            self.dataType = '10s'
        else:
            self.dataType = 'else'


        if self.dataType == 'NY':
            filename = dataFile + r'\\' + Tpyename
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
        elif self.dataType == '10s':
            filename = dataFile + Tpyename
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)[:,:,[13, 17, 30, 36, 59, 60]]
        elif self.dataType == 'ZM':
            filename = dataFile + Tpyename
            with open(filename, 'rb') as f:
                self.dataX, self.dataY = pickle.load(f)
                self.data=self.dataY[:,-1,:,:]
        else:
            filename = dataFile + Tpyename
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)

        if dataName == 'K1':
            self.samlpeInterval = 3
            self.data = self.data[:int(self.data.shape[0] / self.samlpeInterval) * self.samlpeInterval, :, :].reshape(
                (-1, 3, self.data.shape[1], self.data.shape[2]))
            self.data = np.mean(self.data, axis=1)[:, :, [1, 3, 8]]

        self.data=self.data[:,wtnum,0].flatten()

        self.m=m
        self.windL=windL
        self.predL=predL


    def prediction(self,x,y=None):
        yreal=y.reshape((-1,1))
        ###x (windL,)
        windL=x.shape[0]
        x=x.reshape((-1,1))
        resR=[]
        number=5000
        
        if self.data.shape[0]-windL-self.predL<number:
            self.data2=self.data
        else:
            start = np.random.randint(0, self.data.shape[0] - windL - self.predL - number)
            self.data2 = self.data[start:start + number]
        for i in range(self.data2.shape[0]-windL-self.predL):
            dataX=np.copy(self.data2[i:i+windL]).reshape((-1,1))
            model = LinearRegression()
            model.fit(dataX, x)
            R2 = model.score(dataX, x)
            resR.append(R2)
        resR=np.array(resR)
        try:
            lengthM=len(self.m)
            y = np.zeros([lengthM, self.predL])
            for k,m in enumerate(self.m):
                ind = np.argsort(-resR)[:m]
                clf = []
                ym=np.zeros([m,self.predL])
                for j, i in enumerate(ind):
                    dataX = np.copy(self.data2[i:i + windL]).reshape((-1, 1))
                    dataY = np.copy(self.data2[i:i + windL]).reshape((1, -1))
                    model = LinearRegression()
                    model.fit(dataX, x)
                    ym[j, :] = model.predict(self.data2[i + windL:i + windL + self.predL].reshape((-1, 1))).flatten()
                    clf.append(model)
                y[k,:] =np.mean(ym, axis=0)
            return np.mean(y, axis=0)
        except:
            ind=np.argsort(-resR)[:self.m]
            clf=[]
            y=np.zeros([self.m,self.predL])
            for j,i in enumerate(ind):
                dataX = np.copy(self.data2[i:i + windL]).reshape((-1, 1))
                dataY = np.copy(self.data2[i:i + windL]).reshape((1, -1))
                model = LinearRegression()
                model.fit(dataX, x)
                y[j,:]= model.predict(self.data2[i+windL:i + windL+self.predL].reshape((-1,1))).flatten()
                clf.append(model)
            return np.median(y,axis=0)

class EWTSCADADataset(Dataset):
    # +jf33N train together 正常baseline的数据集 没有额外的数据预处理
    def __init__(self, dataName, Tpyename, windL, predL, wtnum,numberewt=200, server=False, noise=False):
        if server:
            dataFileDict = serverDict
        else:
            dataFileDict = mydict
        if server:
            outf = r'/home/daisy001/mountdir/LuoxiaoYANG/Model/WindPrediction'
        else:
            outf = r'D:\YANG Luoxiao\Model\WindSpeed'

        dataFile=dataFileDict[dataName]
        if '-' in dataName:
            self.dataType = 'ZM'
        elif 'NewYork' in dataFile:
            self.dataType = 'NY'
        elif 'WT10s' in dataFile:
            self.dataType = '10s'
        else:
            self.dataType = 'else'
        if self.dataType == 'NY':
            filename = dataFile + r'\\' + Tpyename
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
        elif self.dataType == '10s':
            filename = dataFile + Tpyename
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
        elif self.dataType == 'ZM':
            filename = dataFile + Tpyename
            with open(filename, 'rb') as f:
                self.dataX, self.dataY = pickle.load(f)
                self.data = self.dataY[:, -1, :, :]
        else:
            filename = dataFile + Tpyename
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
        self.predL = predL
        self.noise = noise
        self.windL = windL
        self.numberewt = numberewt
        self.wtnum = wtnum
        if dataName=='K1':
            self.samlpeInterval=3
            self.data=self.data[:int(self.data.shape[0]/self.samlpeInterval)*self.samlpeInterval,:,:].reshape((-1,3,self.data.shape[1],self.data.shape[2]))
            self.data = np.mean(self.data,axis=1)[:,:,[1, 3, 8]]
    def __len__(self):
        if self.dataType == 'kkk':
            return self.dataX.shape[0]

        else:
            return self.data.shape[0] - self.windL - self.predL

    def __getitem__(self, idx):
        if self.dataType == 'kkk':

            x = self.dataX[idx, :, self.wtnum, 0]
            if self.noise:
                x = x+ np.random.randn(x.shape[0]) * (var* x)
            x, mfb, boundaries = ewtpy.EWT1D(x, N= self.numberewt)
            # print(np.sum(np.isnan(x)))
            x = torch.from_numpy(x).float()

            y = self.dataY[idx, :, self.wtnum, 0]
            y = torch.from_numpy(y).float()
            return x, y
        elif self.dataType == '10s':
            x = self.data[idx:idx + self.windL, self.wtnum, 13]  # 15 wt 6 p
            if self.noise:
                x = x + np.random.randn(x.shape[0]) / (var / 22.6 * x)
            x, mfb, boundaries = ewtpy.EWT1D(x, N= self.numberewt)
            x = torch.from_numpy(x).float()

            y = np.copy(self.data[idx + self.windL:idx + self.windL + self.predL, self.wtnum, 13])
            y = torch.from_numpy(y).float()
            return x, y
        else:
            x = np.copy(self.data[idx:idx + self.windL, self.wtnum, 0])
            x, mfb, boundaries = ewtpy.EWT1D(x, N= self.numberewt)
            # print(np.sum(np.isnan(x)))
            x = torch.from_numpy(x).float()

            y = np.copy(self.data[idx + self.windL:idx + self.windL + self.predL, self.wtnum, 0])
            y = torch.from_numpy(y).float()
            return x, y



from scipy.stats import wilcoxon
def Wilcoxon_signed_rank_test(x,y):
    #检验x，y有无显著不同，返回显著性 其实就是下x是不是明显比y好
    return wilcoxon(x,y)[1]



