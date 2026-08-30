import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
from sklearn.preprocessing import MinMaxScaler
class VOC(Dataset):
    def __init__(self,path):
        self.Y = scipy.io.loadmat(path + 'VOC')['Y'].astype(np.int32).reshape(5649,)
        self.V1 = scipy.io.loadmat(path + 'VOC')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'VOC')['X2'].astype(np.float32)
    def __len__(self):
        return 5649
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2)], \
            self.Y[idx], torch.from_numpy(np.array(idx)).long()

class synthetic3d(Dataset):
    def __init__(self,path):
        self.Y = scipy.io.loadmat(path + 'synthetic3d')['Y'].astype(np.int32).reshape(600, )
        self.V1 = scipy.io.loadmat(path + 'synthetic3d')['X'][0][0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'synthetic3d')['X'][1][0].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'synthetic3d')['X'][2][0].astype(np.float32)

        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(self.V1)
        self.V2 = scaler.fit_transform(self.V2)
        self.V3 = scaler.fit_transform(self.V3)

    def __len__(self):
        return 600
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2),
                torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class LandUse_21(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'LandUse-21')['Y'].astype(np.int32).reshape(2100, )
        self.V1 = scipy.io.loadmat(path + 'LandUse-21')['X'][0][0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'LandUse-21')['X'][0][1].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'LandUse-21')['X'][0][2].astype(np.float32)
    def __len__(self):
        return 2100
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)],\
            self.Y[idx], torch.from_numpy(np.array(idx)).long()


class scene_15:
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'scene-15')['Y'].astype(np.int32).reshape(4485, )
        self.V1 = scipy.io.loadmat(path + 'scene-15')['X'][0][0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'scene-15')['X'][0][1].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'scene-15')['X'][0][2].astype(np.float32)
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(self.V1)
        self.V2 = scaler.fit_transform(self.V2)
        self.V3 = scaler.fit_transform(self.V3)

    def __len__(self):
        return 4485
    def __getitem__(self, idx):


        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], \
            self.Y[idx], torch.from_numpy(np.array(idx)).long()

class WIKI(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'WIKI')['label'].astype(np.int32).reshape(2866, )
        self.V1 = scipy.io.loadmat(path + 'WIKI')['Img'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'WIKI')['Txt'].astype(np.float32)



    def __len__(self):
        return 2866
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2)], \
            self.Y[idx], torch.from_numpy(np.array(idx)).long()

class CUB(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'CUB')['gt'].astype(np.int32).reshape(600, )
        self.V1 = scipy.io.loadmat(path + 'CUB')['X'][0][0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'CUB')['X'][0][1].astype(np.float32)

        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(self.V1)
        self.V2 = scaler.fit_transform(self.V2)


    def __len__(self):
        return 600
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2)], \
            self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Deep_Animal:
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Deep Animal')['gt'].astype(np.int32).reshape(10158, )
        self.V1 = scipy.io.loadmat(path + 'Deep Animal')['X'][0][5].astype(np.float32).T
        self.V2 = scipy.io.loadmat(path + 'Deep Animal')['X'][0][6].astype(np.float32).T
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(self.V1)
        self.V2 = scaler.fit_transform(self.V2)
    def __len__(self):
        return 10158
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)],\
            self.Y[idx], torch.from_numpy(np.array(idx)).long()
class onehundredLeaves:
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + '100Leaves')['Y'].astype(np.int32).reshape(1600,)
        self.V1 = scipy.io.loadmat(path + '100Leaves')['X'][0][0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + '100Leaves')['X'][0][1].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + '100Leaves')['X'][0][2].astype(np.float32)

        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(self.V1)
        self.V2 = scaler.fit_transform(self.V2)
        self.V3 = scaler.fit_transform(self.V3)

    def __len__(self):
        return 1600
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2),torch.from_numpy(x3)], \
            self.Y[idx], torch.from_numpy(np.array(idx)).long()


class cifar_10():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar10.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)


    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], \
            self.Y[idx], torch.from_numpy(np.array(idx)).long()

class cifar_100():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar100.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)],\
            self.Y[idx], torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "VOC":
        dataset = VOC('./data/')
        dims = [512, 399]
        dimss = 911
        view = 2
        data_size = 5649
        class_num = 20
    elif dataset =="synthetic3d":
        dataset = synthetic3d('./data/')
        dims =[3,3,3]
        dimss = 9
        view = 3
        data_size = 600
        class_num = 3
    elif dataset =="LandUse-21":
        dataset = LandUse_21('./data/')
        dims = [20,59,40]
        dimss = 119
        view = 3
        data_size = 2100
        class_num = 21
    elif dataset =="scene-15":
        dataset = scene_15('./data/')
        dims = [20,59,40]
        dimss = 119
        view = 3
        data_size = 4485
        class_num = 15

    elif dataset == "WIKI":
        dataset = WIKI('./data/')
        dims = [128,10]
        dimss = 138
        view = 2
        data_size = 2866
        class_num = 10
    elif dataset =="CUB":
        dataset = CUB('./data/')
        dims = [1024,300]
        dimss = 1324
        view = 2
        data_size = 600
        class_num = 10

    elif dataset =="Deep Animal":
        dataset = Deep_Animal('./data/')
        dims = [4096,4096]
        dimss = 8192
        view = 2
        data_size = 10158
        class_num = 50
    elif dataset == "100Leaves":
        dataset = onehundredLeaves('./data/')
        dims = [64,64,64]
        dimss = 192
        view = 3
        data_size = 1600
        class_num = 100
    elif dataset == "Cifar100":
        dataset = cifar_100('./data/')
        dims = [512, 2048, 1024]
        dimss = 3584
        view = 3
        data_size = 50000
        class_num = 100
    elif dataset == "Cifar10":
        dataset = cifar_10('./data/')
        dimss = 3584
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 10
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num, dimss




