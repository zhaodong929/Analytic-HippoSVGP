import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split
from scipy.io import loadmat

class Elevators(object):
    def __init__(self, dataset_dir=None, subsample_ratio=1.0, test_ratio=0., split_seed=0, **kwargs):
        if dataset_dir is None:
            self.dataset_dir = './data/uci'
        else:
            self.dataset_dir = dataset_dir
        self.train_dataset, self.test_dataset = self._preprocess(subsample_ratio, test_ratio, split_seed)

    def _preprocess(self, subsample_ratio, test_ratio, split_seed):
        file_path = os.path.join(self.dataset_dir, 'elevators.mat')
        data = loadmat(file_path)['data']
        inputs = torch.tensor(data[:, :-1], dtype=torch.float64)
        targets = torch.tensor(data[:, -1], dtype=torch.float64).unsqueeze(-1)

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        input_max, _ = inputs.max(0)
        input_min, _ = inputs.min(0)
        input_range = input_max - input_min
        inputs = 2 * ((inputs - input_min) / input_range - 0.5)
        targets = (targets - targets.mean(0)) / targets.std(0)

        dataset = TensorDataset(inputs, targets)
        generator = torch.Generator().manual_seed(split_seed)
        num_samples = int(subsample_ratio * len(dataset))
        dataset, _ = random_split(dataset, [num_samples, len(dataset) - num_samples],
                                  generator=generator)
        dataset = TensorDataset(*dataset[:])

        num_test = int(test_ratio * len(dataset))
        train_dataset, test_dataset = random_split(dataset, [len(dataset) - num_test, num_test],
                                                   generator=generator)
        return train_dataset, test_dataset

class Skillcraft(object):
    def __init__(self, dataset_dir=None, subsample_ratio=1.0, test_ratio=0.,
                 split_seed=0, **kwargs):
        if dataset_dir is None:
            self.dataset_dir = './data/uci'
        else:
            self.dataset_dir = dataset_dir
        self.train_dataset, self.test_dataset = self._preprocess(subsample_ratio, test_ratio, split_seed)

    def _preprocess(self, subsample_ratio, test_ratio, split_seed):
        file_path = os.path.join(self.dataset_dir, 'SkillCraft1_Dataset.csv')
        df = pd.read_csv(file_path)
        res = (df == '?')
        ind = (np.sum(res, 1) == 0)
        df = df[ind]
        df = df.drop('GameID', axis=1)
        df = df.drop('LeagueIndex', axis=1)
        data = np.array(df, dtype=np.float64)

        inputs = torch.tensor(data[:, :-1], dtype=torch.get_default_dtype())
        targets = torch.tensor(data[:, -1], dtype=torch.get_default_dtype()).unsqueeze(-1)

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        input_max, _ = inputs.max(0)
        input_min, _ = inputs.min(0)
        input_range = input_max - input_min
        inputs = 2 * ((inputs - input_min) / input_range - 0.5)
        targets = (targets - targets.mean(0)) / targets.std(0)

        dataset = TensorDataset(inputs, targets)
        generator = torch.Generator().manual_seed(split_seed)
        num_samples = int(subsample_ratio * len(dataset))
        dataset, _ = random_split(dataset, [num_samples, len(dataset) - num_samples],
                                  generator=generator)
        dataset = TensorDataset(*dataset[:])

        num_test = int(test_ratio * len(dataset))
        train_dataset, test_dataset = random_split(dataset, [len(dataset) - num_test, num_test],
                                                   generator=generator)
        return train_dataset, test_dataset

class Powerplant(object):
    def __init__(self, dataset_dir=None, subsample_ratio=1.0, test_ratio=0., 
            split_seed=0, **kwargs):
        if dataset_dir is None:
            self.dataset_dir = './data/uci'
        else:
            self.dataset_dir = dataset_dir
        #self.train_dataset, self.test_dataset, self.ini_dataset = self._preprocess(subsample_ratio, test_ratio, \ #ini_ratio,
        #                                                         split_seed)
        self.train_dataset, self.test_dataset= self._preprocess(subsample_ratio, test_ratio, split_seed)

    def _preprocess(self, subsample_ratio, test_ratio, split_seed):
        file_path = os.path.join(self.dataset_dir, 'Folds5x2_pp.xlsx')
        raw_data = np.array(pd.read_excel(file_path))
        inputs = torch.tensor(raw_data[:, :-1], dtype=torch.float64)
        targets = torch.tensor(raw_data[:, -1], dtype=torch.float64).unsqueeze(-1)

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        input_max, _ = inputs.max(0)
        input_min, _ = inputs.min(0)
        input_range = input_max - input_min
        inputs = 2 * ((inputs - input_min) / input_range - 0.5)
        targets = (targets - targets.mean(0)) / targets.std(0)

        dataset = TensorDataset(inputs, targets)
        generator = torch.Generator().manual_seed(split_seed)
        num_samples = int(subsample_ratio * len(dataset))
        dataset, _ = random_split(dataset, [num_samples, len(dataset) - num_samples],
                                  generator=generator)
        dataset = TensorDataset(*dataset[:])

        num_test = int(test_ratio * len(dataset))
        #num_ini = int(ini_ratio * len(dataset))

        train_dataset, test_dataset = random_split(dataset, [len(dataset) - num_test, num_test],
                                                   generator=generator)
        #train_dataset, ini_dataset = random_split(train_dataset, [len(train_dataset) - num_ini, num_ini],
        #                                           generator=generator)
        return train_dataset, test_dataset#, ini_dataset
    

def get_data_banana(iid=False):
    X = np.loadtxt('./data/uci/banana_train_x.txt', delimiter=',')
    y = np.loadtxt('./data/uci/banana_train_y.txt', delimiter=',')
    y = y.reshape((y.shape[0], 1))
    y[y==-1] = 0
    Xtest = np.loadtxt('./data/uci/banana_test_x.txt', delimiter=',')
    ytest = np.loadtxt('./data/uci/banana_test_y.txt', delimiter=',')
    ytest = ytest.reshape((ytest.shape[0], 1))
    ytest[ytest==-1] = 0
    if not iid: 
        # todo sort
        idxs = np.argsort(X[:, 0])
        X = X[idxs, :]
        y = y[idxs, :]
    return X, y, Xtest, ytest

def get_data_moon():
    data = pd.read_csv('./data/uci/make_moons.csv')
    X1 = data['X']
    X2 = data['Y']
    Y_np = np.array(data['label'])
    
    X1_y0 = X1[Y_np==0]
    X1_y1 = X1[Y_np==1]
    X2_y0 = X2[Y_np==0]
    X2_y1 = X2[Y_np==1]
    ind0 = np.array(np.argsort(X1_y0))
    ind1 = np.array(np.argsort(X1_y1))
    X_sort_y0 = np.stack([np.expand_dims(X1_y0, 1)[ind0], np.expand_dims(X2_y0,1)[ind0]], 1).squeeze(-1)
    X_sort_y1 = np.stack([np.expand_dims(X1_y1, 1)[ind1], np.expand_dims(X2_y1,1)[ind1]], 1).squeeze(-1)
    X_batch1 = np.concatenate([X_sort_y0[:200], X_sort_y1[:200]]) 
    Y_batch1 = np.array([0]* 200 + [1]* 200)
    ind = np.argsort(X_batch1[:, 0])
    X_batch1 = X_batch1[ind]
    Y_batch1 = np.expand_dims(Y_batch1[ind], 1)

    X_batch2 = np.concatenate([X_sort_y0[200:350], X_sort_y1[200:350]]) 
    Y_batch2 = np.array([0]* 150 + [1]* 150)
    ind = np.argsort(X_batch2[:, 0])
    X_batch2 = X_batch2[ind]
    Y_batch2 = np.expand_dims(Y_batch2[ind], 1)

    X_batch3 = np.concatenate([X_sort_y0[350:], X_sort_y1[350:]]) 
    Y_batch3 = np.array([0]* 150 + [1]* 150)
    ind = np.argsort(X_batch3[:, 0])
    X_batch3 = X_batch3[ind]
    Y_batch3 = Y_batch3[ind]
    Y_batch3 = np.expand_dims(Y_batch3[ind], 1)
    """
    X_batch4 = np.concatenate([X_sort_y0[400:], X_sort_y1[400:]]) 
    Y_batch4 = np.array([0]* 100 + [1]* 100)
    ind = np.argsort(X_batch4[:, 0])
    X_batch4 = X_batch4[ind]
    Y_batch4 = Y_batch4[ind]
    Y_batch4 = np.expand_dims(Y_batch4[ind], 1)
    """
    X_train1 = X_batch1[::2]
    X_test1 = X_batch1[1::2]
    Y_train1 = Y_batch1[::2]
    Y_test1 = Y_batch1[1::2]
    
    X_train2 = X_batch2[::2]
    X_test2 = X_batch2[1::2]
    Y_train2 = Y_batch2[::2]
    Y_test2 = Y_batch2[1::2]

    X_train3 = X_batch3[::2]
    X_test3 = X_batch3[1::2]
    Y_train3 = Y_batch3[::2]
    Y_test3 = Y_batch3[1::2]
    """
    X_train4 = X_batch4[::2]
    X_test4 = X_batch4[1::2]
    Y_train4 = Y_batch4[::2]
    Y_test4 = Y_batch4[1::2]
    """
    X = [X_train1, X_train2, X_train3]#, X_train4]
    y = [Y_train1, Y_train2, Y_train3]#, Y_train4]

    Xtest = [X_test1, X_test2, X_test3]#, X_test4]
    ytest = [Y_test1, Y_test2, Y_test3]#, Y_test4]
    return X, y, Xtest, ytest
