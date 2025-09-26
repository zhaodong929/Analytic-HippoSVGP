import torch 
import glob
import os
import numpy as np
import pickle
import joblib

class ERA5Dataset(torch.utils.data.Dataset):
    def __init__(self, folder, load_checkpoint=False, prev_task_folder=None):
        self.files = glob.glob(os.path.join(folder, "sequences", "*scaled.npz"))
        self.load_checkpoint = load_checkpoint
        self.prev_task_folder = prev_task_folder

        # load scaler from joblib.dump
        scaler_file = os.path.join(folder, "../global_scaler.pkl")
        if os.path.exists(scaler_file):
            self.scaler = joblib.load(scaler_file)
        else:
            raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
        # get scale of scaler 
        self.scale = self.scaler.scale_
        self.mean = self.scaler.mean_

        self.indices = [
            0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 29, 30, 31, 33, 34
        ]
        self.scale = self.scale[self.indices]
        self.mean = self.mean[self.indices]
        
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        file = self.files[idx]
        filename = os.path.basename(file)
        data_train = np.load(file)['data_train'].T
        data_val = np.load(file)['data_val'].T
        data_test = np.load(file)['data_test'].T
        data_train = data_train[:, self.indices]
        data_val = data_val[:, self.indices]
        data_test = data_test[:, self.indices]
        time_train = np.load(file)['time_train']
        time_val = np.load(file)['time_val']
        time_test = np.load(file)['time_test']
        if self.load_checkpoint:
            checkpoint_file = os.path.join(self.prev_task_folder, "checkpoints", filename)
            checkpoint_data = np.load(checkpoint_file)['data']
            return data_train, data_val, data_test, time_train, time_val, time_test, checkpoint_data, idx
        return data_train, data_val, data_test, time_train, time_val, time_test, idx