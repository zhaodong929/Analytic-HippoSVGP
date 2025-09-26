import os
import glob
import numpy as np
import torch

class COVIDDataset(torch.utils.data.Dataset):
    def __init__(self, folder, load_checkpoint=False, prev_task_folder=None):
        self.files = glob.glob(os.path.join(folder, "sequences", "*.npz"))
        self.load_checkpoint = load_checkpoint
        self.prev_task_folder = prev_task_folder
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        filename = os.path.basename(file)
        data = np.load(file)

        max_seq_length = data["max_seq_length"]
        Rt = data["Rt"]
        Rt_lower = data["Rt_lower"]
        Rt_upper = data["Rt_upper"]
        deaths = data["deaths"]
        days = data["days"]
        f = data["f"][:max_seq_length]
        EpidemicStart = data["EpidemicStart"]
        population = data["population"]
        num_days = data["num_days"]
        state_index = data["state_index"]
        week_index = data["week_index"]

        # convert start_date to from array('2020-03-22', dtype='<U10') to list of string
        start_date = data["start_date"].astype(str).tolist()

        # Pad the sequences to the maximum length
        Rt = np.pad(Rt, (0, max_seq_length - len(Rt)), mode='constant', constant_values=0)
        Rt_lower = np.pad(Rt_lower, (0, max_seq_length - len(Rt_lower)), mode='constant', constant_values=0)
        Rt_upper = np.pad(Rt_upper, (0, max_seq_length - len(Rt_upper)), mode='constant', constant_values=0)
        deaths = np.pad(deaths, (0, max_seq_length - len(deaths)), mode='constant', constant_values=0)
        days = np.pad(days, (0, max_seq_length - len(days)), mode='constant', constant_values=0)
        week_index = np.pad(week_index, (0, max_seq_length - len(week_index)), mode='constant', constant_values=0)
        

        if self.load_checkpoint:
            checkpoint_file = os.path.join(self.prev_task_folder, "checkpoints", filename)
            checkpoint_data = np.load(checkpoint_file)['data']
            return deaths, days, Rt, Rt_lower, Rt_upper, num_days, f, EpidemicStart, population, filename, start_date, state_index, week_index, checkpoint_data
        return deaths, days, Rt, Rt_lower, Rt_upper, num_days, f, EpidemicStart, population, filename, start_date, state_index, week_index
