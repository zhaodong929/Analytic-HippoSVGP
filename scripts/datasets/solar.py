"""Solar irradiance online dataset."""

import numpy as np
import torch

from scripts.datasets.base import OnlineDataset, TaskData


class SolarOnlineDataset(OnlineDataset):
    """Solar irradiance data split into sequential chunks.

    Args:
        train_path: path to solar_data_train.npy
        test_path: path to solar_data_test.npy
        num_chunks: number of online learning tasks
        device: torch device
    """

    def __init__(self, train_path, test_path, num_chunks=10, device="cpu"):
        self._train_path = train_path
        self._test_path = test_path
        self._num_chunks = num_chunks
        self._device = device

    @property
    def num_tasks(self):
        return self._num_chunks

    def tasks(self):
        data_train = np.load(self._train_path)
        data_test = np.load(self._test_path)

        x = torch.tensor(data_train[:, 0:1], dtype=torch.float64, device=self._device)
        y = torch.tensor(data_train[:, 1:2], dtype=torch.float64, device=self._device)
        x_t = torch.tensor(data_test[:, 0:1], dtype=torch.float64, device=self._device)
        y_t = torch.tensor(data_test[:, 1:2], dtype=torch.float64, device=self._device)

        split_train = x.shape[0] // self._num_chunks
        split_test = x_t.shape[0] // self._num_chunks

        for i in range(self._num_chunks):
            yield TaskData(
                X_train=x[i * split_train : (i + 1) * split_train],
                y_train=y[i * split_train : (i + 1) * split_train],
                X_test=x_t[i * split_test : (i + 1) * split_test],
                y_test=y_t[i * split_test : (i + 1) * split_test],
                task_id=i + 1,
            )
