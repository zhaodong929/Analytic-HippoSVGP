"""COVID-19 online dataset for sequential task learning."""

import os
import glob

import numpy as np
import torch
import random

from scripts.datasets.base import OnlineDataset, TaskData


class COVIDOnlineDataset(OnlineDataset):
    """Brazilian COVID-19 time series split into sequential tasks.

    Each task corresponds to a time period. Data is loaded from the
    processed_timeseries_5tasks directory structure.

    Args:
        data_root: path to processed_timeseries_5tasks directory
        num_tasks: number of tasks (default 5)
        test_fraction: fraction of data to use for testing (default 0.2)
        seed: random seed for train/test split (default 42)
        device: torch device
    """

    def __init__(
        self,
        data_root,
        num_tasks=5,
        test_fraction=0.2,
        seed=42,
        device="cpu",
    ):
        self._data_root = data_root
        self._num_tasks = num_tasks
        self._test_fraction = test_fraction
        self._seed = seed
        self._device = device

    @property
    def num_tasks(self):
        return self._num_tasks

    def _load_task(self, task_id):
        """Load raw data for a single task."""
        task_dir = os.path.join(self._data_root, f"task_{task_id}")
        files = glob.glob(os.path.join(task_dir, "sequences", "*.npz"))
        if not files:
            raise FileNotFoundError(f"No data files found in {task_dir}/sequences/")

        data = np.load(files[0])
        num_days = int(data["num_days"])
        days = data["days"][:num_days].astype(np.float64)
        deaths = data["deaths"][:num_days].astype(np.float64)

        return days, deaths

    def tasks(self):
        for task_id in range(1, self._num_tasks + 1):
            days, deaths = self._load_task(task_id)

            days_t = torch.tensor(days, dtype=torch.float64, device=self._device).unsqueeze(1)
            deaths_t = torch.tensor(deaths, dtype=torch.float64, device=self._device).unsqueeze(1)

            # Deterministic train/test split
            rng_state = random.getstate()
            np_rng_state = np.random.get_state()
            torch_rng_state = torch.random.get_rng_state()

            random.seed(self._seed)
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)

            N = len(days)
            rand_ind = torch.randperm(N)

            random.setstate(rng_state)
            np.random.set_state(np_rng_state)
            torch.random.set_rng_state(torch_rng_state)

            test_size = int(N * self._test_fraction)
            test_idx = rand_ind[:test_size]
            train_idx = rand_ind[test_size:]

            yield TaskData(
                X_train=days_t[train_idx],
                y_train=deaths_t[train_idx],
                X_test=days_t[test_idx],
                y_test=deaths_t[test_idx],
                task_id=task_id,
            )
