"""Solar irradiance online dataset."""

import jax.numpy as jnp
import numpy as np

from scripts.datasets.base import OnlineDataset, TaskData


class SolarOnlineDataset(OnlineDataset):
    """Solar irradiance data split into sequential chunks.

    Args:
        train_path: path to solar_data_train.npy
        test_path: path to solar_data_test.npy
        num_chunks: number of online learning tasks
    """

    def __init__(self, train_path, test_path, num_chunks=10):
        self._train_path = train_path
        self._test_path = test_path
        self._num_chunks = num_chunks

    @property
    def num_tasks(self):
        return self._num_chunks

    def tasks(self):
        data_train = np.load(self._train_path)
        data_test = np.load(self._test_path)

        x = jnp.array(data_train[:, 0:1], dtype=jnp.float64)
        y = jnp.array(data_train[:, 1:2], dtype=jnp.float64)
        x_t = jnp.array(data_test[:, 0:1], dtype=jnp.float64)
        y_t = jnp.array(data_test[:, 1:2], dtype=jnp.float64)

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
