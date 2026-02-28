"""Abstract base class for online learning datasets."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass
class TaskData:
    """Data for a single online learning task."""

    X_train: torch.Tensor  # [N_train, D]
    y_train: torch.Tensor  # [N_train, 1]
    X_test: torch.Tensor  # [N_test, D]
    y_test: torch.Tensor  # [N_test, 1]
    task_id: int
    metadata: Optional[dict] = None


class OnlineDataset(ABC):
    """Base class for online learning datasets.

    Yields (X_train, y_train, X_test, y_test, metadata) for each task.
    """

    @property
    @abstractmethod
    def num_tasks(self) -> int:
        """Number of sequential tasks."""

    @abstractmethod
    def tasks(self) -> Iterator[TaskData]:
        """Iterate over tasks in order."""
