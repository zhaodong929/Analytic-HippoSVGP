"""Abstract base class for likelihoods."""

from abc import ABC, abstractmethod

import torch.nn as nn


class Likelihood(nn.Module, ABC):
    """Base class for all likelihoods."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def log_prob(self, f, y):
        """Compute log p(y | f).

        Args:
            f: function values [N, 1] or [N]
            y: observations [N, 1] or [N]

        Returns:
            Log likelihood tensor.
        """
