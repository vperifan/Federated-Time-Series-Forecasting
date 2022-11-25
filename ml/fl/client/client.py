"""
Client abstract representation.
"""

import sys

from pathlib import Path

parent = Path(__file__).resolve().parents[3]
if parent not in sys.path:
    sys.path.insert(0, str(parent))

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Union, Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader


class Client(ABC):
    """Abstract class for representing clients."""

    @abstractmethod
    def get_parameters(self) -> List[np.ndarray]:
        """Return the current local model."""

    @abstractmethod
    def set_train_parameters(self, params: Dict[str, Union[bool, str, int, float]], verbose: bool):
        """Set the local train parameters"""

    @abstractmethod
    def fit(self, model: Optional[Union[torch.nn.Module, List[np.ndarray]]]) -> Tuple[
        List[np.ndarray], int, float, Dict[str, float], int, float, Dict[str, float]]:
        """Local training.
            Returns:
                1) a list of np.ndarrays containing the local model
                2) the number of local training instances
                3) the local train loss
                4) the local train metrics in a Dict format, e.g. {"MSE", 0.1}
                5) the number of local testing instances
                5) the local test loss
                6) the local test metrics in a Dict format
        Note that clients may not own a local validation/test set, i.e. the validation/test set can be global.
        We need a validation/test set to perform evaluation at each local epoch.
        """

    @abstractmethod
    def evaluate(self, data: Optional[Union[np.ndarray, DataLoader]],
                 model: Optional[Union[torch.nn.Module, List[np.ndarray]]],
                 params: Optional[Dict[str, Any]],
                 method: Optional[str],
                 verbose: Optional[bool]) -> Tuple[
        int, float, Dict[str, float]]:
        """Global model evaluation.
            Returns:
                1) The number of evaluation instances.
                2) The evaluation loss
                3) the evaluation metrics
        Note that the evaluate method can correspond to the evaluation of the global model to
        either the local training instances or the (local) validation/testing instances.
        """
