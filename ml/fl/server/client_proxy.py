"""
Abstract method for Client representation on the server side. This abstract class is equivalent to the
client/client.py representation of clients.
"""

import sys

from pathlib import Path

parent = Path(__file__).resolve().parents[3]
if parent not in sys.path:
    sys.path.insert(0, str(parent))

from abc import ABC, abstractmethod
from typing import Union, Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader


class ClientProxy(ABC):
    """Abstract class for representing clients."""

    def __init__(self, cid: Union[str, int]):
        self.cid = cid

    @abstractmethod
    def get_parameters(self) -> List[np.ndarray]:
        """Return the current local model weights."""

    def set_train_parameters(self, params: Dict[str, Union[str, bool, int, float]],
                             verbose: bool = False) -> None:
        """Set local parameters"""

    @abstractmethod
    def fit(self, model: Optional[Union[torch.nn.Module, List[np.ndarray]]]) -> Tuple[
        List[np.ndarray], int, float, Dict[str, float], int, float, Dict[str, float]]:
        """Local training.
            Returns:
                1) a list of np.ndarray which corresponds to local learned weights
                2) the number of local instances
                3) the local train loss
                4) the local train metrics
                5) the local test loss
                6) the local test metrics
        Note that clients may not own a local validation/test set, i.e. the validation/test set can be global.
        We need a validation/test set to perform evaluation at each local epoch.
        """

    @abstractmethod
    def evaluate(self, data: Optional[DataLoader] = None,
                 model: Optional[Union[torch.nn.Module, List[np.ndarray]]] = None,
                 params: Dict[str, Any] = None,
                 method: Optional[str] = None,
                 verbose: bool = False) -> Tuple[int, float, Dict[str, float]]:
        """Global model evaluation.
            Returns:
                1) The number of evaluation instances.
                2) The evaluation loss
                3) the evaluation metrics
        Note that the evaluate method can correspond to the evaluation
        of the global model to either the local training instances or the (local) testing instances.
        """
