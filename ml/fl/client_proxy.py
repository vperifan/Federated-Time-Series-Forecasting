"""
Client and ClientProxy implementation. ClientProxy instructs Clients to perform operations.
"""

import sys

from pathlib import Path

parent = Path(__file__).resolve().parents[2]
if parent not in sys.path:
    sys.path.insert(0, str(parent))

from typing import Union, Dict, List, Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.fl.server.client_proxy import ClientProxy
from ml.fl.client.client import Client


class SimpleClientProxy(ClientProxy):
    def __init__(self, cid: Union[str, int], client):
        super().__init__(cid)
        self.client: Client = client

    def get_parameters(self) -> List[np.ndarray]:
        """Returns the current local model weights."""
        return self.client.get_parameters()

    def set_train_parameters(self, params: Dict[str, Union[bool, str, int, float]], verbose: bool = False):
        return self.client.set_train_parameters(params, verbose)

    def fit(self, model: Optional[Union[torch.nn.Module, List[np.ndarray]]] = None):
        """Local training."""
        return self.client.fit(model)

    def evaluate(self, data: Optional[DataLoader] = None,
                 model: Optional[Union[torch.nn.Module, List[np.ndarray]]] = None,
                 params: Dict[str, Any] = None, method: Optional[str] = None, verbose: bool = False):
        """Global model evaluation."""
        return self.client.evaluate(data, model, params, method, verbose)
