"""
Implements the Client.
"""

import sys

from pathlib import Path

parent = Path(__file__).resolve().parents[2]
if parent not in sys.path:
    sys.path.insert(0, str(parent))

from logging import INFO, DEBUG
from typing import Dict, Tuple, List, Union, Optional, Any

from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.fl.client.client import Client
from ml.utils.logger import log
from ml.utils.train_utils import train, test


class TorchRegressionClient(Client):
    def __init__(self, cid: Union[str, int], net: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 local_train_params: Optional[Dict[str, Union[str, int, float, bool]]]):
        self.cid = cid
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.initial_train_params = local_train_params
        self.epochs = None
        self.optimizer = None
        self.lr = None
        self.criterion = None
        self.early_stopping = None
        self.patience = None
        self.device = None
        self.reg1 = None
        self.reg2 = None
        self.max_grad_norm = None
        self.fed_prox_mu = None
        self._init_local_train_params()

    def _init_local_train_params(self):

        self.epochs = self.initial_train_params["epochs"]
        self.optimizer = self.initial_train_params["optimizer"]
        self.lr = self.initial_train_params["lr"]
        self.criterion = self.initial_train_params["criterion"]
        self.early_stopping = self.initial_train_params["early_stopping"]
        self.patience = self.initial_train_params["patience"]
        self.device = self.initial_train_params["device"]
        try:
            self.reg1 = self.initial_train_params["reg1"]
        except KeyError:
            self.reg1 = 0.
        try:
            self.reg2 = self.initial_train_params["reg2"]
        except KeyError:
            self.reg2 = 0.
        try:
            self.max_grad_norm = self.initial_train_params["max_grad_norm"]
        except KeyError:
            self.max_grad_norm = 0.

        try:
            self.fed_prox_mu = self.initial_train_params["fedprox_mu"]
        except KeyError:
            self.fed_prox_mu = 0.

    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_train_parameters(self,
                             params: Dict[str, Union[bool, str, int, float]],
                             verbose: bool = False):  # default parameters

        self.epochs = params["epochs"] if "epochs" in params else self.epochs
        self.optimizer = params["optimizer"] if "optimizer" in params else self.optimizer
        self.lr = params["lr"] if "lr" in params else self.lr
        self.criterion = params["criterion"] if "criterion" in params else self.criterion
        self.early_stopping = params["early_stopping"] if "early_stopping" in params else self.early_stopping
        self.patience = params["patience"] if "patience" in params else self.patience
        self.device = params["device"] if "device" in params else self.device
        self.reg1 = params["reg1"] if "reg1" in params else self.reg1
        self.reg2 = params["reg2"] if "reg2" in params else self.reg2
        self.max_grad_norm = params["max_grad_norm"] if "max_grad_norm" in params else self.max_grad_norm
        self.fed_prox_mu = params["fedprox_mu"] if "fedprox_mu" in params else self.fed_prox_mu

        if verbose:
            log(DEBUG, f"Training parameters change for client {self.cid}: "
                       f"epochs={self.epochs}, optimizer={self.optimizer}, lr={self.lr}, "
                       f"criterion={self.criterion}, early_stopping={self.early_stopping}, patience={self.patience}, "
                       f"device={self.device}, reg1={self.reg1}, reg2={self.reg2}, max_grad_norm={self.max_grad_norm}")

    def set_parameters(self, parameters: Union[List[np.ndarray], torch.nn.Module]):
        if not isinstance(parameters, torch.nn.Module):
            params_dict = zip(self.net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            self.net.load_state_dict(state_dict, strict=True)
        else:
            self.net.load_state_dict(parameters.state_dict(), strict=True)

    def fit(self, model: Optional[Union[torch.nn.Module, List[np.ndarray]]] = None) -> Tuple[
        List[np.ndarray], int, float, Dict[str, float], int, float, Dict[str, float]]:

        if model is not None:
            self.set_parameters(model)

        self.net: torch.nn.Module = train(model=self.net, train_loader=self.train_loader, test_loader=self.val_loader,
                                          epochs=self.epochs, optimizer=self.optimizer,
                                          lr=self.lr, criterion=self.criterion,
                                          early_stopping=self.early_stopping, patience=self.patience,
                                          reg1=self.reg1, reg2=self.reg2, max_grad_norm=self.max_grad_norm,
                                          fedprox_mu=self.fed_prox_mu,
                                          log_per=10)
        _, train_loss, train_metrics = self.evaluate(self.train_loader)
        num_test, test_loss, test_metrics = self.evaluate(self.val_loader)

        return self.get_parameters(), len(
            self.train_loader.dataset), train_loss, train_metrics, num_test, test_loss, test_metrics

    def evaluate(self, data: Optional[DataLoader] = None,
                 model: Optional[Union[torch.nn.Module, List[np.ndarray]]] = None,
                 params: Dict[str, Any] = None, method: Optional[str] = None, verbose: bool = False) -> Tuple[
        int, float, Dict[str, float]]:

        if not params or "criterion" not in params:
            params = dict()
            params["criterion"] = torch.nn.MSELoss()

        if model:
            self.set_parameters(model)

        if data is None and method == "test":
            data = self.val_loader
        if data is None and method == "train":
            data = self.train_loader

        loss, mse, rmse, mae, r2, nrmse = test(self.net, data, params["criterion"], device=self.device)

        metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R^2": r2, "NRMSE": nrmse}

        if verbose:
            log(INFO, f"[Client {self.cid} Evaluation on {len(data.dataset)} samples] "
                      f"loss: {loss}, mse: {mse}, rmse: {rmse}, mae: {mae}, nrmse: {nrmse}")

        return len(data.dataset), loss, metrics
