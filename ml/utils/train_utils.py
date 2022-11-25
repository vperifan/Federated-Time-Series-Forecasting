"""
Training pipeline.
"""

import sys

from pathlib import Path

parent = Path(__file__).resolve().parents[2]
if parent not in sys.path:
    sys.path.insert(0, str(parent))

import copy
from logging import INFO
from typing import Tuple, List, Union

import torch
from matplotlib import pyplot as plt

from ml.utils.logger import log
from ml.utils.helpers import get_optim, get_criterion, EarlyStopping, accumulate_metric


def train(model: torch.nn.Module,
          train_loader,
          test_loader,
          epochs: int = 10,
          optimizer: str = "adam",
          lr: float = 1e-3,
          reg1: float = 0.,
          reg2: float = 0.,
          max_grad_norm: float = 0.,
          criterion: str = "mse",
          early_stopping: bool = True,
          patience: int = 50,
          plot_history: bool = False,
          device="cuda",
          fedprox_mu: float = 0.,
          log_per: int = 1,
          use_carbontracker: bool = False):
    """Trains a neural network defined as torch module."""
    best_model, best_loss, best_epoch = None, -1, -1
    train_loss_history, train_rmse_history = [], []
    test_loss_history, test_rmse_history = [], []
    if early_stopping:
        es_trace = True if log_per == 1 else False
        monitor = EarlyStopping(patience, trace=es_trace)
    cb_tracker = None
    if use_carbontracker:
        try:
            from carbontracker.tracker import CarbonTracker
            cb_tracker = CarbonTracker(epochs=epochs, components="all", verbose=1)
        except ImportError:
            pass
    optimizer = get_optim(model, optimizer, lr)
    criterion = get_criterion(criterion)
    global_weight_collector = copy.deepcopy(list(model.parameters()))
    for epoch in range(epochs):
        if use_carbontracker and cb_tracker is not None:
            cb_tracker.epoch_start()
        model.to(device)
        model.train()
        epoch_loss = []
        for x, exogenous, y_hist, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_hist = y_hist.to(device)
            if exogenous is not None and len(exogenous) > 0:
                exogenous = exogenous.to(device)
            else:
                exogenous = None
            optimizer.zero_grad()
            y_pred = model(x, exogenous, device, y_hist)
            loss = criterion(y_pred, y)
            if fedprox_mu > 0.:
                fedprox_reg = 0.
                for param_index, param in enumerate(model.parameters()):
                    fedprox_reg += ((fedprox_mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)

                loss += fedprox_reg
            if reg1 > 0.:
                params = torch.cat([p.view(-1) for name, p in model.named_parameters() if "bias" not in name])
                loss += reg1 * torch.norm(params, 1)
            if reg2 > 0.:
                params = torch.cat([p.view(-1) for name, p in model.named_parameters() if "bias" not in name])
                loss += reg2 * torch.norm(params, 2)
            loss.backward()
            if max_grad_norm > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss.append(loss.item())
        train_loss = sum(epoch_loss) / len(epoch_loss)
        _, train_mse, train_rmse, train_mae, train_r2, train_nrmse = test(model, train_loader,
                                                                                      criterion, device)
        test_loss, test_mse, test_rmse, test_mae, test_r2, test_nrmse = test(model, test_loader,
                                                                                        criterion, device)
        if (epoch + 1) % log_per == 0:
            log(INFO, f"Epoch {epoch + 1} [Train]: loss {train_loss}, mse: {train_mse}, "
                      f"rmse: {train_rmse}, mae {train_mae}, r2: {train_r2}, nrmse: {train_nrmse}")
            log(INFO, f"Epoch {epoch + 1} [Test]: loss {test_loss}, mse: {test_mse}, "
                      f"rmse: {test_rmse}, mae {test_mae}, r2: {test_r2}, nrmse: {test_nrmse}")
        train_loss_history.append(train_mse)
        train_rmse_history.append(train_rmse)
        test_loss_history.append(test_mse)
        test_rmse_history.append(test_rmse)

        if early_stopping:
            monitor(test_loss, model)
            best_loss = abs(monitor.best_score)
            best_model = monitor.best_model
            if epoch + 1 > patience:
                best_epoch = epoch + 1
            elif epoch + 1 == epochs:
                best_epoch = epoch + 1 - monitor.counter
            else:
                best_epoch = epoch + 1 - patience
            if monitor.early_stop:
                log(INFO, "Early Stopping")
                break
        else:
            if best_loss == -1 or test_loss < best_loss:
                best_loss = test_loss
                best_model = copy.deepcopy(model)
                best_epoch = epoch + 1
        if use_carbontracker and cb_tracker is not None:
            cb_tracker.epoch_end()

    if plot_history:
        plt.plot(train_loss_history, label="Train MSE")
        plt.plot(test_loss_history, label="Test MSE")
        plt.legend()
        plt.show()
        plt.close()

        plt.plot(train_rmse_history, label="Train RMSE")
        plt.plot(test_rmse_history, label="Test RMSE")
        plt.legend()
        plt.show()
        plt.close()
    if early_stopping and epochs > patience:
        log(INFO, f"Best Loss: {best_loss}, Best epoch: {best_epoch}")
    else:
        log(INFO, f"Best Loss: {best_loss}")
    return best_model


def test(model, data, criterion, device="cuda") -> Union[
    Tuple[float, float, float, float], List[torch.tensor], torch.tensor]:
    """Tests a trained model."""
    model.to(device)
    model.eval()
    y_true, y_pred = [], []
    loss = 0.
    with torch.no_grad():
        for x, exogenous, y_hist, y in data:
            x, y = x.to(device), y.to(device)
            y_hist = y_hist.to(device)
            if exogenous is not None and len(exogenous) > 0:
                exogenous = exogenous.to(device)
            else:
                exogenous = None
            out = model(x, exogenous, device, y_hist)
            if criterion is not None:
                loss += criterion(out, y).item()
            y_true.extend(y)
            y_pred.extend(out)

    loss /= len(data.dataset)

    y_true = torch.stack(y_true)
    y_pred = torch.stack(y_pred)
    mse, rmse, mae, r2, nrmse = accumulate_metric(y_true.cpu(), y_pred.cpu())
    if criterion is None:
        return mse, rmse, mae, r2, nrmse, y_pred

    return loss, mse, rmse, mae, r2, nrmse
