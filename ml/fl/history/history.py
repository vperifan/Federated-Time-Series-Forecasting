"""
Keeps historical metrics.
"""

from typing import Dict, Union, List


class History:
    """We consider 4 loss metrics and 4 evaluation metrics (MSE, MAE, RMSE, R^2, NRMSE). In any case, you can
    use any evaluation metric.
    Two train losses/metrics concern the clients' local training. Note that clients may have different number of
    train losses/metrics since the random selection process may not choose them in some rounds.
    The last two losses/metrics concerns the corresponding losses/metrics using the global model before local training.
    Note that we measure the averaged global train losses/metrics irrespective of the selected clients in the current
    federated round.
    """

    def __init__(self):
        # holds the local train losses, e.g., {"LesCorts": {1: 0.5}}, i.e., client->round->loss
        self.local_train_losses: Dict[Union[str, int], Dict[int, float]] = dict()
        # holds local train metrics, e.g., {"LesCorts": {1: {"MSE": 0.5}}}, i.e., client->round->{metrics->values}
        self.local_train_metrics: Dict[Union[str, int], Dict[int, Dict[str, float]]] = dict()

        self.local_test_losses: Dict[Union[str, int], Dict[int, float]] = dict()  # the same type as local_train_losses
        self.local_test_metrics: Dict[
            Union[str, int], Dict[int, Dict[str, float]]] = dict()  # the same type as local_test_metrics

        self.global_train_losses: List[float] = []
        self.global_train_metrics: Dict[str, List[float]] = dict()

        self.global_test_losses: List[float] = []
        self.global_test_metrics: Dict[str, List[float]] = dict()

    def add_local_train_loss(self, clients_losses: Dict[Union[str, int], float], fl_round: int) -> None:
        """Add one local train loss entry."""
        for client in clients_losses:
            if client not in self.local_train_losses:
                self.local_train_losses[client] = dict()
            self.local_train_losses[client][fl_round] = clients_losses[client]

    def add_local_train_metrics(self, client_metrics: Dict[str, Dict[str, float]], fl_round: int) -> None:
        """Add one local train metrics entry."""
        for client in client_metrics:
            if client not in self.local_train_metrics:
                self.local_train_metrics[client] = dict()
            self.local_train_metrics[client][fl_round] = client_metrics[client]

    def add_local_test_loss(self, client_losses: Dict[Union[str, int], float], fl_round: int) -> None:
        """Add one local test loss entry."""
        for client in client_losses:
            if client not in self.local_test_losses:
                self.local_test_losses[client] = dict()
            self.local_test_losses[client][fl_round] = client_losses[client]

    def add_local_test_metrics(self, client_metrics: Dict[str, Dict[str, float]], fl_round: int) -> None:
        """Add one local test metrics entry."""
        for client in client_metrics:
            if client not in self.local_test_metrics:
                self.local_test_metrics[client] = dict()
            self.local_test_metrics[client][fl_round] = client_metrics[client]

    def add_global_train_losses(self, averaged_loss: float) -> None:
        """Add one global train loss."""
        self.global_train_losses.append(averaged_loss)

    def add_global_train_metrics(self, averaged_metrics: Dict[str, float]) -> None:
        """Add one global train metrics entry."""
        for key in averaged_metrics:
            if key not in self.global_train_metrics:
                self.global_train_metrics[key] = []
            self.global_train_metrics[key].append(averaged_metrics[key])

    def add_global_test_losses(self, averaged_loss: float) -> None:
        """Add one global train loss."""
        self.global_test_losses.append(averaged_loss)

    def add_global_test_metrics(self, averaged_metrics: Dict[str, float]) -> None:
        """Add one global test metrics entry."""
        for key in averaged_metrics:
            if key not in self.global_test_metrics:
                self.global_test_metrics[key] = []
            self.global_test_metrics[key].append(averaged_metrics[key])

    def __repr__(self) -> str:
        rep = ""

        if self.local_train_losses:
            rep += "\nHistory (client, train losses):\n"
            for client, history in self.local_train_losses.items():
                rep += f"\t{client}: {history}\n"
        if self.local_train_metrics:
            rep += "\nHistory (client, train metrics):\n"
            for client, history in self.local_train_metrics.items():
                rep += f"\t{client}: {history}\n"

        if self.local_test_losses:
            rep += "\nHistory (client, test losses):\n"
            for client, history in self.local_test_losses.items():
                rep += f"\t{client}: {history}\n"
        if self.local_test_metrics:
            rep += "\nHistory (client, test metrics):\n"
            for client, history in self.local_test_metrics.items():
                rep += f"\t{client}: {history}\n"

        if self.global_train_losses:
            rep += "\nHistory (global averaged train losses):\n"
            rep += f"\t{self.global_train_losses}\n"
        if self.global_train_metrics:
            rep += "\nHistory (global averaged train metrics):\n"
            for metric, values in self.global_train_metrics.items():
                rep += f"\t{metric}: {values}\n"

        if self.global_test_losses:
            rep += "\nHistory (global averaged test losses):\n"
            rep += f"\t{self.global_test_losses}\n"
        if self.global_test_metrics:
            rep += "\nHistory (global averaged test metrics):\n"
            for metric, values in self.global_test_metrics.items():
                rep += f"\t{metric}: {values}\n"

        return rep
