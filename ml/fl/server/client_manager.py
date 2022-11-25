"""
Client manager abstract representation and implementation.
It can register/unregister clients and sample available clients.
"""

import sys

from pathlib import Path

parent = Path(__file__).resolve().parents[3]
if parent not in sys.path:
    sys.path.insert(0, str(parent))

from logging import INFO
import random
from abc import ABC, abstractmethod


from typing import Dict, List, Union

from ml.fl.server.client_proxy import ClientProxy
from ml.utils.logger import log


class ClientManager(ABC):
    """Abstract base class for managing clients."""

    @abstractmethod
    def num_available(self, verbose: bool = True) -> int:
        """Returns the number of available clients."""

    @abstractmethod
    def register(self, client: ClientProxy) -> bool:
        """Register client to the pool of clients."""

    @abstractmethod
    def unregister(self, client: ClientProxy) -> None:
        """Unregister ClientProxy instance."""

    @abstractmethod
    def all(self) -> Dict[Union[str, int], ClientProxy]:
        """Return all available clients."""

    @abstractmethod
    def sample(self, c: float) -> List[ClientProxy]:
        """Sample a number of ClientProxy instances."""


class SimpleClientManager(ClientManager):
    def __init__(self):
        self.clients: Dict[str, ClientProxy] = {}

    def __len__(self):
        return len(self.clients)

    def num_available(self, verbose: bool = True) -> int:
        if verbose:
            log(INFO, f"Number of available clients: {len(self)}")
        return len(self)

    def register(self, client: ClientProxy) -> bool:
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client
        log(INFO, f"Registered client with id: {client.cid}")
        return True

    def unregister(self, client: ClientProxy) -> None:
        if client.cid in self.clients:
            del self.clients[client.cid]
            log(INFO, f"Unregistered client with id: {client.cid}")

    def all(self) -> Dict[Union[str, int], ClientProxy]:
        return self.clients

    def sample(self, c: float) -> List[ClientProxy]:
        available_clients = list(self.clients.keys())
        if len(available_clients) == 0:
            log(INFO, f"Cannot sample clients. The number of available clients is zero.")
            return []
        num_selection = int(c * self.num_available(verbose=False))
        if num_selection == 0:
            num_selection = 1
        if num_selection > self.num_available(verbose=False):
            num_selection = self.num_available(verbose=False)
        sampled_clients = random.sample(available_clients, num_selection)
        log(INFO, f"Parameter c={c}. Sampled {num_selection} client(s): {sampled_clients}")
        return [self.clients[cid] for cid in sampled_clients]
