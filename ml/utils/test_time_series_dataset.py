"""
Test time-series dataset representation.
"""

import sys

from pathlib import Path

parent = Path(__file__).resolve().parents[2]
if parent not in sys.path:
    sys.path.insert(0, str(parent))

import numpy as np

from ml.utils.data_utils import to_torch_dataset


def test_timeseries_dataset():
    X_train = []
    y_train = []
    c1, c2 = 1, 3
    for i in range(11):
        tmp_x, tmp_y = [], []
        for j in range(c1, c2):
            tmp_x.append(j)
            tmp_y.append(j + 1)
            c1 = c2
            c2 += 2
        X_train.append(tmp_x)
        y_train.append(tmp_y)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(X_train.shape)
    print(y_train.shape)

    print(len(X_train), len(y_train))

    print(X_train)
    print(y_train)

    loader = to_torch_dataset(X_train, y_train, num_lags=1,
                              num_features=2,
                              indices=[0],
                              batch_size=1, shuffle=False)
    for x, _, _, y in loader:
        print(x.shape, y.shape)
        print(x, y)
    return loader


if __name__ == "__main__":
    test_timeseries_dataset()
