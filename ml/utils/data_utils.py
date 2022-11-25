"""
Utilities for data pre-processing.
"""

import copy
import math
import warnings
from logging import INFO
from typing import List, Tuple, Dict, Any, Union, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import gmean
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from torch.utils.data import TensorDataset, DataLoader

from ml.utils.logger import log

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def read_data(data_path: str, filter_data: str = None) -> pd.DataFrame:
    """Reads a .csv file."""
    df = pd.read_csv(data_path)
    if filter_data is not None:
        log(INFO, f"Reading {filter_data}'s data...")
        df = df.loc[df['District'] == filter_data]
    df.set_index(pd.DatetimeIndex(df["time"]), inplace=True)
    df.drop(["time"], axis=1, inplace=True)
    cols = [col for col in df.columns if col != "District"]
    df[cols] = df[cols].astype("float32")

    return df


def handle_nans(train_data: pd.DataFrame,
                val_data: Optional[pd.DataFrame] = None,
                constant: Optional[int] = 0,
                identifier: Optional[str] = "District") -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Imputes missing values in a dataframe. Currently only constant imputation is supported."""
    if val_data is not None:
        assert list(train_data.columns) == list(val_data.columns)
        val_data = val_data.copy()
    train_data = train_data.copy()

    columns = list(train_data.columns)
    if identifier in columns:
        columns.remove(identifier)

    imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=constant)

    for col in columns:
        train_nans = get_nans(train_data[[col]])
        if train_nans.values > 0:
            tmp_imp = copy.deepcopy(imp)
            tmp_imp.fit(train_data[[col]])
            train_data[col] = tmp_imp.transform(train_data[[col]])
        if val_data is not None:
            val_nans = get_nans(train_nans[[col]])
            if val_nans.values > 0:
                tmp_imp = copy.deepcopy(imp)
                tmp_imp = copy.deepcopy(tmp_imp)
                tmp_imp.fit(val_data[[col]])
                val_data[col] = tmp_imp.transform(val_data[[col]])

    if val_data is not None:
        return train_data, val_data
    return train_data


def get_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Computes the NaN values per column in a dataframe."""
    return df.isna().sum()


def floor_cap_transform(df: pd.DataFrame,
                        columns: Union[List[str], None] = None,
                        identifier: Union[str, None] = None,
                        kwargs: Tuple[int, int] = None) -> Union[pd.DataFrame, np.ndarray]:
    """Imputes the values of points that are less than the specified low percentile with the value of the
     low percentile and the values of points that are greater the specified percentile with the value of the high
     percentile. By default, the low percentile is fixed to 10 and the high percentile to 90."""
    if kwargs is None:
        low_percentile, high_percentile = 10, 90
    else:
        low_percentile, high_percentile = kwargs[0], kwargs[1]

    df = df.copy()
    if columns is not None:
        for col in df.columns:
            if col not in columns or col == identifier:
                continue
            points = df[[col]].values
            low_percentile_val = np.percentile(points, low_percentile)
            high_percentile_val = np.percentile(points, high_percentile)
            # the data points that are less than the 10th percentile are replaced with the 10th percentile value
            b = np.where(points < low_percentile_val, low_percentile_val, points)
            # the data points that are greater than the 90th percentile are replaced with 90th percentile value
            b = np.where(b > high_percentile_val, high_percentile_val, b)
            df[col] = b
        return df
    else:
        points = df.values
        low_percentile_val = np.percentile(points, low_percentile)
        high_percentile_val = np.percentile(points, high_percentile)
        b = np.where(points < low_percentile_val, low_percentile_val, points)
        b = np.where(b > high_percentile_val, high_percentile_val, b)
        return b


def handle_outliers(df: pd.DataFrame,
                    columns: List[str] = ["down", "up"],
                    identifier: str = "District",
                    kwargs: Dict[str, Tuple] = {"ElBorn": (10, 90), "LesCorts": (10, 90), "PobleSec": (10, 90)},
                    exclude: List[str] = None) -> pd.DataFrame:
    log(INFO, f"Using Flooring and Capping and with params: {kwargs}")
    dfs = []
    for area in df[identifier].unique():
        tmp_df = df.loc[df[identifier] == area].copy()
        if kwargs is not None:
            assert area in list(kwargs.keys())
            area_kwargs = kwargs[area]
        else:
            area_kwargs = None

        if exclude is not None and area in exclude:
            dfs.append(tmp_df)
            continue

        tmp_df = floor_cap_transform(tmp_df, columns, identifier, area_kwargs)
        dfs.append(tmp_df)

    df = pd.concat(dfs, ignore_index=False)

    return df


def generate_time_lags(df: pd.DataFrame,
                       n_lags: int = 10,
                       identifier: str = "District",
                       is_y: bool = False) -> pd.DataFrame:
    """Transforms a dataframe to time lags using the shift method.
    If the shifting operation concerns the targets, then lags removal is applied, i.e., only the measurements that
    we try to predict are kept in the dataframe. If the shifting operation concerns the previous time steps (our actual
    input), then measurements removal is applied, i.e., the measurements in the first lag are being removed since they
    are the targets that we try to predict."""
    columns = list(df.columns)
    dfs = []

    for area in df[identifier].unique():
        df_area = df.loc[df[identifier] == area]
        df_n = df_area.copy()

        for n in range(1, n_lags + 1):
            for col in columns:
                if col == "time" or col == identifier:
                    continue
                df_n[f"{col}_lag-{n}"] = df_n[col].shift(n).replace(np.NaN, 0).astype("float64")
        df_n = df_n.iloc[n_lags:]

        dfs.append(df_n)
    df = pd.concat(dfs, ignore_index=False)

    if is_y:
        df = df[columns]
    else:
        if identifier in columns:
            columns.remove(identifier)
        df = df.loc[:, ~df.columns.isin(columns)]
        df = df[df.columns[::-1]]  # reverse order, e.g. lag-1, lag-2 to lag-2, lag-1.

    return df


def time_to_feature(df: pd.DataFrame,
                    use_time_features: bool = True,
                    identifier: str = "District") -> Union[pd.DataFrame, None]:
    """Transforms datetime to actual features using a cyclical representation, i.e., sin(x) and cos(x).
    Here, we only use the hour and minute as features. Uncomment the corresponding datetime feature if you plan to
    use additional information. We term the datetime features as exogenous since they will not be provided as
    input along with the time series, but they will only be used before the final prediction, i.e., after operating on
    time-series."""
    if not use_time_features:
        return None

    df = df.copy()
    columns = list(df.columns)
    if identifier in columns:
        columns.remove(identifier)

    df = (
        df
        # .assign(year=df.index.year)  # year, e.g., 2018
        # .assign(month=df.index.month)  # month, 1 - 12
        # .assign(week=df.index.isocalendar().week.astype(int))  # week of year
        # .assign(day=df.index.day)  # day of month, e.g. 1-31
        .assign(hour=df.index.hour)  # hour of day
        .assign(minute=df.index.minute)  # minute of day
        # .assign(dayofyear=df.index.dayofyear)  # day of year
        # .assign(dayofweek=df.index.dayofweek)  # day of week
    )

    df = to_cyclical(df,
                     {"month": 12, "week": 52, "day": 31, "hour": 24, "minute": 60, "dayofyear": 365, "dayofweek": 7})

    df = df.drop(columns, axis=1)

    return df


def to_cyclical(df: pd.DataFrame,
                col_names_period: Dict[str, int],
                start_num: int = 0) -> pd.DataFrame:
    """Transforms date to cyclical representation, i.e., to cos(x) and sin(x).
    See http://blog.davidkaleko.com/feature-engineering-cyclical-features.html."""
    for col_name in col_names_period:
        if col_name not in df.columns:
            continue
        if col_name == "month" or col_name == "day":
            start_num = 1
        else:
            start_num = 0
        kwargs = {
            f"sin_{col_name}": lambda x: np.sin((df[col_name] - start_num) * (2 * np.pi / col_names_period[col_name])),
            f"cos_{col_name}": lambda x: np.cos((df[col_name] - start_num) * (2 * np.pi / col_names_period[col_name]))
        }
        df = df.assign(**kwargs).drop(columns=[col_name])
    return df


def assign_statistics(X: pd.DataFrame,
                      stats: List[str],
                      lags: int = 10,
                      targets: List[str] = ["up", "down"],
                      identifier: str = "District") -> Union[pd.DataFrame, None]:
    """Assigns the defined statistics as exogenous data. These statistics describe the time series used as input as a
    whole. For example, if we use 10 time lags, then the assigned statistics describe the previous 10 observations."""
    X = X.copy()
    if isinstance(stats, list):
        if next(iter(stats)) is None:
            return None
    elif stats is None:
        return None

    X = X.copy()
    cols = list(X.columns)
    if identifier in cols:
        cols.remove(identifier)

    for col in targets:
        tmp_col_lags = [f"{col}_lag-{x}" for x in range(1, lags + 1)]
        tmp_X: pd.DataFrame = X[[c for c in X.columns if c in tmp_col_lags]]

        if "mean" in stats:
            kwargs = {f"mean_{col}": tmp_X.mean(axis=1)}
            X = X.assign(**kwargs)
        if "median" in stats:
            kwargs = {f"median_{col}": tmp_X.median(axis=1)}
            X = X.assign(**kwargs)
        if "std" in stats:
            kwargs = {f"std_{col}": tmp_X.std(axis=1)}
            X = X.assign(**kwargs)
        if "variance" in stats:
            kwargs = {f"variance_{col}": tmp_X.var(axis=1)}
            X = X.assign(**kwargs)
        if "kurtosis" in stats:
            kwargs = {f"kurtosis_{col}": tmp_X.kurt(axis=1)}
            X = X.assign(**kwargs)
        if "skew" in stats:
            kwargs = {f"skew_{col}": tmp_X.skew(axis=1)}
            X = X.assign(**kwargs)
        if "gmean" in stats:
            kwargs = {f"gmean_{col}": tmp_X.apply(gmean, axis=1)}
            X = X.assign(**kwargs)

    X = X.drop(cols, axis=1)

    return X


def to_timeseries_rep(x: Union[np.ndarray, Dict[Union[str, int], np.ndarray]],
                      num_lags: int = 10,
                      num_features: int = 11) -> Union[np.ndarray, Dict[Union[str, int], np.ndarray]]:
    """Transforms a dataframe to timeseries representation."""
    if isinstance(x, np.ndarray):
        x_reshaped = x.reshape((len(x), num_lags, num_features, -1))
        return x_reshaped
    else:
        xs_reshaped = dict()
        for cid in x:
            tmp_x = x[cid]
            xs_reshaped[cid] = tmp_x.reshape((len(tmp_x), num_lags, num_features, -1))
        return xs_reshaped


def remove_identifiers(X_train: pd.DataFrame,
                       y_train: pd.DataFrame,
                       X_val: Optional[pd.DataFrame] = None,
                       y_val: Optional[pd.DataFrame] = None,
                       identifier: str = "District") -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]:
    """Removes a specified column which describes an area/client, e.g., id."""
    X_train = X_train.drop([identifier], axis=1)
    y_train = y_train.drop([identifier], axis=1)

    if X_val is not None and y_val is not None:
        X_val = X_val.drop([identifier], axis=1)
        y_val = y_val.drop([identifier], axis=1)
        return X_train, y_train, X_val, y_val
    return X_train, y_train


def scale_features(train_data: pd.DataFrame,
                   scaler,
                   val_data: Optional[pd.DataFrame] = None,
                   per_area: bool = False,
                   identifier: str = "District") -> Union[pd.DataFrame, pd.DataFrame, Any]:
    """Scales the features according to the specified scaler."""
    train_data = train_data.copy()
    if scaler is None:
        return train_data, val_data, None
    if isinstance(scaler, str):
        if val_data is not None:
            val_data = val_data.copy()
        if scaler == "minmax":
            scaler = MinMaxScaler()
        elif scaler == "standard":
            scaler = StandardScaler()
        elif scaler == "robust":
            scaler = RobustScaler()
        elif scaler == "maxabs":
            scaler = MaxAbsScaler()
        else:
            return train_data, val_data, None

    cols = [col for col in train_data.columns if col != identifier]
    if per_area:
        scalers = dict()
        train_dfs, val_dfs = [], []
        for area in train_data[identifier].unique():
            tmp_train_data = train_data.loc[train_data[identifier] == area]
            tmp_train_data = tmp_train_data.copy()
            tmp_train_data[cols] = scaler.fit_transform(tmp_train_data[cols])

            tmp_val_data = val_data.loc[val_data[identifier] == area]
            tmp_val_data = tmp_val_data.copy()
            tmp_val_data[cols] = scaler.transform(tmp_val_data[cols])
            train_dfs.append(tmp_train_data)
            val_dfs.append(tmp_val_data)

            scalers[area] = scaler

        train_data = pd.concat(train_dfs)
        val_data = pd.concat(val_dfs)

        return train_data, val_data, scalers

    else:
        if val_data is not None:
            train_data[cols] = scaler.fit_transform(train_data[cols])
            val_data[cols] = scaler.transform(val_data[cols])
            return train_data, val_data, scaler
        else:
            train_data[cols] = scaler.transform(train_data[cols])
            return train_data


def to_train_val(df: pd.DataFrame,
                 train_size: float = 0.8,
                 identifier: str = "District") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the original dataframe to train and validation frames."""
    if df[identifier].nunique() != 1:
        train_data, val_data = [], []
        for area in df[identifier].unique():
            area_data = df.loc[df[identifier] == area]
            num_samples_train = math.ceil(train_size * len(area_data))
            area_train_data = area_data[:num_samples_train]
            area_val_data = area_data[num_samples_train:]
            train_data.append(area_train_data)
            val_data.append(area_val_data)
            log(INFO, f"Observations info in {area}")
            log(INFO, f"\tTotal number of samples:  {len(area_data)}")
            log(INFO, f"\tNumber of samples for training: {num_samples_train}")
            log(INFO, f"\tNumber of samples for validation:  {len(area_data) - num_samples_train}")
        train_data = pd.concat(train_data)
        val_data = pd.concat(val_data)
        log(INFO, "Observations info using all data")
        log(INFO, f"\tTotal number of samples:  {len(train_data) + len(val_data)}")
        log(INFO, f"\tNumber of samples for training: {len(train_data)}")
        log(INFO, f"\tNumber of samples for validation:  {len(val_data)}")
    else:
        num_samples_train = math.ceil(train_size * len(df))
        log(INFO, f"\tTotal number of samples:  {len(df)}")
        log(INFO, f"\tNumber of samples for training: {num_samples_train}")
        log(INFO, f"\tNumber of samples for validation:  {len(df) - num_samples_train}")
        train_data = df[:num_samples_train]
        val_data = df[num_samples_train:]

    return train_data, val_data


def to_Xy(train_data: pd.DataFrame,
          targets: List[str],
          val_data: Optional[pd.DataFrame] = None,
          ignore_cols: Optional[List[str]] = None,
          identifier: str = "District") -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]:
    """Generates the features and targets for the training and validation sets
     or the features and targets for the testing set."""
    targets = copy.deepcopy(targets)
    if identifier not in targets:
        targets.append(identifier)

    y_train = train_data[targets]

    y_val = None
    if val_data is not None:
        y_val = val_data[targets]

    if ignore_cols is None:
        ignore_cols = []

    if identifier in ignore_cols:
        ignore_cols.remove(identifier)

    X_train = train_data

    if ignore_cols is not None and len(ignore_cols) > 0 and next(iter(ignore_cols)) is not None:
        cols = X_train.columns
        for ignore_col in ignore_cols:
            if ignore_col in cols:
                X_train = X_train.drop([ignore_col], axis=1)

    if val_data is not None:
        X_val = val_data
        if ignore_cols is not None and len(ignore_cols) > 0 and next(iter(ignore_cols)) is not None:
            cols = X_val.columns
            for ignore_col in ignore_cols:
                if ignore_col in cols:
                    X_val = X_val.drop([ignore_col], axis=1)

    else:
        return X_train, y_train

    return X_train, X_val, y_train, y_val


def get_data_by_area(X_train: pd.DataFrame, X_val: pd.DataFrame,
                     y_train: pd.DataFrame, y_val: pd.DataFrame,
                     identifier: str = "District") -> Tuple[
    Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Generates the training and testing frames per area."""
    assert list(X_train[identifier].unique()) == list(X_val[identifier].unique())

    area_X_train, area_X_val, area_y_train, area_y_val = dict(), dict(), dict(), dict()
    for area in X_train[identifier].unique():
        # get per area observations
        X_train_area = X_train.loc[X_train[identifier] == area]
        X_val_area = X_val.loc[X_val[identifier] == area]
        y_train_area = y_train.loc[y_train[identifier] == area]
        y_val_area = y_val.loc[y_val[identifier] == area]

        area_X_train[area]: pd.DataFrame = X_train_area
        area_X_val[area]: pd.DataFrame = X_val_area
        area_y_train[area]: pd.DataFrame = y_train_area
        area_y_val[area]: pd.DataFrame = y_val_area

    assert area_X_train.keys() == area_X_val.keys() == area_y_train.keys() == area_y_val.keys()

    return area_X_train, area_X_val, area_y_train, area_y_val


def get_exogenous_data_by_area(exogenous_data_train: pd.DataFrame,
                               exogenous_data_val: Optional[pd.DataFrame] = None,
                               identifier: Optional[str] = "District") -> Union[
    Tuple[Dict[Union[str, int], pd.DataFrame], Dict[Union[str, int], pd.DataFrame]],
    Dict[Union[str, int], pd.DataFrame]]:
    """Generates the exogenous data per area."""
    if exogenous_data_val is not None:
        assert list(exogenous_data_val[identifier].unique()) == list(exogenous_data_train[identifier].unique())
    area_exogenous_data_train, area_exogenous_data_val = dict(), dict()
    for area in exogenous_data_train[identifier].unique():
        # get per area observations
        exogenous_data_train_area = exogenous_data_train.loc[exogenous_data_train[identifier] == area]
        area_exogenous_data_train[area]: pd.DataFrame = exogenous_data_train_area
        if exogenous_data_val is not None:
            exogenous_data_val_area = exogenous_data_val.loc[exogenous_data_val[identifier] == area]
            area_exogenous_data_val[area]: pd.DataFrame = exogenous_data_val_area

    if exogenous_data_val is not None:
        assert area_exogenous_data_train.keys() == area_exogenous_data_val.keys()
    for area in area_exogenous_data_train:
        area_exogenous_data_train[area] = area_exogenous_data_train[area].drop([identifier], axis=1)
        if exogenous_data_val is not None:
            area_exogenous_data_val[area] = area_exogenous_data_val[area].drop([identifier], axis=1)

    if exogenous_data_val is None:
        return area_exogenous_data_train

    return area_exogenous_data_train, area_exogenous_data_val


def to_torch_dataset(X: np.ndarray, y: np.ndarray,
                     num_lags: int = 10,
                     num_features: int = 11,
                     indices: List[int] = [8, 3, 1, 10, 9],
                     batch_size: int = 32,
                     exogenous_data: Optional[np.ndarray] = None,
                     shuffle: bool = False) -> torch.utils.data.DataLoader:
    """Transforms X and y to torch dataset. The dataloader holds 4 different features:
        1. Past observations or x features that will be fed as input to the specified model.
        2. Exogenous data (if specified) such as datetime or statistic-based features. If exogenous data
            are not defined, then the returned value is an empty list.
        3. The past y targets if a teacher-forcing model is applied. In the current version, only the Dual Attention
            AutoEncoder can be used with the teacher-forcing method. In other words, this value hold the last targets
            and forces the model to use them along with the current features to make the prediction.
        4. The future observations which act as targets y.
        """
    X = torch.tensor(X).float()
    y = torch.tensor(y).float()
    if exogenous_data is not None:
        exogenous_data = torch.tensor(exogenous_data).float()
    tensor_dataset = TimeSeriesDataset(X, y, num_lags, num_features, indices, exogenous_data)
    loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


class TimeSeriesDataset(torch.utils.data.Dataset):
    """Dataset wrapper. This class can handle either vector or matrices."""

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 num_lags: int = 10,
                 num_features: int = 11,
                 indices: List[int] = [8, 3, 2, 10, 9],
                 exogenous: Optional[np.ndarray] = None):
        if exogenous is not None:
            assert X.size(0) == y.size(0) == exogenous.size(0), "Size mismatch between tensors"
        else:
            assert X.size(0) == y.size(0), "Size mismatch between tensors"
        self.X = X
        self.y = y
        self.num_features = num_features
        self.num_lags = num_lags
        self.indices = indices
        self.exogenous = exogenous

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, index):
        if index == 0:
            tmp_X = self.X[index]
            if len(self.X.shape) < 3:
                tmp_X = tmp_X.view(self.num_lags, self.num_features, 1)
            y_hist = []
            for i, lag in enumerate(tmp_X):
                if i == 0:
                    pad = torch.zeros_like(lag[self.indices])
                    y_hist.append(pad.reshape(1, -1))
                else:
                    y_hist.append(tmp_X[i - 1][self.indices].reshape(1, -1))
            y_hist = torch.cat(y_hist)

        elif index < self.num_lags + 1:
            last_obs = self.X[index - 1]
            if len(self.X.shape) < 3:
                last_obs = last_obs.view(self.num_lags, self.num_features, 1)
            y_hist = []
            for i, lag in enumerate(last_obs):
                y_hist.append(last_obs[i][self.indices].reshape(1, -1))
            y_hist = torch.cat(y_hist)
        else:
            y_hist = self.y[index - self.num_lags - 1: index - 1]

        if self.exogenous is None:
            return self.X[index], [], y_hist, self.y[index]
        else:
            return self.X[index], self.exogenous[index], y_hist, self.y[index]
