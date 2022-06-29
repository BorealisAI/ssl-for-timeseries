# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Tuple, Dict

import numpy as np

from ssl import constants as c
from ssl.loader import datautils
from ssl.utils import data_dropout


def load_classification_data(dataset: str, loader: str) -> Tuple[str, Dict[str, np.ndarray]]:
    if loader == c.CLASSIFICATION.UCR:
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(dataset)

    elif loader == c.CLASSIFICATION.UEA:
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(dataset)

    else:
        raise KeyError(
            f"{dataset} and {loader} are not valid. Have you specified the correct names?"
        )

    return (
        c.CLASSIFICATION.TASK,
        {
            c.CLASSIFICATION.TRAIN_DATA: train_data,
            c.CLASSIFICATION.TRAIN_LABEL: train_labels,
            c.CLASSIFICATION.TEST_DATA: test_data,
            c.CLASSIFICATION.TEST_LABEL: test_labels,
        },
    )


def load_forecasting_data(dataset: str, loader: str) -> Tuple[str, Dict[str, np.ndarray]]:
    if loader in [c.FORECASTING.CSV, c.FORECASTING.CSV_UNIVAR]:
        is_univar = False if loader == c.FORECASTING.CSV else True
        (
            data,
            train_slice,
            valid_slice,
            test_slice,
            scaler,
            pred_lens,
            n_covariate_cols,
        ) = datautils.load_forecast_csv(dataset, univar=is_univar)

    elif loader in [c.FORECASTING.NUMPY, c.FORECASTING.NUMPY_UNIVAR]:
        is_univar = False if loader == c.FORECASTING.NUMPY else True
        (
            data,
            train_slice,
            valid_slice,
            test_slice,
            scaler,
            pred_lens,
            n_covariate_cols,
        ) = datautils.load_forecast_npy(dataset, univar=is_univar)

    else:
        raise KeyError(
            f"{dataset} and {loader} are not valid. Have you specified the correct names?"
        )

    return (
        c.FORECASTING.TASK,
        {
            c.FORECASTING.DATA: data,
            c.FORECASTING.TRAIN_SLICE: train_slice,
            c.FORECASTING.VALID_SLICE: valid_slice,
            c.FORECASTING.TEST_SLICE: test_slice,
            c.FORECASTING.SCALAR: scaler,
            c.FORECASTING.PRED_LENS: pred_lens,
            c.FORECASTING.NUM_COV_COL: n_covariate_cols,
        },
    )


def load_anomaly_data(dataset: str, loader: str) -> Tuple[str, Dict[str, np.ndarray]]:
    (
        all_train_data,
        all_train_labels,
        all_train_timestamps,
        all_test_data,
        all_test_labels,
        all_test_timestamps,
        delay,
    ) = datautils.load_anomaly(dataset)

    return (
        c.ANOMALY_DETECTION.TASK,
        {
            c.ANOMALY_DETECTION.TRAIN_DATA: all_train_data,
            c.ANOMALY_DETECTION.TRAIN_LABEL: all_train_labels,
            c.ANOMALY_DETECTION.TRAIN_TIMESTAMP: all_train_timestamps,
            c.ANOMALY_DETECTION.TEST_DATA: all_test_data,
            c.ANOMALY_DETECTION.TEST_LABEL: all_test_labels,
            c.ANOMALY_DETECTION.TEST_TIMESTAMP: all_test_timestamps,
            c.ANOMALY_DETECTION.DELAY: delay,
        },
    )


_mapping = {
    c.CLASSIFICATION.UCR: load_classification_data,
    c.CLASSIFICATION.UEA: load_classification_data,
    c.FORECASTING.CSV: load_forecasting_data,
    c.FORECASTING.CSV_UNIVAR: load_forecasting_data,
    c.FORECASTING.NUMPY: load_forecasting_data,
    c.FORECASTING.NUMPY_UNIVAR: load_forecasting_data,
    c.ANOMALY_DETECTION.ANOMALY: load_anomaly_data,
    c.ANOMALY_DETECTION.COLD_START_ANOMALY: load_anomaly_data,
}


def load_data(dataset: str, loader: str, irregular: float) -> Tuple[str, Dict, np.ndarray]:
    loader_func = _mapping.get(loader)

    if loader_func is None:
        raise KeyError(f"Invalid loader is specified. Got {loader}")

    task_type, data_dict = loader_func(dataset, loader)

    if loader in [c.CLASSIFICATION.UCR, c.CLASSIFICATION.UEA]:
        train_data = data_dict[c.CLASSIFICATION.TRAIN_DATA]
        if irregular > 0:
            train_data = data_dropout(train_data, irregular)

    if loader.startswith("forecast"):
        train_data = data_dict[c.FORECASTING.DATA][:, data_dict[c.FORECASTING.TRAIN_SLICE]]

    if loader == c.ANOMALY_DETECTION.ANOMALY:
        train_data = datautils.gen_ano_train_data(data_dict[c.ANOMALY_DETECTION.TRAIN_DATA])

    if loader == c.ANOMALY_DETECTION.COLD_START_ANOMALY:
        train_data, _, _, _ = datautils.load_UCR("FordA")

    return task_type, data_dict, train_data
