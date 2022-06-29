# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
class PATH:
    DATASET = "/shared-data"


class SIMILARITY:
    TEMPORAL = "TEMPORAL"
    INSTANCE = "INSTANCE"
    SIMPLE = "SIMPLE"


class CLASSIFICATION:
    TASK = "classification"
    UCR = "UCR"
    UEA = "UEA"
    TRAIN_DATA = "train_data"
    TRAIN_LABEL = "train_labels"
    TEST_DATA = "test_data"
    TEST_LABEL = "test_labels"


class FORECASTING:
    TASK = "forecasting"
    CSV = "forecast_csv"
    CSV_UNIVAR = "forecast_csv_univar"
    NUMPY = "forecast_npy"
    NUMPY_UNIVAR = "forecast_npy_univar"
    DATA = "data"
    TRAIN_SLICE = "train_slice"
    VALID_SLICE = "valid_slice"
    TEST_SLICE = "test_slice"
    SCALAR = "scaler"
    PRED_LENS = "pred_lens"
    NUM_COV_COL = "n_covariate_cols"


class ANOMALY_DETECTION:
    TASK = "anomaly_detection"
    COLD_START_TASK = "anomaly_detection_coldstart"
    ANOMALY = "anomaly"
    COLD_START_ANOMALY = "anomaly_coldstart"
    TRAIN_DATA = "all_train_data"
    TRAIN_LABEL = "all_train_labels"
    TRAIN_TIMESTAMP = "all_train_timestamps"
    TEST_DATA = "all_test_data"
    TEST_LABEL = "all_test_labels"
    TEST_TIMESTAMP = "all_test_timestamps"
    DELAY = "delay"
