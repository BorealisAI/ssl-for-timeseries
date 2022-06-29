# MIT License
#
# Copyright (c) 2022-present Zhihan Yue
#
import time

import bottleneck as bn
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


# consider delay threshold and missing segments
def get_range_proba(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos : min(pos + delay + 1, sp)]:
                new_predict[pos:sp] = 1
            else:
                new_predict[pos:sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos : min(pos + delay + 1, sp)]:
            new_predict[pos:sp] = 1
        else:
            new_predict[pos:sp] = 0

    return new_predict


def reconstruct_label(timestamp, label):
    timestamp = np.asarray(timestamp, np.int64)
    index = np.argsort(timestamp)

    timestamp_sorted = np.asarray(timestamp[index])
    interval = np.min(np.diff(timestamp_sorted))

    label = np.asarray(label, np.int64)
    label = np.asarray(label[index])

    idx = (timestamp_sorted - timestamp_sorted[0]) // interval

    new_label = np.zeros(
        shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,), dtype=np.int
    )
    new_label[idx] = label

    return new_label


def eval_ad_result(test_pred_list, test_labels_list, test_timestamps_list, delay):
    labels = []
    pred = []
    for test_pred, test_labels, test_timestamps in zip(
        test_pred_list, test_labels_list, test_timestamps_list
    ):
        assert test_pred.shape == test_labels.shape == test_timestamps.shape
        test_labels = reconstruct_label(test_timestamps, test_labels)
        test_pred = reconstruct_label(test_timestamps, test_pred)
        test_pred = get_range_proba(test_pred, test_labels, delay)
        labels.append(test_labels)
        pred.append(test_pred)
    labels = np.concatenate(labels)
    pred = np.concatenate(pred)
    return {
        "f1": f1_score(labels, pred),
        "precision": precision_score(labels, pred),
        "recall": recall_score(labels, pred),
    }


def np_shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def eval_anomaly_detection(
    all_repr,
    all_repr_wom,
    all_train_data,
    all_test_labels,
    all_test_timestamps,
    delay,
):
    t = time.time()
    res_log = []
    labels_log = []
    timestamps_log = []
    for k in all_train_data:
        train_data = all_train_data[k]
        train_length = len(train_data)

        train_err = np.abs(all_repr_wom[k][:train_length] - all_repr[k][:train_length]).sum(axis=1)
        test_err = np.abs(all_repr_wom[k][train_length:] - all_repr[k][train_length:]).sum(axis=1)

        ma = np_shift(bn.move_mean(np.concatenate([train_err, test_err]), 21, min_count=1), 1)
        train_err_adj = (train_err - ma[: len(train_err)]) / ma[: len(train_err)]
        test_err_adj = (test_err - ma[len(train_err) :]) / ma[len(train_err) :]
        train_err_adj = train_err_adj[22:]

        thr = np.mean(train_err_adj) + 4 * np.std(train_err_adj)
        test_res = (test_err_adj > thr) * 1

        for i in range(len(test_res)):
            if i >= delay and test_res[i - delay : i].sum() >= 1:
                test_res[i] = 0

        res_log.append(test_res)
        labels_log.append(all_test_labels[k])
        timestamps_log.append(all_test_timestamps[k])
    t = time.time() - t

    eval_res = eval_ad_result(res_log, labels_log, timestamps_log, delay)
    eval_res["infer_time"] = t
    return res_log, eval_res


def eval_anomaly_detection_coldstart(
    all_data,
    all_repr,
    all_repr_wom,
    all_train_labels,
    all_train_timestamps,
    all_test_labels,
    all_test_timestamps,
    delay,
):
    t = time.time()
    res_log = []
    labels_log = []
    timestamps_log = []

    for k in all_data:
        data = all_data[k]
        labels = np.concatenate([all_train_labels[k], all_test_labels[k]])
        timestamps = np.concatenate([all_train_timestamps[k], all_test_timestamps[k]])

        err = np.abs(all_repr_wom[k] - all_repr[k]).sum(axis=1)
        ma = np_shift(bn.move_mean(err, 21), 1)
        err_adj = (err - ma) / ma

        MIN_WINDOW = len(data) // 10
        thr = bn.move_mean(err_adj, len(err_adj), MIN_WINDOW) + 4 * bn.move_std(
            err_adj, len(err_adj), MIN_WINDOW
        )
        res = (err_adj > thr) * 1

        for i in range(len(res)):
            if i >= delay and res[i - delay : i].sum() >= 1:
                res[i] = 0

        res_log.append(res[MIN_WINDOW:])
        labels_log.append(labels[MIN_WINDOW:])
        timestamps_log.append(timestamps[MIN_WINDOW:])
    t = time.time() - t

    eval_res = eval_ad_result(res_log, labels_log, timestamps_log, delay)
    eval_res["infer_time"] = t
    return res_log, eval_res
