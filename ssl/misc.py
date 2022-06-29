# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import List

from loguru import logger
from tabulate import tabulate


def str2bool(arg):
    if isinstance(arg, bool):
        return arg
    elif isinstance(arg, str):
        if arg.lower() in ["true", "1", "t"]:
            return True
        elif arg.lower() in ["false", "0", "f"]:
            return False
    else:
        raise ValueError("Bad argument!")


def draw_classification_table_results(results: List[List]) -> None:
    avg_acc, avg_auprc = 0.0, 0.0
    for idx, res in enumerate(results):
        avg_acc += res[1]
        avg_auprc += res[2]
    avg_acc /= idx + 1
    avg_auprc /= idx + 1
    results.append(["All", avg_acc, avg_auprc, "-", "-"])
    headers = ["dataset", "acc", "auprc", "num epoch", "data shape"]
    logger.info(f"\n{tabulate(results, headers, tablefmt='github')}")


def draw_anomaly_table_results(results: List[List]) -> None:
    headers = ["dataset"]
    table_results = ["kpi"]
    for metric, value in results[0][1].items():
        headers.append(metric)
        table_results.append(value)
    logger.info(f"\n{tabulate([table_results], headers, tablefmt='github')}")


def draw_forecasting_table_results(results: List[List]) -> None:
    dataset = results[0][0]
    eval_results = results[0][1]

    table_results = []
    for seq_len, value in eval_results["ours"].items():
        for key, metrics in value.items():
            if key == "norm":
                table_results.append(
                    [
                        dataset,
                        seq_len,
                        metrics["MSE"],
                        metrics["MAE"],
                        eval_results["lr_train_time"][seq_len],
                        eval_results["lr_infer_time"][seq_len],
                    ]
                )
    headers = ["dataset", "H (sequence length)", "MSE", "MAE", "lr_train_time", "lr_infer_time"]
    logger.info(f"\n{tabulate(table_results, headers, tablefmt='github')}")
