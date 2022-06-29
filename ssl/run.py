# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from argparse import Namespace

from loguru import logger

from ssl.parser import get_args
from ssl.train_eval import train_eval
from ssl.misc import (
    draw_classification_table_results,
    draw_forecasting_table_results,
    draw_anomaly_table_results,
)
from ssl.utils import init_dl_program


def run_pipeline(args: Namespace):
    logger.info(f"Dataset: {args.dataset}")
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length,
        queue_size=args.queue_size,
        similarities=args.similarity,
    )
    results = []
    for i, dataset in enumerate(args.dataset):
        train_res = [dataset]
        config.update(max_seq_len=int(args.max_seq_len[i]))

        train_res.extend(train_eval(args, device, config, dataset))
        results.append(train_res)

    if args.run_name in ["UCR", "UEA"] and args.eval:
        draw_classification_table_results(results)

    if "forecast" in args.run_name and args.eval:
        draw_forecasting_table_results(results)

    if "anomaly" in args.run_name and args.eval:
        draw_anomaly_table_results(results)


if __name__ == "__main__":
    args = get_args()
    logger.info(f"Arguments: {str(args)}")

    run_pipeline(args)
