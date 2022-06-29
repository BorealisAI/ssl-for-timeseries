# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse

from ssl.misc import str2bool


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="The dataset name", nargs="+", required=True)
    parser.add_argument(
        "--run-name",
        help="The folder name used to save model, output and evaluation metrics. This can be"
        " set to any word",
    )
    parser.add_argument(
        "--loader",
        type=str,
        required=True,
        help="The data loader used to load the experimental data. This can be set to UCR, UEA,"
        " forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="The gpu no. used for training and inference (defaults to 0)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="The batch size (defaults to 8)")
    parser.add_argument(
        "--lr", type=float, default=0.001, help="The learning rate (defaults to 0.001)"
    )
    parser.add_argument(
        "--repr-dims", type=int, default=320, help="The representation dimension (defaults to 320)"
    )
    parser.add_argument(
        "--max-train-length",
        type=int,
        default=3000,
        help="For sequence with a length greater than <max_train_length>, it would be cropped into"
        " some sequences, each of which has a length less than <max_train_length>"
        " (defaults to 3000)",
    )
    parser.add_argument("--iters", type=int, default=None, help="The number of iterations")
    parser.add_argument("--epochs", type=int, default=None, help="The number of epochs")
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save the checkpoint every <save_every> iterations/epochs",
    )
    parser.add_argument("--seed", type=int, default=None, help="The random seed")
    parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        help="The maximum allowed number of threads used by this process",
    )
    parser.add_argument(
        "--eval", action="store_true", help="Whether to perform evaluation after training"
    )
    parser.add_argument(
        "--irregular",
        type=float,
        default=0,
        help="The ratio of missing observations (defaults to 0)",
    )
    parser.add_argument(
        "--queue-size", type=int, help="What queue size to use in the model", default=2048
    )
    parser.add_argument(
        "--max-seq-len", nargs="+", help="The maximum sequence length in the data", required=True
    )
    parser.add_argument("--alpha", type=float, help="alpha to use for loss")
    parser.add_argument(
        "--similarity",
        help="The type of similarity to use in the architecture",
        nargs="+",
        required=True,
    )
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hierarchical", type=str2bool, default=True)
    parser.add_argument("--run-dir", type=str, default="/results/atom-ssl-ts/runs/")
    args = parser.parse_args()

    return args
