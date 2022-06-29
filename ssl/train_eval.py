# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from argparse import Namespace
from typing import Dict, List

from loguru import logger
from torch import device

from ssl.loader import loader
from ssl.ts2vec import TS2Vec
from ssl.utils import (
    name_with_datetime,
    save_checkpoint_callback,
)


def train_eval(args: Namespace, device: device, config: Dict, dataset: str) -> List:
    logger.info("Loading data... ")
    task_type, data_dict, train_data = loader.load_data(dataset, args.loader, args.irregular)

    run_dir = os.path.join(args.run_dir, dataset + "__" + name_with_datetime(args.run_name))
    os.makedirs(run_dir, exist_ok=True)

    if args.save_every is not None:
        unit = "epoch" if args.epochs is not None else "iter"
        config[f"after_{unit}_callback"] = save_checkpoint_callback(run_dir, args.save_every, unit)

    model = TS2Vec(
        input_dims=train_data.shape[-1],
        device=device,
        alpha=args.alpha,
        temperature=args.temperature,
        hierarchical=args.hierarchical,
        run_dir=run_dir,
        task_type=task_type,
        irregular=args.irregular,
        **config,
    )
    logger.info("Training...")
    model.fit(train_data, data_dict, n_epochs=args.epochs, n_iters=args.iters)
    model.save(f"{run_dir}/model.pkl")

    res = []
    if args.eval:
        logger.info("Evaluating...")
        eval_res = model.eval(train_data, data_dict)

        if args.run_name in ["UCR", "UEA"]:
            res.append(eval_res["acc"])
            res.append(eval_res["auprc"])
            res.append(model.n_epochs)
            res.append(train_data.shape)

        else:
            res.append(eval_res)

    return res
