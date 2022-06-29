# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright (c) 2022-present Zhihan Yue
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the TS2Vec (https://arxiv.org/pdf/2106.10466.pdf) implementation
# from https://github.com/yuezhihan/ts2vec by Zhihan Yue
####################################################################################

from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import ssl.tasks as tasks
from ssl import constants as c
from ssl.models import AnchorTSEncoder
from ssl.models.encoder import SimilarityType
from ssl.models.losses import (
    KLD,
    InstanceHierarchicalKLD,
    TemporalHierarchicalKLD,
    TemporalInstanceHierarchicalKLD,
)
from ssl.models.similarity import Similarity
from ssl.utils import (
    take_per_row,
    split_with_nan,
    centerize_vary_length_series,
    torch_pad_nan,
    data_dropout,
    pkl_save,
)


class TS2Vec:
    """We adapt the TS2Vec training procedure to work with our proposed model.
    Refer to https://arxiv.org/pdf/2106.10466.pdf for details on TS2Vec.

    Args:
        input_dims (int): The input dimension. For a univariate time series, this should be
            set to 1.
        output_dims (int): The representation dimension.
        hidden_dims (int): The hidden dimension of the encoder.
        depth (int): The number of hidden residual blocks in the encoder.
        device (int): The gpu used for training and inference.
        lr (int): The learning rate.
        batch_size (int): The batch size.
        max_train_length (Union[int, NoneType]): The maximum allowed sequence length for
            training. For sequence with a length greater than <max_train_length>, it would
            be cropped into some sequences, each of which has a length less than
            <max_train_length>.
        temporal_unit (int): The minimum unit to perform temporal contrast. When training on
            a very long sequence, this param helps to reduce the cost of time and memory.
        after_iter_callback (Union[Callable, NoneType]): A callback function that would be
            called after each iteration.
        after_epoch_callback (Union[Callable, NoneType]): A callback function that would be
            called after each epoch.
        hierarchical: Specifies whether to use the hierarchical loss or not.
        queue_size: Specifies the size of the queue to use for the teacher-student network.
        max_seq_len: Specifies the maximum sequence length for a given dataset.
        similarities: Specifies the list of similarity types to use in the architecture.
        alpha: Specifies the ratio to use for applying the temporal and instance losses.
        temperature: Specifies the temperature of the model.
        run_dir: Specifies the directory path to store run outputs.
        task_type: Specifies the type of the task.
        irregular: Specifies whether to add missing data inputs.
    """

    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device="cuda",
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None,
        hierarchical=False,
        queue_size=128,
        max_seq_len=3000,
        similarities=[c.SIMILARITY.INSTANCE, c.SIMILARITY.TEMPORAL],
        alpha=0.5,
        temperature=0.07,
        run_dir=".",
        task_type=c.CLASSIFICATION.TASK,
        irregular=0,
    ):
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        self.temperature = temperature
        self._run_dir = run_dir
        self._task_type = task_type
        self._irregular = irregular
        self._net = AnchorTSEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
            queue_size=queue_size,
            max_seq_len=max_seq_len,
            similarities=self._get_similarity_types(similarities),
        ).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        self.criterion = self._get_criterion(hierarchical, similarities, alpha)
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        self.n_epochs = 0
        self.n_iters = 0

    def _get_similarity_types(self, similarities: List[str]) -> List[SimilarityType]:
        similarity_types = []
        for similarity in similarities:
            similarity_types.append(SimilarityType[similarity])

        return similarity_types

    def _create_similarity_func_dict(self, similarities: List[str]) -> Dict[str, Similarity]:
        similarity_funcs_dict = {}
        for similarity in similarities:
            if similarity not in [
                c.SIMILARITY.TEMPORAL,
                c.SIMILARITY.INSTANCE,
                c.SIMILARITY.SIMPLE,
            ]:
                raise KeyError(f"Got an invalid similarity name: {similarity}")

            similarity_funcs_dict[similarity] = SimilarityType[similarity].value()

        return similarity_funcs_dict

    def _get_criterion(self, is_hierarchical: bool, similarities: List[str], alpha: float):
        if is_hierarchical:
            if len(similarities) > 1:
                loss_func = TemporalInstanceHierarchicalKLD
            else:
                if c.SIMILARITY.INSTANCE in similarities:
                    loss_func = InstanceHierarchicalKLD
                if c.SIMILARITY.TEMPORAL in similarities:
                    loss_func = TemporalHierarchicalKLD
        else:
            loss_func = KLD

        return loss_func(
            similarities=self._create_similarity_func_dict(similarities),
            alpha=alpha,
        ).cuda()

    def fit(self, train_data, data_dict, n_epochs=None, n_iters=None):
        """Trains the model.

        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of
                (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            data_dict (dict): Specifies the data dictionary from the loader.
                It is used in the evaluation.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training
                stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the
                training stops. If both n_epochs and n_iters are not specified, a default
                setting would be used that sets n_iters to 200 for a dataset with
                size <= 100000, 600 otherwise.
        """
        assert train_data.ndim == 3

        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,
            drop_last=True,
        )
        params = [p for p in self._net.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        loss_log = []
        pbar = tqdm(total=n_epochs if n_epochs is not None else n_iters)
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break

            cum_loss = 0
            n_epoch_iters = 0
            interrupted = False
            for batch in train_loader:

                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)

                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(
                    low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0)
                )

                optimizer.zero_grad()
                x_stu = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
                x_tch = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
                stu_feat, tch_feat = self._net(x_stu, x_tch, crop_length=crop_l)
                loss = self.criterion(
                    stu_feat, tch_feat, queue=self._net.queue, temperature=self.temperature
                )
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)

                cum_loss += loss.item()
                n_epoch_iters += 1
                self.n_iters += 1

                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

                if n_epochs is None:
                    pbar.update(1)

            if interrupted:
                break

            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            self.n_epochs += 1
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

            if n_epochs is not None:
                pbar.update(1)

        pbar.close()

        return loss_log

    def eval(self, train_data, data_dict):
        if self._task_type == c.CLASSIFICATION.TASK:
            if self._irregular > 0:
                data_dict[c.CLASSIFICATION.TEST_DATA] = data_dropout(
                    data_dict[c.CLASSIFICATION.TEST_DATA], self._irregular
                )
            train_repr, test_repr = self._get_classification_encoded_representation(
                train_data, data_dict
            )
            out, eval_res = tasks.eval_classification(
                train_repr,
                test_repr,
                data_dict[c.CLASSIFICATION.TRAIN_LABEL],
                data_dict[c.CLASSIFICATION.TEST_LABEL],
                eval_protocol="svm",
            )
        elif self._task_type == c.FORECASTING.TASK:
            data_repr = self.encode(
                data_dict[c.FORECASTING.DATA],
                casual=True,
                sliding_length=1,
                sliding_padding=200,
                batch_size=256,
            )
            out, eval_res = tasks.eval_forecasting(
                data_repr,
                data_dict[c.FORECASTING.DATA],
                data_dict[c.FORECASTING.TRAIN_SLICE],
                data_dict[c.FORECASTING.VALID_SLICE],
                data_dict[c.FORECASTING.TEST_SLICE],
                data_dict[c.FORECASTING.SCALAR],
                data_dict[c.FORECASTING.PRED_LENS],
                data_dict[c.FORECASTING.NUM_COV_COL],
            )
        elif self._task_type in [c.ANOMALY_DETECTION.TASK, c.ANOMALY_DETECTION.COLD_START_TASK]:
            all_data, all_repr, all_repr_wom = self._get_anomaly_encoded_representation(data_dict)

            if self._task_type == c.ANOMALY_DETECTION.TASK:
                out, eval_res = tasks.eval_anomaly_detection(
                    all_repr,
                    all_repr_wom,
                    data_dict[c.ANOMALY_DETECTION.TRAIN_DATA],
                    data_dict[c.ANOMALY_DETECTION.TEST_LABEL],
                    data_dict[c.ANOMALY_DETECTION.TEST_TIMESTAMP],
                    data_dict[c.ANOMALY_DETECTION.DELAY],
                )
            if self._task_type == c.ANOMALY_DETECTION.COLD_START_TASK:
                out, eval_res = tasks.eval_anomaly_detection_coldstart(
                    all_data,
                    all_repr,
                    all_repr_wom,
                    data_dict[c.ANOMALY_DETECTION.TRAIN_LABEL],
                    data_dict[c.ANOMALY_DETECTION.TRAIN_TIMESTAMP],
                    data_dict[c.ANOMALY_DETECTION.TEST_LABEL],
                    data_dict[c.ANOMALY_DETECTION.TEST_TIMESTAMP],
                    data_dict[c.ANOMALY_DETECTION.DELAY],
                )
        else:
            raise ValueError(f"Given task type, {self._task_type}, is not supported!")

        pkl_save(f"{self._run_dir}/out.pkl", out)
        pkl_save(f"{self._run_dir}/eval_res.pkl", eval_res)

        return eval_res

    def _get_classification_encoded_representation(self, train_data, data_dict):
        train_labels = data_dict[c.CLASSIFICATION.TRAIN_LABEL]
        test_data = data_dict[c.CLASSIFICATION.TEST_DATA]

        train_representation = self.encode(
            train_data, encoding_window="full_series" if train_labels.ndim == 1 else None
        )
        test_representation = self.encode(
            test_data, encoding_window="full_series" if train_labels.ndim == 1 else None
        )
        return train_representation, test_representation

    def _get_anomaly_encoded_representation(self, data_dict):
        all_data = {}
        all_repr, all_repr_wom = {}, {}
        all_train_data = data_dict[c.ANOMALY_DETECTION.TRAIN_DATA]
        all_test_data = data_dict[c.ANOMALY_DETECTION.TEST_DATA]

        pbar = tqdm(total=len(all_train_data))
        for k in all_train_data:
            all_data[k] = np.concatenate([all_train_data[k], all_test_data[k]])

            all_repr[k] = self.encode(
                all_data[k].reshape(1, -1, 1),
                mask="mask_last",
                casual=True,
                sliding_length=1,
                sliding_padding=200,
                batch_size=256,
            ).squeeze()

            all_repr_wom[k] = self.encode(
                all_data[k].reshape(1, -1, 1),
                casual=True,
                sliding_length=1,
                sliding_padding=200,
                batch_size=256,
            ).squeeze()

            pbar.update(1)

        pbar.close()

        return all_data, all_repr, all_repr_wom

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask=mask)
        if encoding_window == "full_series":
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=out.size(1),
            ).transpose(1, 2)

        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2,
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]

        elif encoding_window == "multiscale":
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2), kernel_size=(1 << (p + 1)) + 1, stride=1, padding=1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)

        else:
            if slicing is not None:
                out = out[:, slicing]

        return out.cpu()

    def encode(
        self,
        data,
        mask=None,
        encoding_window=None,
        casual=False,
        sliding_length=None,
        sliding_padding=0,
        batch_size=None,
    ):
        """Compute representations using the model.

        Args:
            data (numpy.ndarray): This should be of shape (n_instance, n_timestamps, n_features).
                All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can
                be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed
                representation would be the max pooling over this window. This can be set to
                'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not
                be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this
                param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for
                inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not
                specified, this would be the same batch size as training.

        Returns:
            repr: The representations for data.
        """
        assert self.net is not None, "please train or load a net first"
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()

        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l < 0 else 0,
                            right=r - ts_l if r > ts_l else 0,
                            dim=1,
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(
                                        sliding_padding, sliding_padding + sliding_length
                                    ),
                                    encoding_window=encoding_window,
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window,
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window,
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0

                    out = torch.cat(reprs, dim=1)
                    if encoding_window == "full_series":
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size=out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == "full_series":
                        out = out.squeeze(1)

                output.append(out)
            output = torch.cat(output, dim=0)

        self.net.train(org_training)
        return output.numpy()

    def save(self, fn):
        """Save the model to a file.

        Args:
            fn (str): filename.
        """
        torch.save(self.net.state_dict(), fn)

    def load(self, fn):
        """Load the model from a file.

        Args:
            fn (str): filename.
        """
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
