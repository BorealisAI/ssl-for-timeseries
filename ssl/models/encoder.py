# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch import nn

from ssl.models.dilated_conv import DilatedConvEncoder, generate_binomial_mask
from ssl.models.similarity import SimilarityType
from ssl.models.utils import get_shuffle_ids


class AnchorTSEncoder(nn.Module):
    """This class implements the main model proposed in the paper:
        - Self-Supervised Time Series Representation Learning with Temporal-Instance
            Similarity Distillation

    It is based on a teacher student encoder architecture and uses similarity distillation.

    Args:
        input_dims: Specifies the input dimension size.
        output_dims: Specifies the output dimension size.
        hidden_dims: Specifies the hidden diemnsion size.
        depth: Specifies the depth of the convolutional encoder.
        mask_mode: Specifies the type of masking used in the encoder for masking inputs.
        queue_size: Specifies the size/length of the memory queue.
        momentum: Specifies the momentum to use to update the teacher network.
        temperature: Specifies the temperature parameter to use in the model.
        dropout: Specifies the dropout parameter used in the encoder.
        max_seq_len: Specifies the maximum sequence length for a given dataset.
        similarities: Specifies the types of similarity functions to use.
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_dims=64,
        depth=10,
        mask_mode="binomial",
        queue_size=128,
        momentum=0.999,
        temperature=0.07,
        dropout=0.1,
        max_seq_len=3000,
        similarities=[SimilarityType.INSTANCE, SimilarityType.TEMPORAL],
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.depth = depth
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.similarities = similarities

        self.student_encoder = self._get_encoder()
        self.teacher_encoder = self._get_encoder()
        self.student_predictor = nn.Sequential(
            nn.Linear(output_dims, output_dims, bias=False),
            nn.BatchNorm1d(output_dims),
            nn.ReLU(inplace=True),
            nn.Linear(output_dims, output_dims, bias=True),
        )
        self._update_teacher_network()
        self._register_queue()
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.int))

    def _register_queue(self):
        if SimilarityType.SIMPLE in self.similarities:
            self.register_buffer("queue", torch.randn(self.queue_size, self.output_dims))
        else:
            self.register_buffer(
                "queue", torch.randn(self.queue_size, self.max_seq_len, self.output_dims)
            )

    def _get_encoder(self):
        return nn.Sequential(
            nn.Linear(self.input_dims, self.hidden_dims),
            DilatedConvEncoder(
                self.hidden_dims,
                [self.hidden_dims] * self.depth + [self.output_dims],
                kernel_size=3,
            ),
            nn.Dropout(p=self.dropout),
        )

    def _update_teacher_network(self):
        for p_stu, p_tch in zip(
            self.student_encoder.parameters(), self.teacher_encoder.parameters()
        ):
            p_tch.data.copy_(p_stu.data)
            p_tch.requires_grad = False

    @torch.no_grad()
    def _momentum_update_teacher_encoder(self):
        for p_stu, p_tch in zip(
            self.student_encoder.parameters(), self.teacher_encoder.parameters()
        ):
            p_tch.data = self.momentum * p_tch.data + (1.0 - self.momentum) * p_stu.data

    def _fill_queue_and_update_ptr(self, index, anchors):
        ptr = int(self.queue_ptr)

        if ptr + index >= self.queue_size:
            self.queue[ptr:] = anchors[: (self.queue_size - ptr)]
            ptr = 0
        else:
            self.queue[ptr : ptr + index] = anchors
            ptr += index

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue(self, anchors):
        if SimilarityType.SIMPLE in self.similarities:
            batch_size, seq_len, num_dim = anchors.shape
            anchors = anchors.contiguous().view(-1, num_dim)
            index = batch_size * seq_len
        else:
            index = anchors.shape[0]

        self._fill_queue_and_update_ptr(index, anchors)

    def _get_feature_and_apply_mask(self, x, encoder, mask=None):
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        # Get features through the first linear layer
        x = encoder[0](x)

        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = "all_true"

        if mask == "binomial":
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "all_true":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == "mask_last":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        return x

    def _get_encoded_representation(self, x, encoder):
        x = x.transpose(1, 2)
        x = encoder[1:](x).transpose(1, 2)
        return x

    def _get_student_prediction(self, x):
        x = self.student_predictor[0](x)
        x = x.transpose(1, 2)
        x = self.student_predictor[1](x).transpose(1, 2)
        x = self.student_predictor[2:](x)

        return x

    def _pad_features_to_max_len(self, feature, padding_length, padding_value=0):
        padded_feature = nn.functional.pad(
            feature, (0, 0, 0, padding_length), mode="constant", value=padding_value
        )
        return padded_feature

    def forward(self, stu_view_x, tch_view_x=None, crop_length=None, mask=None):
        if self.training:
            # Get the overlap feature & normalize over feature dimension i.e. N in (B x L x N)
            stu_feat = self._get_encoded_representation(
                self._get_feature_and_apply_mask(stu_view_x, self.student_encoder, mask),
                self.student_encoder,
            )
            stu_feat = stu_feat[:, -crop_length:]
            stu_feat = nn.functional.normalize(self._get_student_prediction(stu_feat), dim=2)

            # Get representation for the teacher
            with torch.no_grad():
                self._momentum_update_teacher_encoder()

                shuffle_ids, reverse_ids = get_shuffle_ids(tch_view_x.shape[0])
                tch_view_x = tch_view_x[shuffle_ids]

                # forward through the encoder and get the overlap feature
                tch_feat = self._get_encoded_representation(
                    self._get_feature_and_apply_mask(tch_view_x, self.teacher_encoder, mask),
                    self.teacher_encoder,
                )
                tch_feat = tch_feat[:, :crop_length]
                tch_feat = nn.functional.normalize(tch_feat, dim=2)

                # Undo shuffle
                tch_feat = tch_feat[reverse_ids]

            seq_len = stu_feat.shape[1]
            if SimilarityType.SIMPLE not in self.similarities:
                stu_feat = self._pad_features_to_max_len(
                    stu_feat, padding_length=self.max_seq_len - seq_len
                )
                tch_feat = self._pad_features_to_max_len(
                    tch_feat, padding_length=self.max_seq_len - seq_len
                )
            self._dequeue_and_enqueue(tch_feat)

            return stu_feat, tch_feat

        else:
            stu_feat = self._get_encoded_representation(
                self._get_feature_and_apply_mask(stu_view_x, self.student_encoder, mask),
                self.student_encoder,
            )
            return stu_feat
