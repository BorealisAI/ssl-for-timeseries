# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Optional

import torch
from torch import Tensor


class Similarity(ABC):
    """Abstract class for computing similarity features."""

    @abstractmethod
    def get_similarity(
        self,
        student_feature: Tensor,
        teacher_feature: Tensor,
        temperature: float,
        queue: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        pass


class SimpleSimilarity(Similarity):
    def get_similarity(
        self,
        student_feature: Tensor,
        teacher_feature: Tensor,
        temperature: float,
        queue: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:

        batch_size, sequence_len, num_features = student_feature.shape
        student_feature = student_feature.contiguous().view(-1, num_features)
        teacher_feature = teacher_feature.contiguous().view(-1, num_features)

        queue = queue.clone().detach()
        student_similarity = torch.mm(student_feature, queue.t())
        teacher_similarity = torch.mm(teacher_feature, queue.t())

        student_similarity /= temperature
        teacher_similarity /= temperature

        return student_similarity, teacher_similarity


class InstanceSimilarity(Similarity):
    def get_similarity(
        self,
        student_feature: Tensor,
        teacher_feature: Tensor,
        temperature: float,
        queue: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:

        batch_size, sequence_len, num_features = student_feature.shape
        student_feature = student_feature.permute(1, 0, 2)
        teacher_feature = teacher_feature.permute(1, 0, 2)

        queue = queue.clone().detach()
        queue = queue.permute(1, 2, 0)
        student_similarity = torch.matmul(student_feature, queue)
        teacher_similarity = torch.matmul(teacher_feature, queue)

        student_similarity /= temperature
        teacher_similarity /= temperature

        return student_similarity, teacher_similarity


class TemporalSimilarity(Similarity):
    def get_similarity(
        self,
        student_feature: Tensor,
        teacher_feature: Tensor,
        temperature: float,
        queue: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        student_similarity = torch.matmul(student_feature, student_feature.transpose(1, 2))
        teacher_similarity = torch.matmul(teacher_feature, teacher_feature.transpose(1, 2))

        student_similarity /= temperature
        teacher_similarity /= temperature
        return student_similarity, teacher_similarity


class SimilarityType(Enum):
    SIMPLE = SimpleSimilarity
    INSTANCE = InstanceSimilarity
    TEMPORAL = TemporalSimilarity
