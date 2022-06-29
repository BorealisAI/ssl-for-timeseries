# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Optional, Union, Tuple, Dict

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

import ssl.constants as c
from ssl.models.similarity import Similarity


class KLD(Module):
    def __init__(self, similarities: Dict[str, Similarity], alpha: float):
        Module.__init__(self)
        if alpha <= 0:
            raise ValueError(f"Alpha should be a positive number! Got {alpha} instead.")

        self.similarities = similarities
        self.coefficient = {c.SIMILARITY.TEMPORAL: 1 - alpha, c.SIMILARITY.INSTANCE: alpha}

    def _get_kl_divergence(
        self, student_feature: Optional[Tensor], teacher_feature: Optional[Tensor]
    ) -> Union[Tensor, float]:
        student_feature = F.log_softmax(student_feature, dim=1)
        teacher_feature = F.softmax(teacher_feature, dim=1)
        return F.kl_div(student_feature, teacher_feature, reduction="batchmean")

    def _get_similarity_loss(
        self,
        student_feature: Tensor,
        teacher_feature: Tensor,
        temperature: float,
        queue: Optional[Tensor],
    ) -> Tensor:
        loss = 0
        for name, similarity in self.similarities.items():
            student_similarity, teacher_similarity = similarity.get_similarity(
                student_feature, teacher_feature, temperature, queue
            )
            loss += self.coefficient[name] * self._get_kl_divergence(
                student_similarity, teacher_similarity
            )

        return loss

    def forward(
        self,
        student_feature: Tensor,
        teacher_feature: Tensor,
        temperature: float,
        queue: Optional[Tensor],
    ) -> Tensor:
        return self._get_similarity_loss(student_feature, teacher_feature, temperature, queue)


class HierarchicalKLD(KLD):
    def _apply_max_pool(self, feature: Tensor) -> Tensor:
        feature = feature.transpose(1, 2)
        pooled_feature = F.max_pool1d(feature, kernel_size=2)

        return pooled_feature.transpose(1, 2)

    def get_hierarchical_similarity_loss(
        self,
        student_feature: Tensor,
        teacher_feature: Tensor,
        temperature: float,
        queue: Optional[Tensor],
    ) -> Tuple[Tensor, int]:
        depth, loss = 0, 0.0

        while student_feature.size(1) > 1:
            loss += self._get_similarity_loss(student_feature, teacher_feature, temperature, queue)
            student_feature = self._apply_max_pool(student_feature)
            teacher_feature = self._apply_max_pool(teacher_feature)
            if queue is not None:
                queue = self._apply_max_pool(queue)
            depth += 1

        return loss, depth


class InstanceHierarchicalKLD(HierarchicalKLD):
    def forward(
        self,
        student_feature: Tensor,
        teacher_feature: Tensor,
        temperature: float,
        queue: Optional[Tensor],
    ) -> Tensor:
        loss, depth = self.get_hierarchical_similarity_loss(
            student_feature, teacher_feature, temperature, queue
        )

        loss += self._get_similarity_loss(student_feature, teacher_feature, temperature, queue)
        loss /= depth + 1

        return loss


class TemporalHierarchicalKLD(HierarchicalKLD):
    def forward(
        self,
        student_feature: Tensor,
        teacher_feature: Tensor,
        temperature: float,
        queue: Optional[Tensor],
    ) -> Tensor:
        loss, depth = self.get_hierarchical_similarity_loss(
            student_feature, teacher_feature, temperature, queue
        )

        return loss


class TemporalInstanceHierarchicalKLD(HierarchicalKLD):
    def _get_last_iteration_loss(
        self,
        student_feature: Tensor,
        teacher_feature: Tensor,
        temperature: float,
        queue: Optional[Tensor],
    ) -> Tensor:
        similarity = self.similarities[c.SIMILARITY.INSTANCE]
        student_similarity, teacher_similarity = similarity.get_similarity(
            student_feature, teacher_feature, temperature, queue
        )
        loss = self.coefficient[c.SIMILARITY.INSTANCE] * self._get_kl_divergence(
            student_similarity, teacher_similarity
        )
        return loss

    def forward(
        self,
        student_feature: Tensor,
        teacher_feature: Tensor,
        temperature: float,
        queue: Optional[Tensor],
    ) -> Tensor:
        loss, depth = self.get_hierarchical_similarity_loss(
            student_feature, teacher_feature, temperature, queue
        )

        loss += self._get_last_iteration_loss(student_feature, teacher_feature, temperature, queue)
        loss /= depth + 1

        return loss
