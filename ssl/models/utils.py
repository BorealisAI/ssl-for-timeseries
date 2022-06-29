# MIT License
#
# Copyright (c) 2020-present UMBC Vision
#
import torch


def get_shuffle_ids(batch_size):
    forward_inds = torch.randperm(batch_size).long().cuda()
    backward_inds = torch.zeros(batch_size).long().cuda()
    value = torch.arange(batch_size).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds
