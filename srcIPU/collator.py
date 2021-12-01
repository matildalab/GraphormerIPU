# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np

from random import randint

def pad_1d_unsqueeze(x, padlen):
    x = x + 1 # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1 # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_rel_pos_3d_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


class Batch():
    def __init__(self, idx, attn_bias, attn_edge_type, rel_pos, all_rel_pos_3d_1, in_degree, out_degree, x, edge_input, y):
        super(Batch, self).__init__()
        self.idx = idx
        self.in_degree, self.out_degree = in_degree, out_degree
        self.x, self.y = x, y
        self.attn_bias, self.attn_edge_type, self.rel_pos = attn_bias, attn_edge_type, rel_pos
        self.edge_input = edge_input
        self.all_rel_pos_3d_1 = all_rel_pos_3d_1

    def to(self, device):
        self.idx = self.idx.to(device)
        self.in_degree, self.out_degree = self.in_degree.to(device), self.out_degree.to(device)
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.attn_bias, self.attn_edge_type, self.rel_pos = self.attn_bias.to(device), self.attn_edge_type.to(device), self.rel_pos.to(device)
        self.edge_input = self.edge_input.to(device)
        self.all_rel_pos_3d_1 = self.all_rel_pos_3d_1.to(device)
        return self

    def __len__(self):
        return self.in_degree.size(0)


def collator(items, max_node=512, multi_hop_max_dist=20, rel_pos_max = 1024):
    batch_size = len(items)
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    
    #IF THERE ARE REMOVED NODES
    missing_items = batch_size - len(items)
    for i in range(missing_items):
        random_idx = randint(0, batch_size - missing_items - 1)
        items.append(items[random_idx])
    # IF THERE ARE REMOVED NODES

    # for i in range(len(items)):
    #     if items[i] is None or items[i].x.size(0) > max_node:
    #         items.remove(items[i])
    #         items.append(items[randint(0, i-2)])            

    items_ = [(item.idx, item.attn_bias, item.attn_edge_type, item.rel_pos, item.in_degree, item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y) for item in items]
    idxs, attn_biases, attn_edge_types, rel_poses, in_degrees, out_degrees, xs, edge_inputs, ys = zip(*items_)

    items_ = [(item.all_rel_pos_3d_1,) for item in items]
    all_rel_pos_3d_1s, = zip(*items_)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][rel_poses[idx] >= rel_pos_max] = float('-inf')
    # max_node_num = max(i.size(0) for i in xs)
    # max_dist = max(i.size(-2) for i in edge_inputs)
    max_node_num = 30
    max_dist = 20
    
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs]) #RuntimeError: torch.cat(): Sizes of tensors must match except in dimension 0. Got 30 and 31 in dimension 1 


    # edge_input = torch.cat([pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]).int()
    # attn_bias = torch.cat([pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]).half()
    # attn_edge_type = torch.cat([pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]).int()
    # rel_pos = torch.cat([pad_rel_pos_unsqueeze(i, max_node_num) for i in rel_poses]).int()
    # all_rel_pos_3d_1 = torch.cat([pad_rel_pos_3d_unsqueeze(i, max_node_num) for i in all_rel_pos_3d_1s]).half()
    # in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees]).int()
    # out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in out_degrees]).int()
    # returningTuple = (torch.IntTensor(idxs), attn_bias, attn_edge_type, rel_pos, all_rel_pos_3d_1, in_degree, out_degree, x, edge_input, y)

    edge_input = torch.cat([pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]).int()
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat([pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]).int()
    rel_pos = torch.cat([pad_rel_pos_unsqueeze(i, max_node_num) for i in rel_poses]).int()
    all_rel_pos_3d_1 = torch.cat([pad_rel_pos_3d_unsqueeze(i, max_node_num) for i in all_rel_pos_3d_1s])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees]).int()
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in out_degrees]).int()
    returningTuple = (torch.IntTensor(idxs), attn_bias, attn_edge_type, rel_pos, all_rel_pos_3d_1, in_degree, out_degree, x, edge_input, y)

    return returningTuple

    # attn_bias, rel_pos, x = batched_data[1].half(), batched_data[3].int(), batched_data[7].int()
    # in_degree, out_degree = batched_data[5].int(), batched_data[5].int()
    # edge_input, attn_edge_type = batched_data[8].int(), batched_data[2].int()
    # all_rel_pos_3d_1 = batched_data[4].half()