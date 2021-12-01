# from lr import PolynomialDecayLR
# import torch
# import math
# import torch.nn as nn
# from data import get_dataset

# from utils.flag import flag, flag_bounded

# import numpy as np
# import torch.nn.functional as F


# def softplus_inverse(x):
#     return x + np.log(-np.expm1(-x))

# class RBFLayer(nn.Module):
#     def __init__(self, K=64, cutoff=10, dtype=torch.float):
#         super().__init__()
#         self.cutoff = cutoff

#         centers = torch.tensor(softplus_inverse(np.linspace(1.0, np.exp(-cutoff), K)), dtype=dtype)
#         self.centers = nn.Parameter(F.softplus(centers))

#         widths = torch.tensor([softplus_inverse(0.5 / ((1.0 - np.exp(-cutoff) / K)) ** 2)] * K, dtype=dtype)
#         self.widths = nn.Parameter(F.softplus(widths))

#     def cutoff_fn(self, D):
#         x = D / self.cutoff
#         x3, x4, x5 = torch.pow(x, 3.0), torch.pow(x, 4.0), torch.pow(x, 5.0)
#         return torch.where(x < 1, 1-6*x5+15*x4-10*x3, torch.zeros_like(x))

#     def forward(self, D):
#         D = D.unsqueeze(-1)
#         return self.cutoff_fn(D) * torch.exp(-self.widths*torch.pow((torch.exp(-D) - self.centers), 2))

# class GraphFormer(nn.Module):
#     def __init__(
#             self,
#             n_layers,
#             head_size,
#             hidden_dim,
#             dropout_rate,
#             intput_dropout_rate,
#             ffn_dim,
#             dataset_name,
#             edge_type,
#             multi_hop_max_dist,
#             attention_dropout_rate,
#             flag=False,
#         ):
#         super().__init__()
#         # self.save_hyperparameters()

#         self.head_size = head_size
            
#         self.atom_encoder = nn.Embedding(128 * 37 + 1, hidden_dim, padding_idx=0)
#         self.edge_encoder = nn.Embedding(128 * 6 + 1, head_size, padding_idx=0)
#         self.edge_type = edge_type

#         self.edge_dis_encoder = nn.Embedding(128 * head_size * head_size,1)

#         self.rel_pos_encoder = nn.Embedding(512, head_size, padding_idx=0)
#         self.in_degree_encoder = nn.Embedding(512, hidden_dim, padding_idx=0)
#         self.out_degree_encoder = nn.Embedding(512, hidden_dim, padding_idx=0)

#         self.input_dropout = nn.Dropout(intput_dropout_rate)
#         encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, head_size)
#                     for _ in range(n_layers)]
#         self.layers = nn.ModuleList(encoders)
#         self.final_ln = nn.LayerNorm(hidden_dim)
#         self.out_proj = nn.Linear(hidden_dim, 1)
        
#         self.graph_token = nn.Embedding(1, hidden_dim)
#         self.graph_token_virtual_distance = nn.Embedding(1, head_size)

#         self.loss_fn = get_dataset(dataset_name)['loss_fn']
#         self.dataset_name = dataset_name
#         self.multi_hop_max_dist = multi_hop_max_dist

#         self.flag = flag
#         self.hidden_dim = hidden_dim

#         K = 256
#         cutoff = 10
#         self.rbf = RBFLayer(K, cutoff)
#         self.rel_pos_3d_proj = nn.Linear(K, head_size)

#     def forward(self, batched_data, perturb=None):

#         attn_bias, rel_pos, x = batched_data[1], batched_data[3], batched_data[7]
#         in_degree, out_degree = batched_data[5], batched_data[5]
#         edge_input, attn_edge_type = batched_data[8], batched_data[2]
#         all_rel_pos_3d_1 = batched_data[4]

#         n_graph, n_node = x.size()[:2]
#         graph_attn_bias = attn_bias.clone()
#         graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1) # [n_graph, n_head, n_node+1, n_node+1]

#         rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2) # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
#         graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + rel_pos_bias

#         rbf_result = self.rel_pos_3d_proj(self.rbf(all_rel_pos_3d_1)).permute(0, 3, 1, 2)
#         graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + rbf_result

#         t = self.graph_token_virtual_distance.weight.view(1, self.head_size, 1)
#         graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
#         graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

#         rel_pos_ = rel_pos.clone()
#         rel_pos_ = torch.masked_fill(rel_pos_, rel_pos_ == 0, 1)
#         rel_pos_ = torch.where(rel_pos_ > 1, rel_pos_ - 1, rel_pos_) # set 1 to 1, x > 1 to x - 1
#         if self.multi_hop_max_dist > 0:
#             rel_pos_ = rel_pos_.clamp(0, self.multi_hop_max_dist)
#             edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
#         edge_input = self.edge_encoder(edge_input).mean(-2) #[n_graph, n_node, n_node, max_dist, n_head]
#         max_dist = edge_input.size(-2)
#         edge_input_flat = edge_input.permute(3,0,1,2,4).reshape(max_dist, -1, self.head_size)
#         edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(-1, self.head_size, self.head_size)[:max_dist, :, :])
#         edge_input = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.head_size).permute(1,2,3,0,4)
#         edge_input = (edge_input.sum(-2) / (rel_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)

#         graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
#         graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1) # reset

#         node_feature = self.atom_encoder(x).mean(dim=-2)           # [n_graph, n_node, n_hidden]
#         if self.flag and perturb is not None:
#             node_feature += perturb

#         node_feature = node_feature + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
#         graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
#         graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

#         output = self.input_dropout(graph_node_feature)        
#         for enc_layer in self.layers:
#             output = enc_layer(output, graph_attn_bias)
#         output = self.final_ln(output)
        
#         output = self.out_proj(output[:, 0, :])                        # get whole graph rep

#         y_hat = output.view(-1)
#         y_gt = batched_data[9].view(-1)
#         loss = self.loss_fn(y_hat,y_gt)

#         return output, loss

# class FeedForwardNetwork(nn.Module):
#     def __init__(self, hidden_size, ffn_size, dropout_rate):
#         super(FeedForwardNetwork, self).__init__()

#         self.layer1 = nn.Linear(hidden_size, ffn_size)
#         self.gelu = nn.GELU()
#         self.layer2 = nn.Linear(ffn_size, hidden_size)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.gelu(x)
#         x = self.layer2(x)
#         return x


# class MultiHeadAttention(nn.Module):
#     def __init__(self, hidden_size, attention_dropout_rate, head_size):
#         super(MultiHeadAttention, self).__init__()

#         self.head_size = head_size

#         self.att_size = att_size = hidden_size // head_size
#         self.scale = att_size ** -0.5

#         self.linear_q = nn.Linear(hidden_size, head_size * att_size)
#         self.linear_k = nn.Linear(hidden_size, head_size * att_size)
#         self.linear_v = nn.Linear(hidden_size, head_size * att_size)
#         self.att_dropout = nn.Dropout(attention_dropout_rate)

#         self.output_layer = nn.Linear(head_size * att_size, hidden_size)

#     def forward(self, q, k, v, attn_bias=None):
#         orig_q_size = q.size()

#         d_k = self.att_size
#         d_v = self.att_size
#         batch_size = q.size(0)

#         # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
#         q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
#         k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
#         v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

#         q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
#         v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
#         k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

#         # Scaled Dot-Product Attention.
#         # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
#         q = q * self.scale
#         x = torch.matmul(q, k)  # [b, h, q_len, k_len]
#         if attn_bias is not None:
#             x = x + attn_bias

#         x = torch.softmax(x, dim=3)
#         x = self.att_dropout(x)
#         x = x.matmul(v)  # [b, h, q_len, attn]

#         x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
#         x = x.view(batch_size, -1, self.head_size * d_v)

#         x = self.output_layer(x)

#         assert x.size() == orig_q_size
#         return x

# class EncoderLayer(nn.Module):
#     def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
#         super(EncoderLayer, self).__init__()

#         self.self_attention_norm = nn.LayerNorm(hidden_size)
#         self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
#         self.self_attention_dropout = nn.Dropout(dropout_rate)

#         self.ffn_norm = nn.LayerNorm(hidden_size)
#         self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
#         self.ffn_dropout = nn.Dropout(dropout_rate)

#     def forward(self, x, attn_bias=None):
#         y = self.self_attention_norm(x)
#         y = self.self_attention(y, y, y, attn_bias)
#         y = self.self_attention_dropout(y)
#         x = x + y

#         y = self.ffn_norm(x)
#         y = self.ffn(y)
#         y = self.ffn_dropout(y)
#         x = x + y
#         return x

from lr import PolynomialDecayLR
import torch
import math
import torch.nn as nn
from data import get_dataset

from utils.flag import flag, flag_bounded

import numpy as np
import torch.nn.functional as F

import poptorch


def softplus_inverse(x):
    return x + np.log(-np.expm1(-x))

class RBFLayer(nn.Module):
    def __init__(self, K=64, cutoff=10, dtype=torch.float):
        super().__init__()
        self.cutoff = cutoff

        centers = torch.tensor(softplus_inverse(np.linspace(1.0, np.exp(-cutoff), K)), dtype=dtype)
        self.centers = nn.Parameter(F.softplus(centers))

        widths = torch.tensor([softplus_inverse(0.5 / ((1.0 - np.exp(-cutoff) / K)) ** 2)] * K, dtype=dtype)
        self.widths = nn.Parameter(F.softplus(widths))

    def cutoff_fn(self, D):
        x = D / self.cutoff
        x3, x4, x5 = torch.pow(x, 3.0), torch.pow(x, 4.0), torch.pow(x, 5.0)
        return torch.where(x < 1, 1-6*x5+15*x4-10*x3, torch.zeros_like(x))

    def forward(self, D):
        D = D.unsqueeze(-1)
        return self.cutoff_fn(D) * torch.exp(-self.widths*torch.pow((torch.exp(-D) - self.centers), 2))

class GraphFormer(nn.Module):
    def __init__(
            self,
            n_layers,
            head_size,
            hidden_dim,
            dropout_rate,
            intput_dropout_rate,
            weight_decay,
            ffn_dim,
            dataset_name,
            warmup_updates,
            tot_updates,
            peak_lr,
            end_lr,
            edge_type,
            multi_hop_max_dist,
            attention_dropout_rate,
            flag=False,
            flag_m=3,
            flag_step_size=1e-3,
            flag_mag=1e-3,
        ):
        super().__init__()

        self.head_size = head_size
        
        if dataset_name == 'ZINC':
            self.atom_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
            self.edge_encoder = nn.Embedding(64, head_size, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == 'multi_hop':
                self.edge_dis_encoder = nn.Embedding(40 * head_size * head_size,1)
            self.rel_pos_encoder = nn.Embedding(40, head_size, padding_idx=0)
            self.in_degree_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        else:
            self.atom_encoder = nn.Embedding(128 * 37 + 1, hidden_dim, padding_idx=0)
            self.edge_encoder = nn.Embedding(128 * 6 + 1, head_size, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == 'multi_hop':
                self.edge_dis_encoder = nn.Embedding(128 * head_size * head_size,1)
            self.rel_pos_encoder = nn.Embedding(512, head_size, padding_idx=0)
            self.in_degree_encoder = nn.Embedding(512, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(512, hidden_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        if dataset_name == 'PCQM4M-LSC':
            self.out_proj = nn.Linear(hidden_dim, 1)
        else:
            self.downstream_out_proj = nn.Linear(hidden_dim, get_dataset(dataset_name)['num_class'])

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, head_size)

        self.evaluator = get_dataset(dataset_name)['evaluator']
        self.metric = get_dataset(dataset_name)['metric']
        self.loss_fn = get_dataset(dataset_name)['loss_fn']
        self.dataset_name = dataset_name
        
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.multi_hop_max_dist = multi_hop_max_dist

        self.flag = flag
        self.flag_m = flag_m
        self.flag_step_size = flag_step_size
        self.flag_mag = flag_mag
        self.hidden_dim = hidden_dim
        # self.automatic_optimization = not self.flag

        K = 256
        cutoff = 10
        self.rbf = RBFLayer(K, cutoff)
        self.rel_pos_3d_proj = nn.Linear(K, head_size)

    def forward(self, batched_data, perturb=None):
        # attn_bias, rel_pos, x = batched_data[1].half(), batched_data[3].int(), batched_data[7].int()
        # in_degree, out_degree = batched_data[5].int(), batched_data[5].int()
        # edge_input, attn_edge_type = batched_data[8].int(), batched_data[2].int()
        # all_rel_pos_3d_1 = batched_data[4].half()
        attn_bias, rel_pos, x = batched_data[1], batched_data[3], batched_data[7]
        in_degree, out_degree = batched_data[5], batched_data[5]
        edge_input, attn_edge_type = batched_data[8], batched_data[2]
        all_rel_pos_3d_1 = batched_data[4]


        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1) # [n_graph, n_head, n_node+1, n_node+1]

        # rel pos
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2) # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + rel_pos_bias

        # graph_attn_bias_returned = poptorch.ipu_print_tensor(graph_attn_bias)

        rbf_result = self.rel_pos_3d_proj(self.rbf(all_rel_pos_3d_1)).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + rbf_result

        # reset rel pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.head_size, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == 'multi_hop':
            rel_pos_ = rel_pos.clone()
            # rel_pos_[rel_pos_ == 0] = 1 # set pad to 1
            rel_pos_ = torch.masked_fill(rel_pos_, rel_pos_==0, 1)
            rel_pos_ = torch.where(rel_pos_ > 1, rel_pos_ - 1, rel_pos_) # set 1 to 1, x > 1 to x - 1
            if self.multi_hop_max_dist > 0:
                rel_pos_ = rel_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
            edge_input = self.edge_encoder(edge_input).mean(-2) #[n_graph, n_node, n_node, max_dist, n_head]
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3,0,1,2,4).reshape(max_dist, -1, self.head_size)
            edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(-1, self.head_size, self.head_size)[:max_dist, :, :])
            edge_input = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.head_size).permute(1,2,3,0,4)
            # rel_pos_returned = poptorch.ipu_print_tensor(rel_pos_)
            edge_input = (edge_input.sum(-2) / (rel_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
        else:
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2) # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]

        

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input

        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1) # reset

        # graph_attn_bias_returned = poptorch.ipu_print_tensor(graph_attn_bias)

        # node feauture + graph token
        node_feature = self.atom_encoder(x).mean(dim=-2)           # [n_graph, n_node, n_hidden]
        if self.flag and perturb is not None:
            node_feature += perturb

        node_feature = node_feature + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(graph_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, graph_attn_bias)
        output = self.final_ln(output)
        # output part
        if self.dataset_name == 'PCQM4M-LSC':
            output = self.out_proj(output[:, 0, :])                        # get whole graph rep
        else:
            output = self.downstream_out_proj(output[:, 0, :])

        y_hat = output.view(-1)
        y_gt = batched_data[9].view(-1)
        loss = self.loss_fn(y_hat,y_gt)

        return output, loss

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
