# -*- coding: utf-8 -*-
# file: aen.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn

from fxlogger import get_logger
from layers.attention import Attention
from layers.point_wise_feed_forward import PositionwiseFeedForward
from layers.squeeze_embedding import SqueezeEmbedding

logger = get_logger()


# CrossEntropyLoss for Label Smoothing Regularization
class CrossEntropyLoss_LSR(nn.Module):
    def __init__(self, device, para_LSR=0.2, weight=None):
        super(CrossEntropyLoss_LSR, self).__init__()
        self.para_LSR = para_LSR
        self.device = device
        self.logSoftmax = nn.LogSoftmax(dim=-1)
        self.weight = weight

    def _toOneHot_smooth(self, label, batchsize, classes):
        base_prob = self.para_LSR * 1.0 / classes
        one_hot_label = torch.zeros(batchsize, classes) + base_prob

        if self.weight is not None:
            one_hot_label = one_hot_label * self.weight

        for i in range(batchsize):
            index = label[i]
            one_hot_label[i, index] += (1.0 - self.para_LSR)

        return one_hot_label

    def forward(self, pre, label, size_average=True):
        b, c = pre.size()
        pre_logsoftmax = self.logSoftmax(pre)

        one_hot_label = self._toOneHot_smooth(label, b, c).to(self.device)

        loss = torch.sum(-one_hot_label * pre_logsoftmax, dim=1)

        if size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class AEN_Base(nn.Module):
    def __init__(self, language_model_or_embedding_matrix, opt):
        super(AEN_Base, self).__init__()
        logger.info("creating AEN_Base")
        self.device = opt.device
        self.name = opt.model_name
        self.lm_representation = opt.lm_representation

        if self.name in ['aen_bert', 'aen_roberta', 'aen_distilbert']:
            self.language_model = language_model_or_embedding_matrix
            embed_dim = opt.bert_dim
        elif self.name == 'aen_glove':
            self.embed = nn.Embedding.from_pretrained(
                torch.tensor(language_model_or_embedding_matrix, dtype=torch.float))
            embed_dim = self.opt.embed_dim
        else:
            raise Exception('unknown model name: {}'.format(self.name))

        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)

        self.attn_k = Attention(embed_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp',
                                dropout=opt.dropout)
        self.attn_q = Attention(embed_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp',
                                dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)

        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)

        self.dense = nn.Linear(opt.hidden_dim * 3, opt.polarities_dim)

    def apply_lm(self, input):
        if self.name in ['aen_bert', 'aen_roberta']:
            last_hidden, _, all_hidden = self.language_model(input)
        elif self.name == 'aen_distilbert':
            last_hidden, all_hidden = self.language_model(input)
        else:
            raise Exception("unknown model name")

        if self.lm_representation == 'last':
            return last_hidden
        elif self.lm_representation == 'sum_last_four':
            last_four = all_hidden[-4:]  # list of four, each has shape: 16, 80, 768
            last_four_stacked = torch.stack(last_four)  # shape: 4, 16, 80, 768
            sum_last_four = torch.sum(last_four_stacked, dim=0)
            return sum_last_four
        elif self.lm_representation == 'sum_last_two':
            last_two = all_hidden[-2:]
            last_two_stacked = torch.stack(last_two)
            sum_last_two = torch.sum(last_two_stacked, dim=0)
            return sum_last_two
        elif self.lm_representation == 'sum_all':
            all_stacked = torch.stack(all_hidden)
            sum_all = torch.sum(all_stacked, dim=0)
            return sum_all

    def forward(self, inputs):
        context, target = inputs[0], inputs[1]
        context_len = torch.sum(context != 0, dim=-1)
        target_len = torch.sum(target != 0, dim=-1)

        if hasattr(self, 'language_model'):
            context = self.squeeze_embedding(context, context_len)
            context = self.apply_lm(context)
            context = self.dropout(context)

            target = self.squeeze_embedding(target, target_len)
            target = self.apply_lm(target)
            target = self.dropout(target)
        else:
            context = self.embed(context)
            context = self.squeeze_embedding(context, context_len)

            target = self.embed(target)
            target = self.squeeze_embedding(target, target_len)

        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)

        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)

        s1, _ = self.attn_s1(hc, ht)

        context_len = torch.tensor(context_len, dtype=torch.float).to(self.device)
        target_len = torch.tensor(target_len, dtype=torch.float).to(self.device)

        hc_mean = torch.div(torch.sum(hc, dim=1), context_len.view(context_len.size(0), 1))
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len.view(target_len.size(0), 1))
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len.view(context_len.size(0), 1))

        x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)
        out = self.dense(x)

        return out
