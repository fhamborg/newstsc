# -*- coding: utf-8 -*-
# file: SPC_BERT.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch.nn as nn


class SPC_ROBERTA(nn.Module):
    def __init__(self, roberta, opt):
        super(SPC_ROBERTA, self).__init__()
        self.roberta = roberta
        self.dropout = nn.Dropout(opt.dropout)
        self.reduction = opt.spc_reduction
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_with_special_indexes = inputs[0]

        last_hidden_state, pooler_output = self.roberta(text_with_special_indexes)

        # according to https://huggingface.co/transformers/model_doc/bert.html#bertmodel we should not use (as in
        # the original SPC version) the pooler_output but get the last_hidden_state and "averaging or pooling the
        # sequence of hidden-states for the whole input sequence"
        if self.reduction == 'pooler_output':
            prepared_output = pooler_output
        elif self.reduction == 'mean_last_hidden_states':
            mean_last_hidden_state = last_hidden_state.mean(dim=1, keepdim=True)
            prepared_output = mean_last_hidden_state[:, 0, :]

        prepared_output = self.dropout(prepared_output)
        logits = self.dense(prepared_output)

        return logits

class SPC_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(SPC_BERT, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.reduction = opt.spc_reduction
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]

        last_hidden_state, pooler_output = self.bert(text_bert_indices, bert_segments_ids)

        # according to https://huggingface.co/transformers/model_doc/bert.html#bertmodel we should not use (as in
        # the original SPC version) the pooler_output but get the last_hidden_state and "averaging or pooling the
        # sequence of hidden-states for the whole input sequence"
        if self.reduction == 'pooler_output':
            prepared_output = pooler_output
        elif self.reduction == 'mean_last_hidden_states':
            mean_last_hidden_state = last_hidden_state.mean(dim=1, keepdim=True)
            prepared_output = mean_last_hidden_state[:, 0, :]

        prepared_output = self.dropout(prepared_output)
        logits = self.dense(prepared_output)

        return logits


class SPC_DISTILBERT(nn.Module):
    def __init__(self, distilbert, opt):
        super(SPC_DISTILBERT, self).__init__()
        self.distilbert = distilbert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices = inputs[0]

        last_hidden_state = self.distilbert(text_bert_indices)[0]

        mean_last_hidden_state = last_hidden_state.mean(dim=1, keepdim=True)
        prepared_output = mean_last_hidden_state[:, 0, :]

        prepared_output = self.dropout(prepared_output)
        logits = self.dense(prepared_output)

        return logits
