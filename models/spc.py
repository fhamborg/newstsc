# -*- coding: utf-8 -*-
# file: SPC_BERT.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn


class SPC_Base(nn.Module):
    def __init__(self, lm, opt):
        super(SPC_Base, self).__init__()
        self.lm = lm
        self.dropout = nn.Dropout(opt.dropout)
        self.spc_lm_representation = opt.spc_lm_representation
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.name = opt.model_name

    def apply_lm(self, inputs):
        if self.name == 'aen_bert':
            last_hidden, pooler_output, all_hidden = self.language_model(inputs[0], inputs[1])
        elif self.name == 'aen_roberta':
            last_hidden, pooler_output, all_hidden = self.language_model(inputs[0])
        elif self.name == 'aen_distilbert':
            last_hidden, all_hidden = self.language_model(inputs[0])
            pooler_output = None
        else:
            raise Exception("unknown model name")

        return last_hidden, pooler_output, all_hidden

    def forward(self, inputs):
        last_hidden_state, pooler_output, all_hidden_states = self.apply_lm(inputs)

        # according to https://huggingface.co/transformers/model_doc/bert.html#bertmodel we should not use (as in
        # the original SPC version) the pooler_output but get the last_hidden_state and "averaging or pooling the
        # sequence of hidden-states for the whole input sequence"
        if self.spc_lm_representation == 'pooler_output':
            prepared_output = pooler_output
        elif self.spc_lm_representation == 'mean_last':
            prepared_output = last_hidden_state.mean(dim=1)
        elif self.spc_lm_representation == 'mean_last_four':
            prepared_output = torch.stack(all_hidden_states[-4:]).mean(dim=0).mean(dim=1)
        elif self.spc_lm_representation == 'mean_last_two':
            prepared_output = torch.stack(all_hidden_states[-2:]).mean(dim=0).mean(dim=1)
        elif self.spc_lm_representation == 'mean_all':
            prepared_output = torch.stack(all_hidden_states).mean(dim=0).mean(dim=1)
        elif self.spc_lm_representation == 'sum_last':
            prepared_output = last_hidden_state.sum(dim=1)
        elif self.spc_lm_representation == 'sum_last_four':
            prepared_output = torch.stack(all_hidden_states[-4:]).sum(dim=0).sum(dim=1)
        elif self.spc_lm_representation == 'sum_last_two':
            prepared_output = torch.stack(all_hidden_states[-2:]).sum(dim=0).sum(dim=1)
        elif self.spc_lm_representation == 'sum_all':
            prepared_output = torch.stack(all_hidden_states).sum(dim=0).sum(dim=1)

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
