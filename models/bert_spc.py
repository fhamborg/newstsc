# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch.nn as nn


class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.reduction = opt.bert_spc_reduction
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]

        # according to https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        # we should not use (as in the original SPC version) the pooler_output but get the last_hidden_state and
        # "averaging or pooling the sequence of hidden-states for the whole input sequence" TODO
        last_hidden_state, pooler_output = self.bert(text_bert_indices, bert_segments_ids)

        if self.reduction == 'pooler_output':
            prepared_output = pooler_output
        elif self.reduction == 'mean_last_hidden_states':
            mean_last_hidden_state = last_hidden_state.mean(dim=1, keepdim=True)
            prepared_output = mean_last_hidden_state[:, 0, :]

        prepared_output = self.dropout(prepared_output)
        logits = self.dense(prepared_output)

        return logits
