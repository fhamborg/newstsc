# -*- coding: utf-8 -*-
# file: SPC_BERT.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn


class SPC_Base(nn.Module):
    def __init__(self, language_model, opt):
        super(SPC_Base, self).__init__()
        self.language_model = language_model
        self.dropout = nn.Dropout(opt.dropout)
        self.name = opt.model_name
        self.spc_lm_representation = opt.spc_lm_representation
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def apply_lm(self, inputs):
        if self.name == 'spc_bert':
            last_hidden, pooler_output, all_hidden = self.language_model(input_ids=inputs[0], #attention_mask=inputs[1],
                                                                         token_type_ids=inputs[1])
        elif self.name == 'spc_roberta':
            last_hidden, pooler_output, all_hidden = self.language_model(input_ids=inputs[0])#            # attention_mask=inputs[1])
        elif self.name == 'spc_distilbert':
            last_hidden, all_hidden = self.language_model(input_ids=inputs[0])# attention_mask=inputs[1])
            pooler_output = None
        else:
            raise Exception("unknown model name: {}", format(self.name))

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

