# -*- coding: utf-8 -*-
# file: lcf_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.


# The code is based on repository: https://github.com/yangheng95/LCF-ABSA


import torch
import torch.nn as nn

from models.lcf import LCF_BERT


class Global_LCF(nn.Module):
    def __init__(self, bert, opt):
        super(Global_LCF, self).__init__()

        self.max_num_components = 20

        self.lcf = LCF_BERT(bert, opt, is_global_configuration=True)
        self.linear_merge_remainder_comps = nn.Linear(opt.bert_dim * self.max_num_components, opt.bert_dim)
        self.linear_merge_lcf_and_remainder = nn.Linear(opt.bert_dim * 2, opt.polarities_dim)

    def _get_inputs_for_component(self, inputs, component_index):
        assert component_index < self.max_num_components, "component_index({}) >= max_num_components({})".format(
            component_index, self.max_num_components)

        return [inputs[component_index * 4], inputs[component_index * 4 + 1], inputs[component_index * 4 + 2], inputs[
            component_index * 4 + 3]]

    def forward(self, inputs):
        # this is the main component, which we want to classify
        main_comp_inputs = self._get_inputs_for_component(inputs, 0)

        main_lcf_output = self.lcf(main_comp_inputs)

        # process remaining document components, which we don't want to classify but use as context
        # TODO maybe disable gradient in these components? or at least in BERT in them?
        lst_remainder_comp_outputs = []
        for i in range(1, self.max_num_components):
            cur_comp_inputs = self._get_inputs_for_component(inputs, i)
            cur_comp_output = self.lcf(cur_comp_inputs)
            lst_remainder_comp_outputs.append(cur_comp_output)

        remainder_comp_outputs = torch.cat(lst_remainder_comp_outputs, dim=-1)
        remainder_merged = self.linear_merge_remainder_comps(remainder_comp_outputs)

        dense_out = self.linear_merge_lcf_and_remainder(main_lcf_output, remainder_merged)

        return dense_out
