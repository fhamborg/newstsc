# -*- coding: utf-8 -*-
# file: lcf_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.


# The code is based on repository: https://github.com/yangheng95/LCF-ABSA


import numpy as np
import torch
import torch.nn as nn
from pytorch_transformers.modeling_bert import BertPooler, BertSelfAttention


class GlobalContext(nn.Module):
    def __init__(self, global_context_seqs_per_doc):
        super(GlobalContext, self).__init__()
        self.global_context_seqs_per_doc = global_context_seqs_per_doc

    def forward(self, inputs):
        pass


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(
            np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len), dtype=np.float32),
            dtype=torch.float32,
        ).to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class LCF_BERT(nn.Module):
    def __init__(self, bert, opt, is_global_configuration=False):
        super(LCF_BERT, self).__init__()

        self.bert_spc = bert
        self.opt = opt
        self.is_global_configuration = is_global_configuration

        # self.bert_local = copy.deepcopy(bert)  # Uncomment the line to use dual Bert
        self.bert_local = (
            bert  # Default to use single Bert and reduce memory requirements
        )
        self.dropout = nn.Dropout(self.opt.dropout)
        self.bert_SA = SelfAttention(bert.config, self.opt)
        self.linear_double = nn.Linear(self.opt.bert_dim * 2, self.opt.bert_dim)
        self.linear_single = nn.Linear(self.opt.bert_dim, self.opt.bert_dim)
        self.bert_pooler = BertPooler(bert.config)

        if self.opt.use_global_context:
            self.gc_bert = bert
            self.linear_gc_merger = nn.Linear(
                self.opt.bert_dim * self.opt.global_context_seqs_per_doc,
                self.opt.bert_dim,
            )
            self.linear_lcf_and_gc_merger = nn.Linear(
                self.opt.bert_dim * 2, self.opt.bert_dim
            )

        if not self.is_global_configuration:
            self.dense = nn.Linear(self.opt.bert_dim, self.opt.polarities_dim)

    def feature_dynamic_mask(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        mask_len = self.opt.SRD
        masked_text_raw_indices = np.ones(
            (text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
            dtype=np.float32,
        )
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
            except:
                continue
            if asp_begin >= mask_len:
                mask_begin = asp_begin - mask_len
            else:
                mask_begin = 0
            for i in range(mask_begin):
                masked_text_raw_indices[text_i][i] = np.zeros(
                    (self.opt.bert_dim), dtype=np.float
                )
            for j in range(asp_begin + asp_len + mask_len, self.opt.max_seq_len):
                masked_text_raw_indices[text_i][j] = np.zeros(
                    (self.opt.bert_dim), dtype=np.float
                )
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def feature_dynamic_weighted(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        masked_text_raw_indices = np.ones(
            (text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
            dtype=np.float32,
        )
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
                asp_avg_index = (asp_begin * 2 + asp_len) / 2
            except:
                continue
            distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
            for i in range(1, np.count_nonzero(texts[text_i]) - 1):
                if abs(i - asp_avg_index) + asp_len / 2 > self.opt.SRD:
                    distances[i] = 1 - (
                        abs(i - asp_avg_index) + asp_len / 2 - self.opt.SRD
                    ) / np.count_nonzero(texts[text_i])
                else:
                    distances[i] = 1
            for i in range(len(distances)):
                masked_text_raw_indices[text_i][i] = (
                    masked_text_raw_indices[text_i][i] * distances[i]
                )
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def _get_inputs_for_global_context_segment(self, inputs, segment_index, offset):
        assert (
            segment_index < self.opt.global_context_seqs_per_doc
        ), "segment_index({}) >= max_num_components({})".format(
            segment_index, self.opt.global_context_seqs_per_doc
        )

        in0 = inputs[offset + segment_index * 3]
        in1 = inputs[offset + segment_index * 3 + 1]
        in2 = inputs[offset + segment_index * 3 + 2]

        return in0, in1, in2

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, text_local_indices, aspect_indices = (
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
        )
        batch_size = text_bert_indices.shape[0]

        bert_spc_out, _, _ = self.bert_spc(text_bert_indices, bert_segments_ids)
        bert_spc_out = self.dropout(bert_spc_out)

        bert_local_out, _, _ = self.bert_local(text_local_indices)
        bert_local_out = self.dropout(bert_local_out)

        if self.opt.local_context_focus == "cdm":
            masked_local_text_vec = self.feature_dynamic_mask(
                text_local_indices, aspect_indices
            )
            bert_local_out = torch.mul(bert_local_out, masked_local_text_vec)

        elif self.opt.local_context_focus == "cdw":
            weighted_text_local_features = self.feature_dynamic_weighted(
                text_local_indices, aspect_indices
            )
            bert_local_out = torch.mul(bert_local_out, weighted_text_local_features)

        out_cat = torch.cat((bert_local_out, bert_spc_out), dim=-1)
        mean_pool = self.linear_double(out_cat)
        self_attention_out = self.bert_SA(mean_pool)
        pooled_out = self.bert_pooler(self_attention_out)

        if self.opt.use_global_context:
            lst_gc_segments = []

            for i in range(self.opt.global_context_seqs_per_doc):
                gci, gct, gca = self._get_inputs_for_global_context_segment(
                    inputs, i, 4
                )
                _, gc_bert_pooler_output, _ = self.gc_bert(gci, gct, gca)
                gc_bert_pooler_output = self.dropout(gc_bert_pooler_output)
                lst_gc_segments.append(gc_bert_pooler_output)

            # merge with main output
            gc_segments = torch.cat(lst_gc_segments, dim=-1)
            merged_gc_segments = self.linear_gc_merger(gc_segments)
            merged_pooled_out_and_global_context = torch.cat(
                (pooled_out, merged_gc_segments), dim=-1
            )
            pooled_out = self.linear_lcf_and_gc_merger(
                merged_pooled_out_and_global_context
            )

        # if we use LCF as is, i.e., not in the context of globalsenti, then apply the original last dense layer to get
        # 3 classes for each document
        if not self.is_global_configuration:
            dense_out = self.dense(pooled_out)
            return dense_out
        # otherwise return the pooled output, since it seems to be a proper representation of LCF's output
        else:
            return pooled_out
