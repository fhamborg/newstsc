import numpy as np
import torch
import torch.nn as nn
from pytorch_transformers.modeling_bert import BertPooler, BertSelfAttention


class FX_BERT(nn.Module):
    def __init__(self, bert, opt, is_global_configuration=False):
        super(FX_BERT, self).__init__()

        self.bert_spc = bert
        self.opt = opt

        if is_global_configuration:
            raise ValueError("not allowed")

        # self.bert_local = copy.deepcopy(bert)  # Uncomment the line to use dual Bert
        self.bert_local = (
            bert  # Default to use single Bert and reduce memory requirements
        )
        self.dropout = nn.Dropout(self.opt.dropout)

        self.dense = nn.Linear(self.opt.bert_dim, self.opt.polarities_dim)

    def forward(self, inputs):
        (
            text_bert_indices,
            bert_segments_ids,
            text_local_indices,
            aspect_indices,
            focus_vector,
        ) = (inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])
        batch_size = text_bert_indices.shape[0]

        (
            bert_text_out,
            bert_text_out_pooler,
            bert_text_out_all_hidden_states,
        ) = self.bert_local(text_local_indices)

        bert_text_out = self.dropout(bert_text_out)

        focus_expanded = focus_vector.reshape(
            (batch_size, self.opt.max_seq_len, 1)
        ).repeat(1, 1, self.opt.bert_dim)
        out = bert_text_out * focus_expanded

        out = out.mean(dim=1)

        # TODO IMPORTANT
        # TODO: focus_vector and text_bert_indices (and others) are currently not aligned probably, because text_bert_indices contains BERT's special characters, such as CLS and SEP

        logits = self.dense(out)

        return logits
