import torch
import torch.nn as nn
from layers.squeeze_embedding import SqueezeEmbedding


class DISTILBERT_SPC(nn.Module):
    def __init__(self, distilbert, opt):
        super(DISTILBERT_SPC, self).__init__()
        self.distilbert = distilbert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        _, pooled_output = self.bert(text_bert_indices, bert_segments_ids, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits
