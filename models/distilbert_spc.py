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
        text_bert_indices = inputs[0]

        last_hidden_state = self.distilbert(text_bert_indices)[0]

        mean_last_hidden_state = last_hidden_state.mean(dim=1, keepdim=True)
        prepared_output = mean_last_hidden_state[:, 0, :]

        prepared_output = self.dropout(prepared_output)
        logits = self.dense(prepared_output)

        return logits
