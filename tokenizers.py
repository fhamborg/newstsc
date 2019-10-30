import pickle
import time
from abc import ABC, abstractmethod

import numpy as np
import torch
from gensim.models import KeyedVectors
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer

from embeddings.glove import gensim_path, pickle_path
from fxlogger import get_logger


class FXTokenizer(ABC):
    def __init__(self):
        self.logger = get_logger()

    def create_text_to_indexes(self, text_left, target_phrase, text_right, polarity):
        text_raw_indices = self.text_to_sequence(text_left + " " + target_phrase + " " + text_right)
        text_raw_without_target_phrase_indices = self.text_to_sequence(text_left + " " + text_right)
        text_left_indices = self.text_to_sequence(text_left)
        text_left_with_target_phrase_indices = self.text_to_sequence(text_left + " " + target_phrase)
        text_right_indices = self.text_to_sequence(text_right, reverse=True)
        text_right_with_target_phrase_indices = self.text_to_sequence(" " + target_phrase + " " + text_right,
                                                                      reverse=True)
        target_phrase_indexes = self.text_to_sequence(target_phrase)
        left_context_len = np.sum(text_left_indices != 0)
        target_phrase_len = np.sum(target_phrase_indexes != 0)
        target_phrase_in_text = torch.tensor(
            [left_context_len.item(), (left_context_len + target_phrase_len - 1).item()])

        text_with_special_indexes = self.text_to_sequence(
            self.with_special_tokens(text_left + " " + target_phrase + " " + text_right, target_phrase))

        bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (target_phrase_len + 1))
        bert_segments_ids = FXTokenizer.pad_and_truncate(bert_segments_ids, self.max_seq_len)

        text_raw_with_special_indices = self.text_to_sequence(
            self.with_special_tokens(text_left + " " + target_phrase + " " + text_right))
        target_phrase_with_special_indexes = self.text_to_sequence(self.with_special_tokens(target_phrase))

        data = {
            'text_with_special_indexes': text_with_special_indexes,
            'bert_segments_ids': bert_segments_ids,
            'text_raw_with_special_indices': text_raw_with_special_indices,
            'target_phrase_with_special_indexes': target_phrase_with_special_indexes,
            'text_raw_indices': text_raw_indices,
            'text_raw_without_aspect_indices': text_raw_without_target_phrase_indices,
            'text_left_indices': text_left_indices,
            'text_left_with_aspect_indices': text_left_with_target_phrase_indices,
            'text_right_indices': text_right_indices,
            'text_right_with_aspect_indices': text_right_with_target_phrase_indices,
            'target_phrase_indexes': target_phrase_indexes,
            'target_phrase_in_text': target_phrase_in_text,
            'polarity': polarity,
        }
        return data

    @abstractmethod
    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        pass

    @abstractmethod
    def with_special_tokens(self, text_a, text_b=None):
        pass

    @staticmethod
    def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', pad_value=0):
        x = (np.ones(maxlen) * pad_value).astype(dtype)

        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]

        trunc = np.asarray(trunc, dtype=dtype)

        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc

        return x


class Tokenizer4Distilbert(FXTokenizer):
    def __init__(self, max_seq_len, pretrained_distilbert_name):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_distilbert_name)
        self.max_seq_len = max_seq_len
        self.pad_value = 0
        # verified with
        # print(DistilBertTokenizer.from_pretrained('distilbert-base-uncased').encode("[PAD]"))

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return self.pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating,
                                     pad_value=self.pad_value)

    def with_special_tokens(self, text_a, text_b=None):
        # https://huggingface.co/transformers/model_doc/distilbert.html#distilbertmodel
        # DistilBert doesn’t have token_type_ids, you don’t need to indicate which token belongs to which segment.
        # Just separate your segments with the separation token tokenizer.sep_token (or [SEP])
        # https://huggingface.co/transformers/model_doc/distilbert.html#distilberttokenizer
        if text_b:
            return "[CLS] {} [SEP] {} [SEP]".format(text_a, text_b)
        else:
            return "[CLS] {} [SEP]".format(text_a)


class Tokenizer4Bert(FXTokenizer):
    def __init__(self, max_seq_len, pretrained_bert_name):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len
        self.pad_value = 0  # taken from original ABSA code, plus the 0th vocab entry is pad, see
        # https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip file: vocab.txt

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return self.pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating,
                                     pad_value=self.pad_value)

    def with_special_tokens(self, text_a, text_b=None):
        if text_b:
            return "[CLS] {} [SEP] {} [SEP]".format(text_a, text_b)
        else:
            return "[CLS] {} [SEP]".format(text_a)


class Tokenizer4Roberta(FXTokenizer):
    def __init__(self, max_seq_len, pretrained_model_name):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)
        self.max_seq_len = max_seq_len
        self.pad_value = 1  # <pad> https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json
        # verified with: print(RobertaTokenizer.from_pretrained('roberta-base').encode("<pad>"))

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return self.pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating,
                                     pad_value=self.pad_value)

    def with_special_tokens(self, text_a, text_b=None):
        if text_b:
            return "<s> {} </s> </s> {} </s>".format(text_a, text_b)
        else:
            return "<s> {} </s>".format(text_a)


class Tokenizer4GloVe(FXTokenizer):
    def __init__(self, max_seq_len, lower=True, force_build=False, dev_mode=False):
        super().__init__()
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.gensim_path = gensim_path
        self.pickle_path = pickle_path
        self.pad_value = 0  # the 0th row in the embedding matrix is a 0 vector

        require_new_build = force_build

        if not require_new_build:
            # attempt deserializing (takes only 1 min on my MBP 2016)
            if not self.attempt_deserialize():
                require_new_build = True

        if require_new_build:
            # takes 16 min on my MacBook Pro 2016
            # 15 min (for reading the GloVe txt file) + 1 min (for building the matrix) + 2 min (for serializing)
            word_limit = None
            if dev_mode:
                word_limit = 1000
                self.gensim_path = self.gensim_path + '.dev'
                self.pickle_path = self.pickle_path + '.dev'

            self.logger.info(f"loading GloVe from {self.gensim_path}...")
            start = time.time()
            glove = KeyedVectors.load_word2vec_format(gensim_path, limit=word_limit, binary=False)
            elapsed = (time.time() - start) / 60
            self.logger.info("done in {:.2f} min".format(elapsed))
            self.vocab = glove.vocab

            # build embeddings matrix
            # idx 0 and len(word2idx)+1 are all-zeros
            self.embedding_matrix = np.insert(glove.vectors, 0, np.zeros((1, 300)), axis=0)
            self.embedding_matrix = np.append(self.embedding_matrix, np.zeros((1, 300)), axis=0)
            self.logger.debug("built matrix for embeddings")

            # store to disk what we need next time
            self.serialize()

    def serialize(self):
        with open(self.pickle_path, 'wb') as handle:
            pickle.dump((self.embedding_matrix, self.vocab), handle, protocol=4)
            self.logger.info("serialized Tokenizer4GloVe to {}".format(self.pickle_path))

    def attempt_deserialize(self):
        try:
            with open(self.pickle_path, 'rb') as handle:
                self.logger.info("trying to deserialize Tokenizer4GloVe from {}...".format(self.pickle_path))
                self.embedding_matrix, self.vocab = pickle.load(handle)
                self.logger.info("done")
                return True
        except Exception as e:
            self.logger.info("failed: {}".format(e))
            return False

    def with_special_tokens(self, text_a, text_b=None):
        return ""

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        unknownidx = len(self.vocab) + 1

        if self.lower:
            text = text.lower()
        words = text.split()

        sequence_indexes = []
        for word in words:
            vocab_item = self.vocab.get(word)
            if vocab_item:
                word_index = vocab_item.index
            else:
                word_index = unknownidx
            sequence_indexes.append(word_index)

        if len(sequence_indexes) == 0:
            sequence_indexes = [0]
        if reverse:
            sequence_indexes = sequence_indexes[::-1]

        return self.pad_and_truncate(sequence_indexes, self.max_seq_len, padding=padding,
                                     truncating=truncating, pad_value=self.pad_value)
