import pickle
import time
from abc import ABC, abstractmethod

import numpy as np
import torch
from gensim.models import KeyedVectors
from transformers import BertTokenizer, DistilBertTokenizer

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
        target_phrase_indices = self.text_to_sequence(target_phrase)
        left_context_len = np.sum(text_left_indices != 0)
        target_phrase_len = np.sum(target_phrase_indices != 0)
        target_phrase_in_text = torch.tensor(
            [left_context_len.item(), (left_context_len + target_phrase_len - 1).item()])

        text_bert_indices = self.text_to_sequence(
            '[CLS] ' + text_left + " " + target_phrase + " " + text_right + ' [SEP] ' + target_phrase + " [SEP]")

        bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (target_phrase_len + 1))
        bert_segments_ids = FXTokenizer.pad_and_truncate(bert_segments_ids, self.max_seq_len)

        text_raw_bert_indices = self.text_to_sequence(
            "[CLS] " + text_left + " " + target_phrase + " " + text_right + " [SEP]")
        target_phrase_bert_indices = self.text_to_sequence("[CLS] " + target_phrase + " [SEP]")

        data = {
            'text_bert_indices': text_bert_indices,
            'bert_segments_ids': bert_segments_ids,
            'text_raw_bert_indices': text_raw_bert_indices,
            'aspect_bert_indices': target_phrase_bert_indices,
            'text_raw_indices': text_raw_indices,
            'text_raw_without_aspect_indices': text_raw_without_target_phrase_indices,
            'text_left_indices': text_left_indices,
            'text_left_with_aspect_indices': text_left_with_target_phrase_indices,
            'text_right_indices': text_right_indices,
            'text_right_with_aspect_indices': text_right_with_target_phrase_indices,
            'aspect_indices': target_phrase_indices,
            'aspect_in_text': target_phrase_in_text,
            'polarity': polarity,
        }
        return data

    @abstractmethod
    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        pass

    @staticmethod
    def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
        x = (np.ones(maxlen) * value).astype(dtype)
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

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return self.pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert(FXTokenizer):
    def __init__(self, max_seq_len, pretrained_bert_name):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return self.pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4GloVe(FXTokenizer):
    def __init__(self, max_seq_len, lower=True, force_build=False):
        super().__init__()
        self.lower = lower
        self.max_seq_len = max_seq_len

        require_new_build = force_build

        if not require_new_build:
            # attempt deserializing (takes only 1 min on my MBP 2016)
            if not self.attempt_deserialize():
                require_new_build = True

        if require_new_build:
            # takes 16 min on my MacBook Pro 2016
            # 15 min (for reading the GloVe txt file) + 1 min (for building the matrix) + 2 min (for serializing)
            word_limit = None

            self.logger.info(f"loading GloVe from {gensim_path}...")
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
        with open(pickle_path, 'wb') as handle:
            pickle.dump((self.embedding_matrix, self.vocab), handle, protocol=4)
            self.logger.info("serialized Tokenizer4GloVe to {}".format(pickle_path))

    def attempt_deserialize(self):
        try:
            with open(pickle_path, 'rb') as handle:
                self.logger.info("trying to deserialize Tokenizer4GloVe from {}...".format(pickle_path))
                self.embedding_matrix, self.vocab = pickle.load(handle)
                self.logger.info("done")
                return True
        except Exception as e:
            self.logger.info("failed: {}".format(e))
            return False

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
                                     truncating=truncating)
