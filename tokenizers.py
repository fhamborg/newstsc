import math
import pickle
import time
from abc import ABC, abstractmethod

import numpy as np
import torch
from gensim.models import KeyedVectors
from newstsc.embeddings.glove import gensim_path, pickle_path
from pytorch_transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer

from fxlogger import get_logger


class FXTokenizer(ABC):
    def __init__(self, global_context_max_seqs_per_doc):
        self.logger = get_logger()
        self.count_truncated = 0
        self.count_all_sequences_where_we_count_truncation = 0
        self.count_truncated_long_docs = 0
        self.max_seqs_per_doc = global_context_max_seqs_per_doc

    def create_text_to_indexes(self, text_left, target_phrase, text_right, use_target_phrase_placeholders,
                               global_context=None):
        if use_target_phrase_placeholders:
            target_phrase = 'placeholder'

        text_raw_indices = self.text_to_sequence(text_left + " " + target_phrase + " " + text_right,
                                                 count_truncated=True)
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

        special_text = self.text_to_sequence(
            self.with_special_tokens(text_left + " " + target_phrase + " " + text_right))
        special_text_len = np.sum(special_text != 0)
        special_target = self.text_to_sequence(
            self.with_special_tokens(target_phrase))
        special_target_len = np.sum(special_target != 0)
        special_text_target = self.text_to_sequence(
            self.with_special_tokens(text_left + " " + target_phrase + " " + text_right, target_phrase))
        special_target_text = self.text_to_sequence(
            self.with_special_tokens(target_phrase, text_left + " " + target_phrase + " " + text_right))

        segments_ids_text_target = np.asarray([0] * special_text_len + [1] * special_target_len)
        segments_ids_text_target = self.pad_and_truncate(segments_ids_text_target, self.max_seq_len)

        segments_ids_target_text = np.asarray([0] * special_target_len + [1] * special_text_len)
        segments_ids_target_text = self.pad_and_truncate(segments_ids_target_text, self.max_seq_len)

        text_raw_with_special_indices = self.text_to_sequence(
            self.with_special_tokens(text_left + " " + target_phrase + " " + text_right))
        target_phrase_with_special_indexes = self.text_to_sequence(self.with_special_tokens(target_phrase))

        if global_context:
            global_context_ids, global_context_type_ids, global_context_attention_mask = self.long_text_to_sequences(
                global_context)
        else:
            global_context_ids, global_context_type_ids, global_context_attention_mask = None, None, None

        indexes = {
            'special_text_target': special_text_target,
            'special_target_text': special_target_text,
            'segments_ids_text_target': segments_ids_text_target,
            'segments_ids_target_text': segments_ids_target_text,
            'text_raw_with_special_indices': text_raw_with_special_indices,
            'target_phrase_with_special_indexes': target_phrase_with_special_indexes,
            'text_raw_indices': text_raw_indices,
            'text_raw_without_target_phrase_indices': text_raw_without_target_phrase_indices,
            'text_left_indices': text_left_indices,
            'text_left_with_target_phrase_indices': text_left_with_target_phrase_indices,
            'text_right_indices': text_right_indices,
            'text_right_with_target_phrase_indices': text_right_with_target_phrase_indices,
            'target_phrase_indexes': target_phrase_indexes,
            # 'target_phrase_in_text': target_phrase_in_text,
        }

        if global_context:
            for i, id in enumerate(global_context_ids):
                indexes[f"global_context_ids{i}"] = id
                indexes[f"global_context_type_ids{i}"] = global_context_type_ids[i]
                indexes[f"global_context_attention_mask{i}"] = global_context_attention_mask[i]

        return indexes

    @abstractmethod
    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post', count_truncated=False):
        pass

    @abstractmethod
    def long_text_to_sequences(self, text):
        pass

    @abstractmethod
    def with_special_tokens(self, text_a, text_b=None):
        pass

    def pad_and_truncate(self, sequence, maxlen, dtype='int64', padding='post', truncating='post', pad_value=0,
                         count_truncated=False, masked_value=0, unmasked_value=1, get_attention_mask=False):
        x = (np.ones(maxlen) * pad_value).astype(dtype)
        attention_mask = (np.ones(maxlen) * unmasked_value).astype(dtype)

        if count_truncated:
            self.count_all_sequences_where_we_count_truncation += 1
            if len(sequence) > maxlen:
                self.logger.debug("had to truncate by {} items: {}".format(len(sequence) - maxlen, sequence))
                self.count_truncated += 1

        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]

        trunc = np.asarray(trunc, dtype=dtype)

        len_trunc = len(trunc)
        no_attention = (np.ones(maxlen - len_trunc) * masked_value).astype(dtype)

        if padding == 'post':
            x[:len_trunc] = trunc
            attention_mask[len_trunc:] = no_attention
        else:
            x[-len_trunc:] = trunc
            attention_mask[:len(no_attention)] = no_attention

        if get_attention_mask:
            return x, attention_mask
        else:
            return x


class Tokenizer4Distilbert(FXTokenizer):
    def __init__(self, pretrained_distilbert_name, max_seq_len):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_distilbert_name)
        self.max_seq_len = max_seq_len
        self.pad_value = 0
        # verified with
        # print(DistilBertTokenizer.from_pretrained('distilbert-base-uncased').encode("[PAD]"))

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post', count_truncated=False):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return self.pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating,
                                     pad_value=self.pad_value, count_truncated=count_truncated)

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
    def __init__(self, pretrained_bert_name, max_seq_len, global_context_seqs_per_doc):
        super().__init__(global_context_seqs_per_doc)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len
        self.max_seq_per_doc = global_context_seqs_per_doc
        self.pad_value = 0  # taken from original ABSA code, plus the 0th vocab entry is pad, see
        # https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip file: vocab.txt

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post', count_truncated=False):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return self.pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating,
                                     pad_value=self.pad_value, count_truncated=count_truncated)

    def long_text_to_sequences(self, text):
        tokenized_text = self.tokenizer.tokenize(text)

        max_input_length = self.max_seq_len  # before: 512
        max_sequences_per_document = math.ceil(len(tokenized_text) / (max_input_length - 2))
        document_too_large = max_sequences_per_document > self.max_seq_per_doc
        if document_too_large:
            self.logger.debug("document too large")
            self.count_truncated_long_docs += 1

        all_input_ids = []
        all_input_type_ids = []
        all_input_attention_mask = []

        count_seq = 0
        for seq_index, i in enumerate(range(0, len(tokenized_text), (max_input_length - 2))):
            raw_tokens = tokenized_text[i:i + (max_input_length - 2)]
            tokens = []
            input_type_ids = []

            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in raw_tokens:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_masks = [1] * len(input_ids)

            while len(input_ids) < max_input_length:
                input_ids.append(0)
                input_type_ids.append(0)
                attention_masks.append(0)

            assert len(input_ids) == self.max_seq_len and len(attention_masks) == self.max_seq_len and len(
                input_type_ids) == self.max_seq_len

            all_input_ids.append(np.asarray(input_ids).reshape((self.max_seq_len,)))
            all_input_type_ids.append(np.asarray(input_type_ids).reshape((self.max_seq_len,)))
            all_input_attention_mask.append(np.asarray(attention_masks).reshape((self.max_seq_len,)))

            count_seq += 1

        input_ids = [0] * max_input_length
        input_type_ids = [0] * max_input_length
        attention_masks = [0] * max_input_length

        diff = self.max_seq_per_doc - count_seq
        if diff > 0:
            for remainder_index in range(diff):
                all_input_ids.append(np.asarray(input_ids).reshape((self.max_seq_len,)))
                all_input_type_ids.append(np.asarray(input_type_ids).reshape((self.max_seq_len,)))
                all_input_attention_mask.append(np.asarray(attention_masks).reshape((self.max_seq_len,)))

        return all_input_ids, all_input_type_ids, all_input_attention_mask

    def with_special_tokens(self, text_a, text_b=None):
        if text_b:
            return "[CLS] {} [SEP] {} [SEP]".format(text_a, text_b)
        else:
            return "[CLS] {} [SEP]".format(text_a)


class Tokenizer4Roberta(FXTokenizer):
    def __init__(self, pretrained_model_name, max_seq_len):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)
        self.max_seq_len = max_seq_len
        self.pad_value = 1  # <pad> https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json
        # verified with: print(RobertaTokenizer.from_pretrained('roberta-base').encode("<pad>"))

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post', count_truncated=False):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return self.pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating,
                                     pad_value=self.pad_value, count_truncated=count_truncated)

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

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post', count_truncated=False):
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
                                     truncating=truncating, pad_value=self.pad_value, count_truncated=count_truncated)
