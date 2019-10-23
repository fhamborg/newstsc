import os
import pickle

import jsonlines
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer, DistilBertTokenizer

from fxlogger import get_logger

# get logger
logger = get_logger()

DEV_MODE = True

if DEV_MODE:
    logger.warning("DEV MODE IS ENABLED!!!")
    logger.warning("DEV MODE IS ENABLED!!!")
    logger.warning("DEV MODE IS ENABLED!!!")


def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        logger.info('loading tokenizer: {}'.format(dat_fname))
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


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


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Distilbert:
    def __init__(self, max_seq_len, pretrained_distilbert_name):
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_distilbert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class TextTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def create_text_to_indexes(self, text_left, target_phrase, text_right, polarity):
        text_raw_indices = self.tokenizer.text_to_sequence(text_left + " " + target_phrase + " " + text_right)
        text_raw_without_target_phrase_indices = self.tokenizer.text_to_sequence(text_left + " " + text_right)
        text_left_indices = self.tokenizer.text_to_sequence(text_left)
        text_left_with_target_phrase_indices = self.tokenizer.text_to_sequence(text_left + " " + target_phrase)
        text_right_indices = self.tokenizer.text_to_sequence(text_right, reverse=True)
        text_right_with_target_phrase_indices = self.tokenizer.text_to_sequence(" " + target_phrase + " " + text_right,
                                                                                reverse=True)
        target_phrase_indices = self.tokenizer.text_to_sequence(target_phrase)
        left_context_len = np.sum(text_left_indices != 0)
        target_phrase_len = np.sum(target_phrase_indices != 0)
        target_phrase_in_text = torch.tensor(
            [left_context_len.item(), (left_context_len + target_phrase_len - 1).item()])

        text_bert_indices = self.tokenizer.text_to_sequence(
            '[CLS] ' + text_left + " " + target_phrase + " " + text_right + ' [SEP] ' + target_phrase + " [SEP]")
        bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (target_phrase_len + 1))
        bert_segments_ids = pad_and_truncate(bert_segments_ids, self.tokenizer.max_seq_len)

        text_raw_bert_indices = self.tokenizer.text_to_sequence(
            "[CLS] " + text_left + " " + target_phrase + " " + text_right + " [SEP]")
        target_phrase_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + target_phrase + " [SEP]")

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


class FXDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        self.polarity_associations = {'positive': 2, 'neutral': 1, 'negative': 0}

        self.data_preparer = TextTokenizer(tokenizer)
        self.data = []

        logger.info("reading dataset file {}".format(filepath))

        tasks = []
        with jsonlines.open(filepath, 'r') as reader:
            for task in reader:
                tasks.append(task)

        if DEV_MODE:
            logger.info("devmode=True: truncating dataset to 20 lines")
            tasks = tasks[:20]

        with tqdm(total=len(tasks)) as pbar:
            for task in tasks:
                item = self.task_to_dataset_item(task)
                self.data.append(item)
                pbar.update(1)

    def task_to_dataset_item(self, task):
        text = task['text']
        target_phrase = task['targetphrase']
        outlet = task['outlet']
        year_month_publish = task['year_month_publish']
        example_id = task['example_id']
        label = task['label']
        start_char = task['targetphrase_in_sentence_start']
        end_char = task['targetphrase_in_sentence_end']
        text_left = text[:start_char - 1]
        text_right = text[end_char + 1:]
        polarity = self.polarity_associations[task['label']]

        # text to indexes
        data = self.data_preparer.create_text_to_indexes(text_left, target_phrase, text_right, polarity)

        # add reference information
        data['example_id'] = example_id

        return data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
