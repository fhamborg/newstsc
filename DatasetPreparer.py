"""
Used by controller.py to prepare datasets for an experiment, thereby most importantly considering whether the experiment
uses cross validation or not.

Currently, the class is partially hard coded for the poltsanews dataset.
"""
import math
import os
import random
from collections import Counter

import jsonlines
from tabulate import tabulate

from fxlogger import get_logger


class DatasetPreparer:
    def __init__(self):
        self.name = 'poltsanews'
        self.human_created_filenames = ['human.jsonl']
        self.non_human_created_filenames = ['train_20191021_233454.jsonl']
        self.human_created_filepaths = [self.get_filepath_by_name(x) for x in self.human_created_filenames]
        self.non_human_created_filepaths = [self.get_filepath_by_name(x) for x in self.non_human_created_filenames]
        self.data_types = ['human', 'nonhum']

        self.random_seed = 1337
        random.seed(self.random_seed)
        self.logger = get_logger()

        self.examples_human = self.files_to_dictlst(self.human_created_filepaths)
        self.examples_nonhum = self.files_to_dictlst(self.non_human_created_filepaths)

        self.logger.info("shuffling example lists with seed {}".format(self.random_seed))
        random.shuffle(self.examples_human)
        random.shuffle(self.examples_nonhum)

        self.logger.info(
            "{} examples read created by humans (from: {})".format(len(self.examples_human),
                                                                   self.human_created_filepaths))
        self.logger.info("{} examples read not created by humans (from: {})".format(len(self.examples_nonhum),
                                                                                    self.non_human_created_filepaths))

    def get_filepath_by_name(self, filename):
        return os.path.join('datasets', self.name, filename)

    def file_to_dictlst(self, filepath):
        dict_lst = []
        with jsonlines.open(filepath, 'r') as reader:
            for line in reader:
                dict_lst.append(line)
        self.logger.debug("{} examples read from {}".format(len(dict_lst), filepath))
        return dict_lst

    def files_to_dictlst(self, list_filepaths):
        dict_lst = []
        for filepath in list_filepaths:
            dict_lst.extend(self.file_to_dictlst(filepath))
        return dict_lst

    def create_slices(self, sets_info, data_type):
        """

        :param sets_info:
        :param data_type: 'human', 'nonhum'
        :return:
        """
        set_names = list(sets_info.keys())
        assert data_type in self.data_types

        # get some vars
        _id_relative_weight = '{}-rel-weight'.format(data_type)
        _id_examples = '{}-examples'.format(data_type)

        prev_split_pos = 0
        for set_index, set_name in enumerate(set_names):
            cur_setinfo = sets_info[set_name]

            if _id_relative_weight in cur_setinfo:
                cur_relative_weight = cur_setinfo[_id_relative_weight]

                if cur_relative_weight:
                    split_pos = prev_split_pos + math.floor(cur_relative_weight * len(self.examples_human))
                    if set_index == len(set_names) - 1:
                        # just to be sure to not miss a single example because of rounding
                        split_pos = len(self.examples_human)

                    example_slice = self.examples_human[prev_split_pos:split_pos]
                    cur_setinfo[_id_examples] = example_slice
                    self.logger.info("{}: added {} {} examples ({:.2f})".format(set_name, len(example_slice), data_type,
                                                                                cur_relative_weight))

                    prev_split_pos = split_pos
                else:
                    cur_setinfo[_id_examples] = []

    def merge_slices(self, sets_info):
        total_examples_count = 0
        for set_name, cur_set in sets_info.items():
            cur_merged_data = []
            for data_type in self.data_types:
                _id_examples = '{}-examples'.format(data_type)

                if _id_examples in cur_set:
                    cur_merged_data.extend(cur_set[_id_examples])

            cur_set['examples'] = cur_merged_data
            total_examples_count += len(cur_merged_data)

        for set_name, cur_set in sets_info.items():
            cur_set['examples-rel'] = len(cur_set['examples']) / total_examples_count

    def print_set_info(self, sets_info):
        header = ['set_name', 'human rel', 'human abs', 'non-hum rel', 'non-hum abs', 'rel', 'abs']
        rows = []
        for set_name, cur_set in sets_info.items():
            row = [set_name, cur_set['human-rel-weight'], len(cur_set['human-examples']), cur_set['nonhum-rel-weight'],
                   len(cur_set['nonhum-examples']), cur_set['examples-rel'], len(cur_set['examples'])]
            rows.append(row)

        self.logger.info('\n' + tabulate(rows, header))

    def create_set(self, sets_info):
        set_names = list(sets_info.keys())

        weights_human = Counter()
        weights_nonhum = Counter()
        nonnull_set_names = []
        # sum weights and thereby filter datasets that are 0 in size
        for set_name in set_names:
            setinfo = sets_info[set_name]
            weights_human[set_name] = setinfo['human-weight']
            weights_nonhum[set_name] = setinfo['nonhum-weight']

            if setinfo['human-weight'] + setinfo['nonhum-weight'] > 0:
                nonnull_set_names.append(set_name)
            else:
                self.logger.info("discard {}, because would be empty".format(set_name))
        set_names = nonnull_set_names

        human_weight_sum = sum(weights_human.values())
        nonhuman_weight_sum = sum(weights_nonhum.values())

        # add relative weights
        for set_name in set_names:
            cur_setinfo = sets_info[set_name]
            if cur_setinfo['human-weight']:
                cur_setinfo['human-rel-weight'] = cur_setinfo['human-weight'] / human_weight_sum
            else:
                cur_setinfo['human-rel-weight'] = 0
            if cur_setinfo['nonhum-weight']:
                cur_setinfo['nonhum-rel-weight'] = cur_setinfo['nonhum-weight'] / nonhuman_weight_sum
            else:
                cur_setinfo['nonhum-rel-weight'] = 0
        # at this point, setsinfo contains only positive (non-0) relative weights (or no such key, if the abs. value was 0, too)

        # create slices
        self.create_slices(sets_info, 'human')
        self.create_slices(sets_info, 'nonhum')

        # merge human and non human sets in each set
        self.merge_slices(sets_info)

        self.print_set_info(sets_info)

    @classmethod
    def poltsanews_rel801010(cls):
        sets_info = {
            'train':
                {'human-weight': 80, 'nonhum-weight': 0},
            'dev':
                {'human-weight': 10, 'nonhum-weight': 0},
            'test':
                {'human-weight': 10, 'nonhum-weight': 0},
        }
        return cls().create_set(sets_info)


if __name__ == '__main__':
    DatasetPreparer.poltsanews_rel801010()
