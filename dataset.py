import random
from collections import Counter

import jsonlines
import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset
from tqdm import tqdm

from fxlogger import get_logger

# get logger
logger = get_logger()


class RandomOversampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset: Dataset, random_seed=None):
        x = []
        y = []
        for ind, example in enumerate(dataset):
            x.append(ind)
            y.append(example['polarity'])

        x_arr = np.asarray(x).reshape((len(x), 1))
        y_arr = np.asarray(y).ravel()

        ros = RandomOverSampler(random_state=random_seed)
        x_sampled, y_sampled = ros.fit_resample(x_arr, y_arr)
        self.sampled_indexes = x_sampled.ravel().tolist()
        sampled_labels = y_sampled.tolist()

        assert len(self.sampled_indexes) == len(sampled_labels)

        random.shuffle(self.sampled_indexes)

        get_logger().info(
            f"upsampled to {len(self.sampled_indexes)} samples. label distribution: {Counter(sampled_labels)}")

    def __len__(self):
        return len(self.sampled_indexes)

    def __iter__(self):
        return iter(self.sampled_indexes)


class FXDataset(Dataset):
    def __init__(self, filepath, tokenizer, named_polarity_to_class_number, sorted_expected_label_names,
                 use_tp_placeholders, absa_task_format=False, devmode=False):
        self.polarity_associations = named_polarity_to_class_number
        self.sorted_expected_label_names = sorted_expected_label_names
        self.tokenizer = tokenizer
        self.data = []
        self.use_target_phrase_placeholders = use_tp_placeholders
        self.absa_task_format = absa_task_format

        logger.info("reading dataset file {}".format(filepath))

        tasks = []
        with jsonlines.open(filepath, 'r') as reader:
            for task in reader:
                tasks.append(task)

        if devmode:
            logger.warning("DEV MODE IS ENABLED!")
            logger.info("devmode=True: truncating dataset to 60 lines")
            tasks = tasks[:60]

        self.label_counter = Counter()
        with tqdm(total=len(tasks)) as pbar:
            for task in tasks:
                item, label = self.task_to_dataset_item(task)
                self.label_counter[label] += 1
                self.data.append(item)
                pbar.update(1)
        logger.info("label distribution: {}".format(self.label_counter))

    def task_to_dataset_item(self, task):
        if not self.absa_task_format:
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
            polarity = self.polarity_associations[label]
        else:
            text_left = task['text_left']
            text_right = task['text_right']
            target_phrase = task['target_phrase']
            polarity = int(task['polarity']) + 1
            label = self.sorted_expected_label_names[polarity]
            example_id = -1

        # text to indexes
        example = self.tokenizer.create_text_to_indexes(text_left, target_phrase, text_right,
                                                        self.use_target_phrase_placeholders)

        # add polarity
        example['polarity'] = polarity

        # add reference information
        example['example_id'] = example_id

        # add original text and target (we can use that to create a mistake table)
        example['orig_text'] = text
        example['orig_target'] = target_phrase

        return example, label

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
