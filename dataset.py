import random
from collections import Counter

import jsonlines
import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset
from tqdm import tqdm

from dependencyparser import DependencyParser
from fxlogger import get_logger

# get logger
logger = get_logger()


class RandomOversampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset: Dataset, random_seed=None):
        x = []
        y = []
        for ind, example in enumerate(dataset):
            x.append(ind)
            y.append(example["polarity"])

        x_arr = np.asarray(x).reshape((len(x), 1))
        y_arr = np.asarray(y).ravel()

        ros = RandomOverSampler(random_state=random_seed)
        x_sampled, y_sampled = ros.fit_resample(x_arr, y_arr)
        self.sampled_indexes = x_sampled.ravel().tolist()
        sampled_labels = y_sampled.tolist()

        assert len(self.sampled_indexes) == len(sampled_labels)

        random.shuffle(self.sampled_indexes)

        get_logger().info(
            f"oversampled to {len(self.sampled_indexes)} samples. label distribution: {Counter(sampled_labels)}"
        )

    def __len__(self):
        return len(self.sampled_indexes)

    def __iter__(self):
        return iter(self.sampled_indexes)


class FXDataset(Dataset):
    def __init__(
        self,
        filepath,
        tokenizer,
        named_polarity_to_class_number,
        sorted_expected_label_names,
        use_tp_placeholders,
        task_format="newstsc",
        devmode=False,
        use_global_context=False,
        dependency_parser: DependencyParser = None,
    ):
        self.polarity_associations = named_polarity_to_class_number
        self.sorted_expected_label_names = sorted_expected_label_names
        self.tokenizer = tokenizer
        self.data = []
        self.use_target_phrase_placeholders = use_tp_placeholders
        self.task_format = task_format
        self.use_global_context = use_global_context
        self.dependency_parser = dependency_parser

        logger.info("reading dataset file {}".format(filepath))

        tasks = []
        with jsonlines.open(filepath, "r") as reader:
            for task in reader:
                tasks.append(task)

        if devmode:
            k = 3
            logger.warning("DEV MODE IS ENABLED")
            logger.info("devmode=True: truncating dataset to {} lines".format(k))
            tasks = tasks[:k]

        self.label_counter = Counter()
        with tqdm(total=len(tasks)) as pbar:
            for task in tasks:
                item, label = self.task_to_dataset_item(task)
                self.label_counter[label] += 1
                self.data.append(item)
                pbar.update(1)
        logger.info("label distribution: {}".format(self.label_counter))

    def task_to_dataset_item(self, task):
        if self.task_format == "newstsc_old":
            text = task["text"]
            target_phrase = task["targetphrase"]
            global_context = None
            outlet = task["outlet"]
            year_month_publish = task["year_month_publish"]
            example_id = task["example_id"]
            label = task["label"]
            start_char = task["targetphrase_in_sentence_start"]
            end_char = task["targetphrase_in_sentence_end"]
            text_left = text[: start_char - 1]
            text_right = text[end_char + 1 :]
            polarity = self.polarity_associations[label]

        elif self.task_format == "newstsc":
            target_phrase = task["target_mention"]
            # global_context = task["global_context"]
            global_context = None

            text = task["local_context"]
            start_char = task["target_local_from"]
            end_char = task["target_local_to"]
            text_left = text[: start_char - 1]
            text_right = text[end_char + 1 :]

            label = task["label"]
            polarity = self.polarity_associations[label]

            example_id = task.get("example_id", -1)

        elif self.task_format == "absa":
            text_left = task["text_left"]
            text_right = task["text_right"]
            target_phrase = task["target_phrase"]
            polarity = int(task["polarity"]) + 1
            label = self.sorted_expected_label_names[polarity]
            global_context = None
            example_id = -1
        else:
            raise Exception

        focus_vector = None
        if self.dependency_parser:
            focus_vector = self.dependency_parser.process_sentence(
                text, start_char, end_char
            )

        # text to indexes
        example = self.tokenizer.create_text_to_indexes(
            text_left,
            target_phrase,
            text_right,
            self.use_target_phrase_placeholders,
            global_context=global_context,
            focus_vector=focus_vector,
        )

        # add polarity
        example["polarity"] = polarity

        # add reference information
        example["example_id"] = example_id

        # add original text and target (we can use that to create a mistake table)
        example["orig_text"] = text
        example["orig_target"] = target_phrase

        return example, label

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
