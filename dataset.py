from collections import Counter

import jsonlines
from torch.utils.data import Dataset
from tqdm import tqdm

from fxlogger import get_logger

# get logger
logger = get_logger()


class FXDataset(Dataset):
    def __init__(self, filepath, tokenizer, named_polarity_to_class_number, column_getter, use_tp_placeholders,
                 devmode=False):
        self.polarity_associations = named_polarity_to_class_number
        self.tokenizer = tokenizer
        self.data = []
        self.use_target_phrase_placeholders = use_tp_placeholders

        logger.info("reading dataset file {}".format(filepath))

        tasks = []
        with jsonlines.open(filepath, 'r') as reader:
            for task in reader:
                tasks.append(task)

        if devmode:
            logger.warning("DEV MODE IS ENABLED!!!")
            logger.info("devmode=True: truncating dataset to 60 lines")
            tasks = tasks[:60]

        self.label_counter = Counter()
        with tqdm(total=len(tasks)) as pbar:
            for task in tasks:
                item = self.task_to_dataset_item(task)
                example = self.get_relevant_data_from_dataset_item(item, column_getter)
                self.label_counter[task['label']] += 1
                self.data.append(example)
                pbar.update(1)
        logger.info("label distribution: {}".format(self.label_counter))

    def get_relevant_data_from_dataset_item(self, item, column_getter):
        data = {}
        data['inputs'] = column_getter(item)
        data['polarity'] = item.polarity
        data['example_id'] = item.example_id
        return data

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
        example = self.tokenizer.create_text_to_indexes(text_left, target_phrase, text_right,
                                                        self.use_target_phrase_placeholders)

        # add polarity
        example.polarity = polarity

        # add reference information
        example.example_id = example_id

        return example

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
