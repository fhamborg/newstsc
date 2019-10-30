from collections import Counter

import jsonlines
from torch.utils.data import Dataset
from tqdm import tqdm

from fxlogger import get_logger

# get logger
logger = get_logger()


class FXDataset(Dataset):
    def __init__(self, filepath, tokenizer, named_polarity_to_class_number, dev_mode):
        self.polarity_associations = named_polarity_to_class_number
        self.tokenizer = tokenizer
        self.data = []

        logger.info("reading dataset file {}".format(filepath))

        tasks = []
        with jsonlines.open(filepath, 'r') as reader:
            for task in reader:
                tasks.append(task)

        if dev_mode:
            logger.warning("DEV MODE IS ENABLED!!!")
            logger.warning("DEV MODE IS ENABLED!!!")
            logger.warning("DEV MODE IS ENABLED!!!")
            logger.info("devmode=True: truncating dataset to 60 lines")
            tasks = tasks[:60]

        self.label_counter = Counter()
        with tqdm(total=len(tasks)) as pbar:
            for task in tasks:
                item = self.task_to_dataset_item(task)
                self.label_counter[task['label']] += 1
                self.data.append(item)
                pbar.update(1)
        logger.info("label distribution: {}".format(self.label_counter))

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
        data = self.tokenizer.create_text_to_indexes(text_left, target_phrase, text_right, polarity)

        # add reference information
        data['example_id'] = example_id

        return data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
