import jsonlines
from collections import Counter



files = ['train.jsonl', "test.jsonl"]
for file in files:
    classes_train = Counter()

    with jsonlines.open(file) as reader:
        for item in reader:
            classes_train[item["label"]] += 1

    print(classes_train)
