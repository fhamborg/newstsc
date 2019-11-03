from jsonlines import jsonlines
from tqdm import tqdm


def convert(filepath):
    fin = open(filepath, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()

    all_data = []

    with tqdm(total=len(lines)) as pbar:
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            target_phrase = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            data = {}
            data['text_left'] = text_left
            data['text_right'] = text_right
            data['target_phrase'] = target_phrase
            data['polarity'] = polarity
            all_data.append(data)
            pbar.update(3)

    with jsonlines.open(filepath + '.jsonl', 'w') as writer:
        writer.write_all(all_data)


if __name__ == '__main__':
    convert('controller_data/datasets/acl14twitter/test.raw')
    convert('controller_data/datasets/acl14twitter/train.raw')
    convert('controller_data/datasets/semeval14restaurants/Restaurants_Test_Gold.xml.seg')
    convert('controller_data/datasets/semeval14restaurants/Restaurants_Train.xml.seg')
    convert('controller_data/datasets/semeval14laptops/Laptops_Test_Gold.xml.seg')
    convert('controller_data/datasets/semeval14restaurants/Restaurants_Train.xml.seg')
