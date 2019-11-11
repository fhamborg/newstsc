import argparse
import json
import shelve

import pandas as pd

from fxlogger import get_logger

logger = get_logger()


def rename_flatten(dictionary, key_prefix):
    new_dict = {}

    for k, v in dictionary.items():
        new_k = key_prefix + "-" + k
        new_dict[new_k] = v

    return new_dict


def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}


def non_scalar_to_str(d):
    new_d = {}
    for k, v in d.items():
        if type(v) in [list, dict]:
            new_v = json.dumps(v)
        else:
            new_v = v
        new_d[k] = new_v
    return new_d


def main(opt):
    completed_tasks = shelve.open(opt.results_path)
    logger.info("found {} results".format(len(completed_tasks)))

    flattened_results = {}

    for named_id, result in completed_tasks.items():
        if result['rc'] == 0:
            test_stats = rename_flatten(result['details']['test_stats'], 'test_stats')
            dev_stats = rename_flatten(result['details']['dev_stats'], 'dev_stats')

            flattened_result = {**without_keys(result, ['details']), **dev_stats,
                                **test_stats}
        else:
            flattened_result = {**without_keys(result, ['details'])}

        scalared_flattened_result = non_scalar_to_str(flattened_result)
        flattened_results[named_id] = scalared_flattened_result

    df = pd.DataFrame(data=flattened_results.values())
    df.to_excel(opt.results_path + ".xlsx")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, required=True)
    opt = parser.parse_args()

    main(opt)
