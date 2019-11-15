"""
This script is used for the paper and finds from all experiment configurations the best performing variant.
"""

# these are the variants we print in the table, i.e., we need to have a single best value for each combination of them
import shelve

import pandas as pd

language_models = ['default',  # BERT
                   'laptops_1mio_ep30',  # ADA-L
                   'restaurants_10mio_ep3',  # ADA-R
                   'laptops_and_restaurants_2mio_ep15',  # ADA-LR
                   'TODO']  # ADA-N
methods = ['aen_bert', 'spc_bert', 'lcf_bert']
datasets = ['acl14twitter_10', 'semeval14laptops_10', 'semeval14restaurants_10', 'sentinews_10']

# these are possible hyperparameters. for each combination yielded from above, use these hyperparameters to find the best
# just skip this for now and simply search for the best value from all lines, no matter the hyper


# config
performance_table_filename = 'overview.xlsx'
result_basename = 'results_'
col_method = 'model_name'
col_language_model = 'pretrained_model_name'
col_rc = 'rc'
col_details = 'details'
col_dataset = 'dataset'
col_primary_eval = 'recall_avg'
col_secondary_evals = ['accuracy', 'f1_macro', 'f1_posneg']
col_stats = 'test_stats'


def get_best_result(results, lm, method, col_compare):
    best_score = -1
    best_result = None
    for result_name, result in results.items():
        if result[col_method] == method and result[col_language_model] == lm:
            if result[col_rc] == 0:
                score = result[col_details][col_stats][col_compare]
                if score > best_score:
                    best_score = score
                    best_result = result

    return best_score, best_result


def get_additional_scores(result):
    stats = result[col_details][col_stats]
    scores = {}
    for col in col_secondary_evals:
        scores[col] = stats[col]
    return scores


def main():
    performance_table = []

    for dataset in datasets:
        dataset_results_filename = result_basename + dataset
        _results = shelve.open(dataset_results_filename)
        dataset_results = dict(_results)
        _results.close()

        for language_model in language_models:
            for method in methods:
                best_score, best_result = get_best_result(dataset_results, language_model, method, col_primary_eval)
                if best_score != -1:
                    secondary_scores = get_additional_scores(best_result)
                else:
                    secondary_scores = {}

                performance_row = {col_dataset: dataset,
                                   col_language_model: language_model,
                                   col_method: method,
                                   col_primary_eval: best_score,
                                   **secondary_scores}

                performance_table.append(performance_row)

    df = pd.DataFrame(data=performance_table)
    df.to_excel(performance_table_filename)


if __name__ == '__main__':
    main()
