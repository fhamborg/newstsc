"""
Definitions:
setup: represents a fixed, unique combination of for instance:
* model, e.g., BERT, DISTILBERT
* hyper-parameters, such as learning rate loss function, weights of classes for loss function
* others, such as softmax,
* input style, e.g., input style: QA style, AEN, ...

Procedure:
Given a single annotated dataset named alldat (also called both.jsonl, since it consists of both splits of the devtest
set):
* splits alldat into testdat and remainderdat

Given a set of experiment setup descriptions
1) Iterate setup descriptions, for each:
a) create 10-fold CV to well evaluate the performance of that particular setup
b) return model from best epoch (or from best fold???!?)
2) retrievel all best performing models and test them on testdat
3) return model that performs best on testdat
"""
import os
import pprint
import subprocess
from collections import Counter
from datetime import datetime
from itertools import product

from tabulate import tabulate
from tqdm import tqdm

from DatasetPreparer import DatasetPreparer
from fxlogger import get_logger


class SetupController:
    def __init__(self):
        self.logger = get_logger()

        self.use_cross_validation = 0  # if 0: do not use cross validation
        args_names_ordered = ['snem', 'model_name', 'optimizer', 'initializer', 'learning_rate', 'batch_size',
                              'lossweighting', 'devmode', 'num_epoch', 'lsr', 'bert_spc_reduction']
        # keys in the dict must match parameter names accepted by train.py. values must match accepted values for such
        # parameters in train.py
        combinations = {
            'model_name': ['distilbert_spc', 'bert_spc', 'aen_bert', 'aen_distilbert', 'aen_glove'],
            'snem': ['recall_avg'],
            'optimizer': ['adam'],
            'initializer': ['xavier_uniform_'],
            # TODO check this and other parameters, compare with available options in train.py
            'learning_rate': ['1e-3', '2e-5', '3e-5', '5e-5'],
            'batch_size': ['16', '32', '64', '128'],
            'lossweighting': ['True', 'False'],
            'devmode': ['True'],
            'num_epoch': ['200'],
            'lsr': ['True', 'False'],
            'bert_spc_reduction': ['pooler_output', 'mean_last_hidden_states']
        }
        # key: name of parameter that is only applied if its conditions are met
        # value: list of tuples, consisting of parameter name and the value it needs to have in order for the
        # condition to be satisfied
        # Note that all tuples in this list are OR connected, so if at least one is satisfied, the conditions are met.
        # If we need AND connected conditions, my idea is to add an outer list, resulting in a list of lists (of
        # tuples) where all lists are AND connected.
        # If a condition is not satisfied, the corresponding parameter will still be pass
        conditions = {
            'bert_spc_reduction': [('model_name', 'bert_spc')]
        }

        assert len(args_names_ordered) == len(combinations.keys())
        assert len(combinations['snem']) == 1

        self.experiment_base_id = datetime.today().strftime('%Y%m%d-%H%M%S')
        self.basecmd = ['python', 'train.py']
        self.basepath = 'controller_data'
        self.basepath_data = os.path.join(self.basepath, 'datasets')

        combination_count = 1
        _combination_values = []
        for arg_name in args_names_ordered:
            arg_values = list(combinations[arg_name])
            combination_count = combination_count * len(arg_values)
            _combination_values.append(arg_values)

        combinations = list(product(*_combination_values))
        assert len(combinations) == combination_count

        self.logger.info(
            "{} arguments, totaling in {} combinations".format(len(args_names_ordered), combination_count))

        # apply conditions
        self.named_combinations, count_duplicates = self._apply_conditions(combinations, args_names_ordered, conditions)
        self.logger.info("applied conditions. removed {} combinations. {} -> {}".format(count_duplicates,
                                                                                        combination_count,
                                                                                        len(self.named_combinations)))
        self.combination_count = len(self.named_combinations)

        self.logger.info("combinations:")
        pp = pprint.PrettyPrinter(indent=2)
        self.logger.info("{}".format(pp.pformat(self.named_combinations)))

        if self.use_cross_validation > 0:
            self.logger.info("using {}-fold cross validation".format(self.use_cross_validation))
            self.dataset_preparer = DatasetPreparer.poltsanews_crossval8010_allhuman(self.basepath_data)
        else:
            self.logger.info("not using cross validation".format(self.use_cross_validation))
            self.dataset_preparer = DatasetPreparer.poltsanews_rel801010_allhuman(self.basepath_data)

    def _apply_conditions(self, combinations, args_names_ordered, conditions):
        named_combinations = []
        seen_experiment_ids = set()
        count_duplicates = 0

        for combination in combinations:
            named_combination = {}
            full_named_combination = self._args_combination_to_single_arg_values(combination, args_names_ordered)

            # for a parameter combination, pass only those parameters that are valid for that combination
            for arg_index, arg_name in enumerate(args_names_ordered):
                # iterate each parameter and validate - using the other parameter names and values - whether its
                # conditions are met
                if self._check_conditions(arg_name, full_named_combination, conditions, args_names_ordered):
                    # if yes, pass it
                    named_combination[arg_name] = combination[arg_index]
                    self.logger.debug("using '{}' in combination {}".format(arg_name, combination))
                else:
                    self.logger.debug("not using '{}' in combination {}".format(arg_name, combination))

            # check if experiment_id of named combination was already seen
            experiment_id = self._experiment_id_from_named_combination(named_combination)
            if experiment_id not in seen_experiment_ids:
                seen_experiment_ids.add(experiment_id)
                named_combinations.append(named_combination)
            else:
                count_duplicates += 1

        return named_combinations, count_duplicates

    def _check_conditions(self, arg_name, full_named_combination, conditions, args_names_ordered):
        """
        For a given parameter, checks whether its conditions are satisfied. If so, returns True, else False.
        :param arg_name:
        :param arg_value:
        :return:
        """
        if arg_name in conditions and len(conditions[arg_name]) >= 1:
            # at this point we know that there are conditions for the given parameters
            or_connected_conditions = conditions[arg_name]

            for cond_tup in or_connected_conditions:
                cond_param_name = cond_tup[0]
                cond_param_value = cond_tup[1]

                # get parameter and its value in current combination
                if full_named_combination[cond_param_name] == cond_param_value:
                    return True

            # since there was at least one condition due to our check above, we return False here, since the for loop
            # did not return True
            return False

        else:
            # if there is no condition associated with arg_name just return true
            return True

    def _build_args(self, named_args):
        args_list = []
        for arg_name, arg_val in named_args.items():
            self._add_arg(args_list, arg_name, arg_val)

        return args_list

    def _add_arg(self, args_list, name, value):
        args_list.append("--" + name)
        args_list.append(str(value))
        return args_list

    def _prepare_experiment_env(self, experiment_path):
        os.makedirs(experiment_path, exist_ok=True)
        self.dataset_preparer.export(os.path.join(experiment_path, 'datasets'))

    def _args_combination_to_single_arg_values(self, args_combination, args_names_ordered):
        args_names_values = {}
        for arg_index, arg_name in enumerate(args_names_ordered):
            args_names_values[arg_name] = args_combination[arg_index]
        return args_names_values

    def _experiment_id_from_named_combination(self, named_combination):
        return "__".join(["{}={}".format(k, v) for (k, v) in named_combination.items()])

    def execute_single_setup(self, named_combination):
        experiment_id = self._experiment_id_from_named_combination(named_combination)
        experiment_path = "./experiments/{}/{}/".format(self.experiment_base_id, experiment_id)

        self._prepare_experiment_env(experiment_path)

        args = self._build_args(named_combination)
        self._add_arg(args, 'dataset_name', 'poltsanews')
        self._add_arg(args, 'experiment_path', experiment_path)
        self._add_arg(args, 'crossval', self.use_cross_validation)

        cmd = self.basecmd + args

        self.logger.info("starting single setup: {}".format(" ".join(cmd)))
        with open(os.path.join(experiment_path, 'stdlog.out'), "w") as file_stdout, open(
                os.path.join(experiment_path, 'stdlog.err'), "w") as file_stderr:
            completed_process = subprocess.run(cmd, stdout=file_stdout, stderr=file_stderr)

        snem = self.get_experiment_result_detailed(experiment_path)

        return {**named_combination,
                **{'rc': completed_process.returncode, 'experiment_id': experiment_id, 'snem': snem}}

    def get_experiment_result_detailed(self, experiment_path):
        # each experiments reports detailed information about its performance to the stdout, immediately before it exits
        # thus, get the last line of the stdlog.out
        lines = [line.rstrip('\n') for line in open(os.path.join(experiment_path, 'stdlog.out'))]

        snem_line = lines[-1]

        return snem_line

    def run(self):
        results = []

        with tqdm(total=self.combination_count) as pbar:
            for named_combination in self.named_combinations:
                result = self.execute_single_setup(named_combination)
                results.append(result)
                pbar.update(1)

        experiments_rc_overview = Counter()
        for result in results:
            rc = result['rc']
            experiments_rc_overview[rc] += 1

            if rc != 0:
                self.logger.warning("experiment did not return 0: {}".format(result['experiment_id']))

        # snem-based performance sort
        results.sort(key=lambda x: x['snem'], reverse=True)
        headers = results[0].keys()
        rows = [x.values() for x in results]

        self.logger.info("all experiments finished. statistics:")
        self.logger.info("return codes: {}".format(experiments_rc_overview))
        self.logger.info("snem-based performances:")
        self.logger.info("\n" + tabulate(rows, headers))


if __name__ == '__main__':
    SetupController().run()
