"""
Definitions:
setup: represents a fixed, unique combination of for instance:
* model, e.g., BERT, DISTILBERT
* hyper-parameters, such as learning rate loss function, weights of classes for loss function
* others, such as softmax,
* input style, e.g., input style: QA style, AEN, ...

Procedure:
Given a single annotated dataset named alldat:
* splits alldat into testdat and remainderdat

Given a set of experiment setup descriptions
1) Iterate setup descriptions, for each:
a) create 10-fold CV to well evaluate the performance of that particular setup
b) return model from best epoch (or from best fold???!?)
2) retrievel all best performing models and test them on testdat
3) return model that performs best on testdat
"""
import os
import subprocess
from collections import Counter
from datetime import datetime
from itertools import product
from shutil import copytree

from tabulate import tabulate
from tqdm import tqdm

from fxlogger import get_logger


class SetupController:
    def __init__(self):
        self.logger = get_logger()

        self.experiment_base_id = datetime.today().strftime('%Y%m%d%H%M%S')
        self.basecmd = ['python', 'train.py']
        self.dataset_name = 'poltsanews'

        self.args_names_ordered = ['snem', 'model_name', 'optimizer', 'initializer', 'learning_rate', 'batch_size']
        # keys in the dict must match parameter names accepted by train.py. values must match accepted values for such
        # parameters in train.py
        self.combinations = {
            'model_name': ['aen_bert', 'aen_distilbert'],
            'snem': ['recall_avg'],
            'optimizer': ['adam'],
            'initializer': ['xavier_uniform_'],
            'learning_rate': ['1e-3'],  # , '2e-5', '5e-5'],
            'batch_size': ['16'],  # , '32', '64'],
        }
        assert len(self.args_names_ordered) == len(self.combinations.keys())
        assert len(self.combinations['snem']) == 1

        self.combination_count = 1
        _combination_values = []
        for arg_name in self.args_names_ordered:
            arg_values = list(self.combinations[arg_name])
            self.combination_count = self.combination_count * len(arg_values)
            _combination_values.append(arg_values)

        self.combinations = list(product(*_combination_values))
        assert len(self.combinations) == self.combination_count

        self.logger.info(
            "{} arguments, totaling in {} combinations".format(len(self.args_names_ordered), self.combination_count))
        self.logger.info("combinations:")
        self.logger.info("{}".format(self.combinations))

    def _build_args(self, args_combination):
        args_list = []
        for arg_index, arg_name in enumerate(self.args_names_ordered):
            args_list = self._add_arg(args_list, arg_name, args_combination[arg_index])
        return args_list

    def _add_arg(self, args_list, name, value):
        args_list.append("--" + name)
        args_list.append(value)
        return args_list

    def _prepare_experiment_env(self, experiment_path):
        os.makedirs(experiment_path)

        # copy datasets in there
        copytree(os.path.join('datasets', self.dataset_name),
                 os.path.join(experiment_path, 'datasets', self.dataset_name))

    def _args_combination_to_single_arg_values(self, args_combination):
        args_names_values = {}
        for arg_index, arg_name in enumerate(self.args_names_ordered):
            args_names_values[arg_name] = args_combination[arg_index]
        return args_names_values

    def execute_single_setup(self, args_combination):
        args_names_values = self._args_combination_to_single_arg_values(args_combination)
        experiment_id = "_".join(args_combination)
        experiment_path = "./experiments/{}/{}/".format(self.experiment_base_id, experiment_id)

        self._prepare_experiment_env(experiment_path)

        args = self._build_args(args_combination)
        args = self._add_arg(args, 'dataset_name', 'poltsanews')
        args = self._add_arg(args, 'experiment_path', experiment_path)

        cmd = self.basecmd + args

        with open(os.path.join(experiment_path, 'stdlog.out'), "w") as file_stdout, open(
                os.path.join(experiment_path, 'stdlog.err'), "w") as file_stderr:
            completed_process = subprocess.run(cmd, stdout=file_stdout, stderr=file_stderr)

        snem = self.get_experiment_result_detailed(experiment_path)

        return {**args_names_values,
                **{'rc': completed_process.returncode, 'experiment_id': experiment_id, 'snem': snem}}

    def get_experiment_result_detailed(self, experiment_path):
        # each experiments reports detailed information about its performance to the stdout, immediately before it exits
        # thus, get the last line of the stdlog.out
        lines = [line.rstrip('\n') for line in open(os.path.join(experiment_path, 'stdlog.out'))]

        snem_line = lines[-1]

        return snem_line

    def run(self):
        with tqdm(total=self.combination_count) as pbar:

            results = []
            for combination in self.combinations:
                result = self.execute_single_setup(combination)
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
