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
import argparse
import multiprocessing
import os
import shelve
import subprocess
import time
from collections import Counter
from datetime import datetime
from itertools import product

from jsonlines import jsonlines
from tabulate import tabulate
from tqdm import tqdm

from DatasetPreparer import DatasetPreparer
from combinations_absadata_0 import combinations_absadata_0
from fxlogger import get_logger

completed_tasks = None  # will be shelve (dict) later
completed_tasks_in_this_run_count = 0


def start_worker(experiment_id, experiment_named_id, named_combination, cmd, human_cmd, experiment_path):
    logger = get_logger()

    logger.debug("starting single setup: {}".format(human_cmd))
    with open(os.path.join(experiment_path, 'stdlog.out'), "w") as file_stdout, open(
            os.path.join(experiment_path, 'stdlog.err'), "w") as file_stderr:
        completed_process = subprocess.run(cmd, stdout=file_stdout, stderr=file_stderr)

    experiment_details = get_experiment_result_detailed(experiment_path)

    return {**named_combination,
            **{'rc': completed_process.returncode, 'experiment_id': experiment_id},
            'details': experiment_details, 'experiment_named_id': experiment_named_id}


def on_task_done(x):
    # result_list is modified only by the main process, not the pool workers.
    completed_tasks[x['experiment_named_id']] = x
    completed_tasks.sync()
    global completed_tasks_in_this_run_count
    completed_tasks_in_this_run_count += 1


def on_task_error(x):
    # result_list is modified only by the main process, not the pool workers.
    print(x)
    completed_tasks[x['experiment_named_id']] = x
    completed_tasks.sync()
    global completed_tasks_in_this_run_count
    completed_tasks_in_this_run_count += 1


def get_experiment_result_detailed(experiment_path):
    experiment_results_path = os.path.join(experiment_path, 'experiment_results.jsonl')
    try:
        with jsonlines.open(experiment_results_path, 'r') as reader:
            lines = []
            for line in reader:
                lines.append(line)
            assert len(lines) == 1
        return lines[0]
    except FileNotFoundError:
        return None


class SetupController:
    def __init__(self, options):
        self.logger = get_logger()
        self.opt = options

        self.cuda_devices = os.environ.get('SGE_GPU')
        if self.cuda_devices:
            self.logger.info("cuda devices:" + self.cuda_devices)
            self.cuda_devices = self.cuda_devices.split(',')
            self.logger.info(f"was assigned {len(self.cuda_devices)} cuda devices: {self.cuda_devices}")
            if self.opt.num_workers < 0:
                self.logger.info("num_workers < 0: using cuda device count")
                self.opt.num_workers = len(self.cuda_devices)
        else:
            self.logger.warning("env not given: SGE_GPU")

        self.use_cross_validation = 0  # if 0: do not use cross validation
        self.snem = 'recall_avg'
        self.experiment_base_path = self.opt.experiments_path

        args_names_ordered = ['model_name', 'optimizer', 'initializer', 'learning_rate', 'batch_size',
                              'balancing', 'num_epoch', 'lsr', 'use_tp_placeholders',
                              'spc_lm_representation', 'spc_input_order', 'aen_lm_representation',
                              'spc_lm_representation_distilbert', 'finetune_glove',
                              'eval_only_after_last_epoch', 'devmode', 'local_context_focus', 'SRD',
                              'pretrained_model_name']
        combinations = combinations_absadata_0
        # key: name of parameter that is only applied if its conditions are met
        # pad_value: list of tuples, consisting of parameter name and the pad_value it needs to have in order for the
        # condition to be satisfied
        # Note that all tuples in this list are OR connected, so if at least one is satisfied, the conditions are met.
        # If we need AND connected conditions, my idea is to add an outer list, resulting in a list of lists (of
        # tuples) where all lists are AND connected.
        # If a condition is not satisfied, the corresponding parameter will still be pass
        conditions = {
            'spc_lm_representation_distilbert':
                [('model_name', 'distilbert')],
            'spc_lm_representation':
                [('model_name', 'spc_bert'), ('model_name', 'spc_roberta')],
            'spc_input_order':
                [('model_name', 'spc_bert'), ('model_name', 'spc_roberta'), ('model_name', 'spc_distilbert')],
            'aen_lm_representation':
                [('model_name', 'aen_bert'), ('model_name', 'aen_roberta'), ('model_name', 'aen_distilbert')],
            'use_early_stopping':
                [('num_epoch', '10')],
            'finetune_glove':
                [('model_name', 'aen_glove')],
            'local_context_focus':
                [('model_name', 'lcf_bert')],
            'SRD':
                [('model_name', 'lcf_bert')],
            'pretrained_model_name':
                [('model_name', 'lcf_bert'), ('model_name', 'aen_bert'), ('model_name', 'spc_bert')],
        }

        assert len(args_names_ordered) == len(combinations.keys())

        self.experiment_base_id = self.opt.dataset + '_' + datetime.today().strftime('%Y%m%d-%H%M%S')
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
        self.logger.info("applying conditions...")
        self.named_combinations, count_duplicates = self._apply_conditions(combinations, args_names_ordered, conditions)
        self.logger.info("applied conditions. removed {} combinations. {} -> {}".format(count_duplicates,
                                                                                        combination_count,
                                                                                        len(self.named_combinations)))
        self.combination_count = len(self.named_combinations)

        if self.use_cross_validation > 0:
            self.logger.info("using {}-fold cross validation".format(self.use_cross_validation))
            self.dataset_preparer = DatasetPreparer.poltsanews_crossval8010_allhuman(self.basepath_data)
        else:
            self.logger.info("not using cross validation".format(self.use_cross_validation))
            if self.opt.dataset == 'poltsanews_rel801010_allhuman':
                self.dataset_preparer, self.datasetname, self.absa_task_format = DatasetPreparer.poltsanews_rel801010_allhuman(
                    self.basepath_data)
            elif self.opt.dataset == 'semeval14restaurants':
                self.dataset_preparer, self.datasetname, self.absa_task_format = DatasetPreparer.semeval14restaurants(
                    self.basepath_data)
            elif self.opt.dataset == 'semeval14laptops':
                self.dataset_preparer, self.datasetname, self.absa_task_format = DatasetPreparer.semeval14laptops(
                    self.basepath_data)
            elif self.opt.dataset == 'acl14twitter':
                self.dataset_preparer, self.datasetname, self.absa_task_format = DatasetPreparer.acl14twitter(
                    self.basepath_data)
            elif self.opt.dataset == 'sentinews':
                self.dataset_preparer, self.datasetname, self.absa_task_format = DatasetPreparer.sentinews(
                    self.basepath_data)
            else:
                raise Exception("unknown dataset: {}".format(self.opt.dataset))

    def _apply_conditions(self, combinations, args_names_ordered, conditions):
        named_combinations = []
        seen_experiment_ids = set()
        count_duplicates = 0

        with tqdm(total=len(combinations)) as pbar:
            for combination in combinations:
                named_combination = {}
                full_named_combination = self._args_combination_to_single_arg_values(combination, args_names_ordered)

                # for a parameter combination, pass only those parameters that are valid for that combination
                for arg_index, arg_name in enumerate(args_names_ordered):
                    # iterate each parameter and validate - using the other parameter names and values - whether its
                    # conditions are met
                    if self._check_conditions(arg_name, full_named_combination, conditions):
                        # if yes, pass it
                        named_combination[arg_name] = combination[arg_index]
                        self.logger.debug("using '{}' in combination {}".format(arg_name, combination))
                    else:
                        self.logger.debug("not using '{}' in combination {}".format(arg_name, combination))

                # check if experiment_id of named combination was already seen
                experiment_id = self._experiment_named_id_from_named_combination(named_combination)
                if experiment_id not in seen_experiment_ids:
                    seen_experiment_ids.add(experiment_id)
                    named_combinations.append(named_combination)
                else:
                    count_duplicates += 1

                pbar.update(1)

        return named_combinations, count_duplicates

    def _check_conditions(self, arg_name, full_named_combination, conditions):
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

                # get parameter and its pad_value in current combination
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

    def _experiment_named_id_from_named_combination(self, named_combination):
        return "__".join(["{}={}".format(k, v) for (k, v) in named_combination.items()])

    def prepare_single_setup(self, named_combination, experiment_id):
        experiment_path = "{}/{}/".format(self.experiment_base_id, experiment_id)
        experiment_path = os.path.join(self.experiment_base_path, experiment_path)

        self._prepare_experiment_env(experiment_path)

        args = self._build_args(named_combination)
        self._add_arg(args, 'snem', self.snem)
        self._add_arg(args, 'dataset_name', self.datasetname)
        self._add_arg(args, 'experiment_path', experiment_path)
        self._add_arg(args, 'crossval', self.use_cross_validation)
        self._add_arg(args, 'absa_task_format', self.absa_task_format)

        if self.cuda_devices:
            cuda_device = experiment_id % len(self.cuda_devices)
            cuda_device_name = 'cuda:' + str(cuda_device)
            self._add_arg(args, 'device', cuda_device_name)

        cmd = self.basecmd + args
        human_cmd = " ".join(cmd)

        with open(os.path.join(experiment_path, 'experiment_cmd.sh'), 'w') as writer:
            writer.write(human_cmd)

        return cmd, human_cmd, experiment_path

    def run(self):
        global completed_tasks

        if not self.opt.results_path:
            results_path = "results_{}".format(self.datasetname)
        else:
            results_path = self.opt.results_path

        if not self.opt.continue_run:
            self.logger.info("not continuing")
            os.remove(results_path)
        else:
            self.logger.info("continuing previous run(s)")

        completed_tasks = shelve.open(results_path)
        self.logger.info("found {} previous results".format(len(completed_tasks)))

        self.logger.info("preparing experiment setups...")
        experiment_descs = []
        previous_tasks = Counter()
        with tqdm(total=self.combination_count) as pbar:
            for i, named_combination in enumerate(self.named_combinations):
                _experiment_named_id = self._experiment_named_id_from_named_combination(named_combination)
                if _experiment_named_id in completed_tasks:
                    task_desc = completed_tasks[_experiment_named_id]

                    if self.opt.rerun_non_rc0 and task_desc['rc'] != 0:
                        self.logger.debug(
                            "task {} was already executed, but with rc={}. rerunning.".format(_experiment_named_id,
                                                                                              task_desc['rc']))
                        cmd, human_cmd, experiment_path = self.prepare_single_setup(named_combination, i)
                        experiment_descs.append((i, _experiment_named_id, named_combination, cmd, human_cmd,
                                                 experiment_path))
                        previous_tasks['rcnon0'] += 1
                        del completed_tasks[_experiment_named_id]
                    else:
                        # rerun tasks where the rc != 0 (always rerun tasks that have not been executed at all, yet)
                        self.logger.debug("skipping experiment: {}".format(_experiment_named_id))
                        self.logger.debug("previous result: {}".format(completed_tasks[_experiment_named_id]))
                        previous_tasks['rc0'] += 1
                else:
                    cmd, human_cmd, experiment_path = self.prepare_single_setup(named_combination, i)
                    experiment_descs.append((i, _experiment_named_id, named_combination, cmd, human_cmd,
                                             experiment_path))
                    previous_tasks['new'] += 1
                pbar.update(1)

        self.logger.info("summary (rc0 is increased also for non-0 tasks, if rerun_non_rc0 is not set)")
        self.logger.info("{}".format(previous_tasks))

        self.logger.info("starting {} experiments".format(self.combination_count))
        self.logger.info("creating process pool with {} workers".format(self.opt.num_workers))

        pool = multiprocessing.Pool(processes=self.opt.num_workers)
        for desc in experiment_descs:
            pool.apply_async(start_worker, desc, callback=on_task_done, error_callback=on_task_error)

        self.logger.info("waiting for workers to complete all jobs...")
        prev_count_done = 0
        with tqdm(total=previous_tasks['new'], initial=prev_count_done) as pbar:
            while True:
                time.sleep(10)
                if completed_tasks_in_this_run_count >= previous_tasks['new']:
                    self.logger.info(
                        f"finished all tasks ({completed_tasks_in_this_run_count} of {previous_tasks['new']})")
                    break

                update_inc = completed_tasks_in_this_run_count - prev_count_done
                if update_inc > 0:
                    pbar.update(update_inc)
                    prev_count_done = completed_tasks_in_this_run_count

                    best_dev_snem = self._get_best_dev_snem()
                    pbar.set_postfix_str(f"dev-snem: {best_dev_snem:.4f}")

        processed_results = dict(completed_tasks)
        completed_tasks.close()

        experiments_rc_overview = Counter()
        non_okay_experiment_ids = []
        for experiment_named_id, experiment_result in processed_results.items():
            rc = experiment_result['rc']
            experiments_rc_overview[rc] += 1

            if rc != 0:
                non_okay_experiment_ids.append(experiment_result['experiment_id'])

        if non_okay_experiment_ids:
            self.logger.warning(
                f"{len(non_okay_experiment_ids)} experiments did not return 0: {non_okay_experiment_ids}")

        # snem-based performance sort
        sorted_results = list(dict(processed_results).values())
        for result in sorted_results:
            if result['details']:
                result['dev_snem'] = result['details']['dev_stats'][self.snem]
                del result['details']
            else:
                result['dev_snem'] = -1.0
        sorted_results.sort(key=lambda x: x['dev_snem'], reverse=True)
        headers = list(sorted_results[0].keys())
        rows = [x.values() for x in sorted_results]

        self.logger.info("all experiments finished. statistics:")
        self.logger.debug("snem-based performances:")
        self.logger.debug("\n" + tabulate(rows, headers))
        self.logger.info("return codes: {}".format(experiments_rc_overview))

    def _get_best_dev_snem(self):
        best_dev_snem = -1.0
        for task in completed_tasks.values():
            if task.get('details') and task['details'].get('dev_stats'):
                best_dev_snem = max(best_dev_snem, task['details']['dev_stats'][self.snem])
        return best_dev_snem


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean pad_value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--experiments_path', default='./experiments', type=str)
    parser.add_argument("--continue_run", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--rerun_non_rc0", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--results_path', type=str, default=None)
    opt = parser.parse_args()

    SetupController(opt).run()
