# main entry point to start training of a single model for target-dependent sentiment analysis in news

# author: Felix Hamborg <felix.hamborg@uni-konstanz.de>
# This file is based on https://github.com/songyouwei/ABSA-PyTorch/blob/master/train.py
# original author: songyouwei <youwei0314@gmail.com>

import argparse
import math
import os
import random
import sys
import time

import numpy
import torch
import torch.nn as nn
from jsonlines import jsonlines

cur_path = os.path.dirname(os.path.realpath(__file__))
par_path = os.path.dirname(cur_path)
sys.path.append(cur_path)
sys.path.append(par_path)

from newstsc.crossentropylosslsr import CrossEntropyLoss_LSR
from newstsc.dataset import FXDataset, RandomOversampler
from newstsc.earlystopping import EarlyStopping
from newstsc.evaluator import Evaluator
from newstsc.fxlogger import get_logger
from newstsc.models.aen import AEN_Base
from newstsc.models.lcf import LCF_BERT
from newstsc.models.ram import RAM
from newstsc.models.spc import SPC_Base
from newstsc.models.globalsenti import Global_LCF
from newstsc.plotter_utils import create_save_plotted_confusion_matrix
from newstsc.tokenizers import Tokenizer4Bert, Tokenizer4Distilbert, Tokenizer4GloVe, Tokenizer4Roberta
from pytorch_transformers import BertModel, DistilBertModel, RobertaModel, PreTrainedModel
from torch.utils.data import DataLoader, random_split, ConcatDataset

logger = get_logger()


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        self.create_model()
        logger.info("initialized pretrained model: {}".format(opt.model_name))

        self.polarity_associations = {'positive': 2, 'neutral': 1, 'negative': 0}
        self.polarity_associations_inv = {2: 'positive', 1: 'neutral', 0: 'negative'}
        self.sorted_expected_label_values = [0, 1, 2]
        self.sorted_expected_label_names = ['negative', 'neutral', 'positive']

        self.evaluator = Evaluator(self.sorted_expected_label_values, self.polarity_associations, self.opt.snem)

        if self.opt.training_mode:
            self.load_datasets()

        self._print_args()

    def load_datasets(self):
        if self.opt.crossval > 0:
            logger.info("loading datasets {} from {}".format(self.opt.dataset_name, self.opt.dataset_path))
            self.crossvalset = FXDataset(self.opt.dataset_path + 'crossval.jsonl', self.tokenizer,
                                         self.polarity_associations, self.sorted_expected_label_names,
                                         self.opt.use_tp_placeholders, self.opt.task_format, self.opt.devmode,
                                         self.opt.use_global_context)
            self.testset = FXDataset(self.opt.dataset_path + 'test.jsonl', self.tokenizer, self.polarity_associations,
                                     self.sorted_expected_label_names, self.opt.use_tp_placeholders,
                                     self.opt.task_format, self.opt.devmode, self.opt.use_global_context)
            self.all_datasets = [self.crossvalset, self.testset]
            logger.info("loaded crossval datasets from {}".format(self.opt.dataset_path))
        else:
            logger.info("loading datasets {} from {}".format(self.opt.dataset_name, self.opt.dataset_path))
            self.trainset = FXDataset(self.opt.dataset_path + 'train.jsonl', self.tokenizer, self.polarity_associations,
                                      self.sorted_expected_label_names, self.opt.use_tp_placeholders,
                                      self.opt.task_format, self.opt.devmode, self.opt.use_global_context)
            self.devset = FXDataset(self.opt.dataset_path + 'dev.jsonl', self.tokenizer, self.polarity_associations,
                                    self.sorted_expected_label_names, self.opt.use_tp_placeholders,
                                    self.opt.task_format, self.opt.devmode, self.opt.use_global_context)
            self.testset = FXDataset(self.opt.dataset_path + 'test.jsonl', self.tokenizer, self.polarity_associations,
                                     self.sorted_expected_label_names, self.opt.use_tp_placeholders,
                                     self.opt.task_format, self.opt.devmode, self.opt.use_global_context)
            self.all_datasets = [self.trainset, self.devset, self.testset]
            logger.info("loaded datasets from {}".format(self.opt.dataset_path))

        logger.info("truncated sequences of in total: {} / {}".format(self.tokenizer.count_truncated,
                                                                      self.tokenizer.count_all_sequences_where_we_count_truncation))
        logger.info("truncated long docs: {}".format(self.tokenizer.count_truncated_long_docs))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def create_model(self, only_model=False):
        logger.info("creating model {}".format(self.opt.model_name))

        if self.opt.model_name in ['aen_bert', 'aen_distilbert', 'aen_roberta', 'aen_distilroberta', 'spc_distilbert',
                                   'spc_bert', 'spc_roberta', 'lcf_bert']:
            if not only_model:
                if self.opt.model_name in ['aen_bert', 'spc_bert', 'lcf_bert']:
                    self.tokenizer = Tokenizer4Bert(self.opt.pretrained_model_name, self.opt.max_seq_len,
                                                    self.opt.global_context_seqs_per_doc)
                elif self.opt.model_name in ['aen_distilbert', 'spc_distilbert']:
                    self.tokenizer = Tokenizer4Distilbert(self.opt.pretrained_model_name, self.opt.max_seq_len, )
                elif self.opt.model_name in ['aen_roberta', 'spc_roberta']:
                    self.tokenizer = Tokenizer4Roberta(self.opt.pretrained_model_name, self.opt.max_seq_len, )
                elif self.opt.model_name in ['aen_distilroberta', 'spc_distiloberta']:
                    self.tokenizer = Tokenizer4Roberta(self.opt.pretrained_model_name, self.opt.max_seq_len, )

            if self.opt.model_name in ['aen_bert', 'spc_bert', 'lcf_bert']:
                pretrained_model = BertModel.from_pretrained(self.opt.pretrained_model_name, output_hidden_states=True)
            elif self.opt.model_name in ['aen_distilbert', 'spc_distilbert']:
                pretrained_model = DistilBertModel.from_pretrained(self.opt.pretrained_model_name,
                                                                   output_hidden_states=True)
            elif self.opt.model_name in ['aen_roberta', 'spc_roberta']:
                pretrained_model = RobertaModel.from_pretrained(self.opt.pretrained_model_name,
                                                                output_hidden_states=True)

            self.model = self.opt.model_class(pretrained_model, self.opt).to(self.opt.device)

            if self.opt.state_dict:
                # load weights from the state_dict
                logger.info(f"loading weights from {self.opt.state_dict}")
                self.model.load_state_dict(torch.load(self.opt.state_dict, map_location=self.opt.device))

        elif self.opt.model_name in ['aen_glove', 'ram']:
            if not only_model:
                self.tokenizer = Tokenizer4GloVe(self.opt.max_seq_len)

            if self.opt.model_name == 'aen_glove':
                self.model = self.opt.model_class(self.tokenizer.embedding_matrix, self.opt).to(
                    self.opt.device)
            elif self.opt.model_name == 'ram':
                self.model = self.opt.model_class(self.opt).to(self.opt.device)

        else:
            raise Exception("model_name unknown: {}".format(self.opt.model_name))

    def _reset_params(self):
        for child in self.model.children():
            if not issubclass(child.__class__, PreTrainedModel):  # if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _create_prepare_model_path(self, snem, epoch, fold_number=None):
        selected_model_filename = '{0}_{1}_val_{2}_{3}_epoch{4}'.format(self.opt.model_name, self.opt.dataset_name,
                                                                        self.opt.snem, round(snem, 4), epoch)
        if fold_number is not None:
            selected_model_filename += '_cvf' + str(fold_number)

        pathdir = os.path.join(self.opt.experiment_path, 'state_dict')

        os.makedirs(pathdir, exist_ok=True)
        selected_model_path = os.path.join(pathdir, selected_model_filename)

        return selected_model_filename, selected_model_path

    def _select_inputs(self, sample_batched):
        """
        Selects the input data fields, thereby handles other options that influence data selection, too, e.g.,
        use_global_context.
        """
        inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.input_columns]

        if self.opt.use_global_context:
            str_gci = 'global_context_ids{}'
            str_gct = 'global_context_type_ids{}'
            str_gca = 'global_context_attention_mask{}'
            for i in range(self.opt.global_context_seqs_per_doc):
                gci = sample_batched[str_gci.format(i)]
                gct = sample_batched[str_gct.format(i)]
                gca = sample_batched[str_gca.format(i)]
                inputs.append(gci.to(self.opt.device))
                inputs.append(gct.to(self.opt.device))
                inputs.append(gca.to(self.opt.device))

        return inputs

    def _train(self, criterion, optimizer, train_data_loader, dev_data_loader, fold_number=None):
        global_step = 0
        selected_model_path = None
        selected_model_filename = None
        selected_model_dev_stats = None

        # initialize the early_stopping object
        early_stopping = EarlyStopping()

        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {} (num_epoch: {})'.format(epoch, self.opt.num_epoch))
            n_correct, n_total, loss_total = 0, 0, 0

            # switch model to training mode
            self.model.train()

            # train on batches
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()
                inputs = self._select_inputs(sample_batched)
                targets = sample_batched['polarity'].to(self.opt.device)

                outputs = self.model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            dev_stats = self._evaluate(dev_data_loader)
            self.evaluator.print_stats(dev_stats, "dev during training")

            dev_snem = dev_stats[self.opt.snem]

            early_stopping(dev_snem)
            if self.opt.eval_only_after_last_epoch:
                if epoch >= self.opt.num_epoch - 1:
                    # return the model that was trained through all epochs as the selected model
                    logger.info("all epochs finished, saving model to disk...")

                    selected_model_dev_stats = dev_stats

                    selected_model_filename, selected_model_path = self._create_prepare_model_path(dev_snem, epoch,
                                                                                                   fold_number)
                    torch.save(self.model.state_dict(), selected_model_path)
                    logger.info('>> saved: {}'.format(selected_model_path))

                    # save confusion matrices
                    filepath_stats_base = os.path.join(self.opt.experiment_path, 'statistics', selected_model_filename)
                    if not filepath_stats_base.endswith('/'):
                        filepath_stats_base += '/'
                    os.makedirs(filepath_stats_base, exist_ok=True)
                    create_save_plotted_confusion_matrix(dev_stats['confusion_matrix'],
                                                         expected_labels=self.sorted_expected_label_values,
                                                         basepath=filepath_stats_base)
                    logger.debug("created confusion matrices in path: {}".format(filepath_stats_base))
            else:
                # return the best model during any epoch
                if early_stopping.flag_has_score_increased_since_last_check:
                    logger.info("model yields best performance so far, saving to disk...")

                    selected_model_dev_stats = dev_stats

                    selected_model_filename, selected_model_path = self._create_prepare_model_path(dev_snem, epoch,
                                                                                                   fold_number)

                    torch.save(self.model.state_dict(), selected_model_path)
                    logger.info('>> saved: {}'.format(selected_model_path))

                    # save confusion matrices
                    filepath_stats_base = os.path.join(self.opt.experiment_path, 'statistics', selected_model_filename)
                    if not filepath_stats_base.endswith('/'):
                        filepath_stats_base += '/'
                    os.makedirs(filepath_stats_base, exist_ok=True)
                    create_save_plotted_confusion_matrix(dev_stats['confusion_matrix'],
                                                         expected_labels=self.sorted_expected_label_values,
                                                         basepath=filepath_stats_base)
                    logger.debug("created confusion matrices in path: {}".format(filepath_stats_base))

            if early_stopping.early_stop and self.opt.use_early_stopping:
                logger.info("early stopping after {} epochs without improvement, total epochs: {} of {}".format(
                    early_stopping.patience, epoch, self.opt.num_epoch))
                break

        return selected_model_path, selected_model_filename, selected_model_dev_stats

    def _evaluate(self, data_loader, get_examples=False, basepath=None):
        t_labels_all, t_outputs_all = None, None
        t_texts_all, t_targets_all = [], []

        # switch model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = self._select_inputs(t_sample_batched)
                t_labels = t_sample_batched['polarity'].to(self.opt.device)

                t_texts = t_sample_batched['orig_text']
                t_targets = t_sample_batched['orig_target']

                t_outputs = self.model(t_inputs)

                if t_labels_all is None:
                    t_labels_all = t_labels
                    t_outputs_all = t_outputs
                else:
                    t_labels_all = torch.cat((t_labels_all, t_labels), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

                t_texts_all.extend(t_texts)
                t_targets_all.extend(t_targets)

        # softmax: get predictions from outputs
        y_pred = torch.argmax(t_outputs_all, -1).cpu()
        y_true = t_labels_all.cpu()

        stats = self.evaluator.calc_statistics(y_true, y_pred)

        if get_examples:
            self.evaluator.write_error_table(y_true, y_pred, t_texts_all, t_targets_all, basepath + 'errortable.jsonl')

        return stats

    def get_normalized_inv_class_frequencies(self):
        inv_freqs = []
        for label_name in self.sorted_expected_label_names:
            inv_freq_of_class = 1.0 / self.testset.label_counter[label_name]
            inv_freqs.append(inv_freq_of_class)

        sum_of_inv_freqs = sum(inv_freqs)
        for i in range(len(inv_freqs)):
            inv_freqs[i] = inv_freqs[i] / sum_of_inv_freqs

        return inv_freqs

    def run_crossval(self):
        raise Exception("run_crossval needs to get updated as to saving its experiment results, and because the new "
                        "reset_params does not - as before - load the model newly, but only resets params as in ABSA. "
                        "also lossweighting has been adapted")

        # Loss and Optimizer
        if self.opt.lossweighting:
            inv_class_freqs = self.get_normalized_inv_class_frequencies()
            logger.info("weighting losses of classes: {}".format(inv_class_freqs))
            class_weights = torch.tensor(inv_class_freqs).to(self.opt.device)
        else:
            class_weights = None

        if self.opt.lsr:
            criterion = CrossEntropyLoss_LSR(self.opt.device, para_LSR=0.2, weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        valset_len = len(self.crossvalset) // self.opt.crossval
        splitedsets = random_split(self.crossvalset, tuple([valset_len] * (self.opt.crossval - 1) + [
            len(self.crossvalset) - valset_len * (self.opt.crossval - 1)]))

        logger.info("starting training...")
        all_test_stats = []
        for fid in range(self.opt.crossval):
            logger.info('>' * 100)
            logger.info('fold : {}'.format(fid))

            trainset = ConcatDataset([x for i, x in enumerate(splitedsets) if i != fid])
            valset = splitedsets[fid]
            train_data_loader = DataLoader(dataset=trainset, batch_size=self.opt.batch_size, shuffle=True)
            val_data_loader = DataLoader(dataset=valset, batch_size=self.opt.batch_size, shuffle=False)

            self._reset_params()
            best_model_path, best_model_filename = self._train(criterion, optimizer,
                                                               train_data_loader,
                                                               val_data_loader,
                                                               fid)

            # evaluate the model that performed best during training,
            self.model.load_state_dict(torch.load(best_model_path))
            test_stats = self._evaluate(test_data_loader)
            # append its results to the list of results, which will be aggregated after all folds are completed
            all_test_stats.append(test_stats)

            self.evaluator.print_stats(test_stats, "evaluation of fold {} on test-set".format(fid))

        logger.info("aggregating performance statistics from all {} folds".format(self.opt.crossval))
        mean_test_stats = self.evaluator.mean_from_all_statistics(all_test_stats)
        self.evaluator.print_stats(mean_test_stats,
                                   "aggregated evaluation results of all folds on test-set".format(fid))

        logger.info("finished execution of this crossval run. exiting.")

        # print snem pad_value to stdout, for the controller to parse it
        print(mean_test_stats[self.opt.snem])

    def run(self):
        # balancing modes
        class_weights = None
        sampler_train = None
        if self.opt.balancing == 'lossweighting':
            inv_class_freqs = self.get_normalized_inv_class_frequencies()
            logger.info("weighting losses of classes: {}".format(inv_class_freqs))
            class_weights = torch.tensor(inv_class_freqs).to(self.opt.device)
        elif self.opt.balancing == 'oversampling':
            sampler_train = RandomOversampler(self.trainset, self.opt.seed)

        # loss
        if self.opt.lsr:
            criterion = CrossEntropyLoss_LSR(self.opt.device, para_LSR=0.2, weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        # optimizer
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        # data loaders
        if sampler_train:
            train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, sampler=sampler_train)
        else:
            train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        dev_data_loader = DataLoader(dataset=self.devset, batch_size=self.opt.batch_size, shuffle=False)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)

        # start training
        self._reset_params()
        logger.info("starting training...")
        time_training_start = time.time()
        best_model_path, best_model_filename, selected_model_dev_stats = self._train(criterion, optimizer,
                                                                                     train_data_loader, dev_data_loader)
        time_training_elapsed_mins = (time.time() - time_training_start) // 60
        logger.info("training finished. duration={}mins".format(time_training_elapsed_mins))

        self.post_training(best_model_path, best_model_filename, test_data_loader, selected_model_dev_stats,
                           time_training_elapsed_mins)

    def get_serializable_stats(self, stats):
        sstats = stats.copy()
        sstats['recalls_of_classes'] = stats['recalls_of_classes'].tolist()
        sstats['confusion_matrix'] = stats['confusion_matrix'].tolist()
        return sstats

    def get_serializable_opts(self):
        opts = vars(self.opt)
        sopts = opts.copy()
        del sopts['optimizer']
        del sopts['initializer']
        del sopts['device']
        del sopts['model_class']
        return sopts

    def post_training(self, selected_model_path, selected_model_filename, test_data_loader, selected_model_dev_stats,
                      time_training_elapsed_mins):
        logger.info("loading selected model from training: {}".format(selected_model_path))
        self.model.load_state_dict(torch.load(selected_model_path))

        logger.info("evaluating selected model on test-set")
        # set model into evaluation mode (cf. https://pytorch.org/docs/stable/nn.html#torch.nn.Module.train)
        self.model.eval()

        # do the actual evaluation
        filepath_stats_prefix = os.path.join(self.opt.experiment_path, 'statistics', selected_model_filename)
        os.makedirs(filepath_stats_prefix, exist_ok=True)
        if not filepath_stats_prefix.endswith('/'):
            filepath_stats_prefix += '/'

        test_stats = self._evaluate(test_data_loader, get_examples=True, basepath=filepath_stats_prefix)
        test_snem = test_stats[self.opt.snem]

        self.evaluator.print_stats(test_stats, "evaluation on test-set")

        # save dev and test results
        experiment_results = {}
        experiment_results['test_stats'] = self.get_serializable_stats(test_stats)
        experiment_results['dev_stats'] = self.get_serializable_stats(selected_model_dev_stats)
        experiment_results['options'] = self.get_serializable_opts()
        experiment_results['time_training_elapsed_mins'] = time_training_elapsed_mins

        experiment_results_path = os.path.join(self.opt.experiment_path, 'experiment_results.jsonl')
        with jsonlines.open(experiment_results_path, 'w') as writer:
            writer.write(experiment_results)

        # save confusion matrices
        test_confusion_matrix = test_stats['confusion_matrix']

        create_save_plotted_confusion_matrix(test_confusion_matrix,
                                             expected_labels=self.sorted_expected_label_values,
                                             basepath=filepath_stats_prefix)

        logger.info("finished execution of this run. exiting.")

        # print snem pad_value to stdout, for the controller to parse it
        print(test_snem)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean pad_value expected.')


def prepare_and_start_instructur(opt):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    opt.base_path = dir_path

    if opt.eval_only_after_last_epoch:
        assert not opt.use_early_stopping

    if not opt.balancing or opt.balancing == 'None':
        opt.balancing = None

    if opt.spc_lm_representation_distilbert:
        logger.info("spc_lm_representation_distilbert defined, overwriting spc_lm_representation")
        opt.spc_lm_representation = opt.spc_lm_representation_distilbert

    if opt.seed is not None:
        logger.info("setting random seed: {}".format(opt.seed))
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'global_lcf': Global_LCF,
        'ram': RAM,
        # AEN
        'aen_bert': AEN_Base,
        'aen_glove': AEN_Base,
        'aen_roberta': AEN_Base,
        'aen_distilbert': AEN_Base,
        'aen_distilroberta': AEN_Base,
        # SPC
        'spc_bert': SPC_Base,
        'spc_distilbert': SPC_Base,
        'spc_roberta': SPC_Base,
        # LCF
        'lcf_bert': LCF_BERT,
    }
    model_name_to_pretrained_model_name = {
        'global_lcf': 'bert-base-uncased',
        # bert
        'aen_bert': 'bert-base-uncased',
        'spc_bert': 'bert-base-uncased',
        'lcf_bert': 'bert-base-uncased',
        # distilbert
        'aen_distilbert': 'distilbert-base-uncased',
        'spc_distilbert': 'distilbert-base-uncased',
        # roberta
        'aen_roberta': 'roberta-base',
        'spc_roberta': 'roberta-base',
        # distilroberta
        'aen_distilroberta': 'distilroberta-base',
    }

    if opt.spc_input_order == 'target_text':
        input_columns_spc = ['special_target_text', 'segments_ids_target_text']
    else:
        input_columns_spc = ['special_text_target', 'segments_ids_text_target']

    input_columns = {
        'global_lcf': ['special_text_target', 'segments_ids_text_target', 'text_raw_with_special_indices',
                       'target_phrase_with_special_indexes'],
        # SPC
        'spc_bert': input_columns_spc,
        'spc_distilbert': input_columns_spc,
        'spc_roberta': input_columns_spc,
        # AEN
        'aen_bert': ['text_raw_with_special_indices', 'target_phrase_with_special_indexes'],
        'aen_distilbert': ['text_raw_with_special_indices', 'target_phrase_with_special_indexes'],
        'aen_roberta': ['text_raw_with_special_indices', 'target_phrase_with_special_indexes'],
        'aen_glove': ['text_raw_indices', 'target_phrase_indexes'],
        # LCF
        'lcf_bert': ['special_text_target', 'segments_ids_text_target', 'text_raw_with_special_indices',
                     'target_phrase_with_special_indexes'],
    }
    opt.input_columns = input_columns[opt.model_name]

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]

    # add base_path to model-related paths
    if not opt.pretrained_model_name or opt.pretrained_model_name == 'default':
        if opt.model_name in ['ram', 'aen_glove']:
            pass
        else:
            opt.pretrained_model_name = model_name_to_pretrained_model_name[opt.model_name]
    else:
        opt.pretrained_model_name = os.path.join(opt.base_path, 'pretrained_models', opt.pretrained_model_name)

    if opt.state_dict and opt.state_dict != 'None':
        opt.state_dict = os.path.join(opt.base_path, 'pretrained_models', 'state_dicts', opt.state_dict)
    else:
        opt.state_dict = None

    if not opt.experiment_path:
        opt.experiment_path = '.'
    if not opt.experiment_path.endswith('/'):
        opt.experiment_path = opt.experiment_path + '/'

    if torch.cuda.is_available():
        logger.info("arg: cuda device: {}".format(opt.device))
        if opt.device:
            opt.device = torch.device(opt.device)
        else:
            opt.device = torch.device('cuda:0')
        logger.info("GPGPU enabled. CUDA dev index: {}".format(opt.device.index))
    else:
        opt.device = torch.device('cpu')

    if opt.device.type == 'cuda':
        logger.info('using GPU (cuda memory allocated: {})'.format(torch.cuda.memory_allocated()))
    else:
        logger.info("using CPU (cuda not available)")

    if opt.training_mode:
        if not opt.dataset_path:
            logger.debug("dataset_path not defined, creating from dataset_name...")
            opt.dataset_path = os.path.join('datasets', opt.dataset_name)
            if not opt.dataset_path.endswith('/'):
                opt.dataset_path = opt.dataset_path + '/'
            logger.debug("dataset_path created from dataset_name: {}".format(opt.dataset_path))
        else:
            logger.debug("dataset_path defined: {}".format(opt.dataset_path))

        # set dataset_path to include experiment_path
        opt.dataset_path = os.path.join(opt.experiment_path, opt.dataset_path)

    ins = Instructor(opt)

    if opt.training_mode:
        opt.initializer = initializers[opt.initializer]
        opt.optimizer = optimizers[opt.optimizer]

        if opt.crossval > 0:
            ins.run_crossval()
        else:
            ins.run()
    else:
        return ins


def parse_arguments(override_args=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_mode', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--dataset_name', default=None, type=str,
                        help='name of the sub-folder in \'datasets\' containing files called [train,dev,test].jsonl')
    parser.add_argument('--dataset_path', default=None, type=str,
                        help='relative or absolute path to dataset folder. If defined, will be used instead of dataset_name')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=64, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--max_seq_len', default=150, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g., cuda:0; if None, CPU will be used')
    parser.add_argument('--seed', default=1337, type=int, help='set seed for reproducibility')
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT '
                                                           'model')
    parser.add_argument('--snem', default='recall_avg', help='see evaluator.py for valid options')
    parser.add_argument("--devmode", type=str2bool, nargs='?', const=True, default=False,
                        help="devmode, default off, enable by using True")
    parser.add_argument('--experiment_path', default=None, type=str,
                        help='if defined, all data will be read from / saved to a folder in the experiments folder')
    parser.add_argument('--crossval', default=0, type=int,
                        help='if k>0 k-fold crossval mode is enabled. the tool will merge ')
    parser.add_argument('--balancing', type=str, default=None)
    parser.add_argument("--lsr", type=str2bool, nargs='?', const=True, default=False,
                        help="True: enable label smoothing regularization; False: disable")
    parser.add_argument('--spc_lm_representation', type=str, default='mean_last')
    parser.add_argument('--spc_lm_representation_distilbert', type=str, default=None)
    parser.add_argument('--use_tp_placeholders', type=str2bool, nargs='?', const=True, default=False,
                        help="replace target_phrases with a placeholder. default: off")
    parser.add_argument('--spc_input_order', type=str, default='text_target', help='SPC: order of input; target_text '
                                                                                   'or text_target')
    parser.add_argument('--aen_lm_representation', type=str, default='last')
    parser.add_argument('--use_early_stopping', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--finetune_glove', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--eval_only_after_last_epoch', type=str2bool, nargs='?', const=True, default=False,
                        help="if False, evaluate the best model that was seen during any training epoch. if True, "
                             "evaluate only the model that was trained through all num_epoch epochs.")
    parser.add_argument('--task_format', type=str, default='newstsc')
    parser.add_argument('--pretrained_model_name', type=str, default=None,
                        help='has to be placed in folder pretrained_models')
    parser.add_argument('--state_dict', type=str, default=None)
    parser.add_argument('--use_global_context', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--global_context_seqs_per_doc', type=int, default=20)

    # if own_args == None -> parse_args will use sys.argv
    # if own_args == [] -> parse_args will use this empty list instead
    own_args = None
    if override_args:
        own_args = []

    return parser.parse_args(args=own_args)


if __name__ == '__main__':
    opt = parse_arguments()
    prepare_and_start_instructur(opt)
