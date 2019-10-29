#
# main entry point to start training of a single model for target-dependent sentiment analysis in news
#
# author: Felix Hamborg <felix.hamborg@uni-konstanz.de>
# This file is based on https://github.com/songyouwei/ABSA-PyTorch/blob/master/train.py
# original author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018-2019. All Rights Reserved.

import argparse
import os
import random
import time

import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from transformers import BertModel, DistilBertModel

from data_utils import Tokenizer4Bert, FXDataset, Tokenizer4Distilbert
from earlystopping import EarlyStopping
from evaluator import Evaluator
from fxlogger import get_logger
from models import RAM
from models.aen import AEN_BERT, AEN_DISTILBERT, CrossEntropyLoss_LSR
from models.bert_spc import BERT_SPC
from models.distilbert_spc import DISTILBERT_SPC
from plotter_utils import create_save_plotted_confusion_matrix

logger = get_logger()


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        logger.info(opt)

        self.create_model()

        logger.info("initialized pretrained model: {}".format(opt.model_name))

        self.polarity_associations = {'positive': 2, 'neutral': 1, 'negative': 0}
        self.sorted_expected_label_values = [0, 1, 2]
        self.sorted_expected_label_names = ['negative', 'neutral', 'positive']
        self.evaluator = Evaluator(self.sorted_expected_label_values, self.polarity_associations, self.opt.snem)

        if self.opt.crossval > 0:
            logger.info("loading datasets {} from {}".format(opt.dataset_name, opt.dataset_path))
            self.crossvalset = FXDataset(opt.dataset_path + 'crossval.jsonl', self.tokenizer,
                                         self.polarity_associations, self.opt.devmode)
            self.testset = FXDataset(opt.dataset_path + 'test.jsonl', self.tokenizer, self.polarity_associations,
                                     self.opt.devmode)
            logger.info("loaded datasets from {}".format(opt.dataset_path))
        else:
            logger.info("loading datasets {} from {}".format(opt.dataset_name, opt.dataset_path))
            self.trainset = FXDataset(opt.dataset_path + 'train.jsonl', self.tokenizer, self.polarity_associations,
                                      self.opt.devmode)
            self.devset = FXDataset(opt.dataset_path + 'dev.jsonl', self.tokenizer, self.polarity_associations,
                                    self.opt.devmode)
            self.testset = FXDataset(opt.dataset_path + 'test.jsonl', self.tokenizer, self.polarity_associations,
                                     self.opt.devmode)
            logger.info("loaded datasets from {}".format(opt.dataset_path))

        self._print_args()

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

    def create_model(self):
        logger.info("creating model {}".format(self.opt.model_name))
        if self.opt.model_name in ['aen_bert', 'bert_spc']:
            self.tokenizer = Tokenizer4Bert(self.opt.max_seq_len, self.opt.pretrained_model_name)
            pretrained_model = BertModel.from_pretrained(self.opt.pretrained_model_name)
            self.model = self.opt.model_class(pretrained_model, self.opt).to(self.opt.device)
        elif self.opt.model_name in ['aen_distilbert', 'distilbert_spc']:
            self.tokenizer = Tokenizer4Distilbert(self.opt.max_seq_len, self.opt.pretrained_model_name)
            pretrained_model = DistilBertModel.from_pretrained(self.opt.pretrained_model_name)
            self.model = self.opt.model_class(pretrained_model, self.opt).to(self.opt.device)
        else:
            raise Exception("model_name {} unknown".format(self.opt.model_name))
        self.pretrained_model_state_dict = pretrained_model.state_dict()

    def _reset_params(self):
        self.create_model()
        # for child in self.model.children():
        #     if type(child) != BertModel:  # skip bert params
        #         for p in child.parameters():
        #             if p.requires_grad:
        #                 if len(p.shape) > 1:
        #                     self.opt.initializer(p)
        #                 else:
        #                     stdv = 1. / math.sqrt(p.shape[0])
        #                     torch.nn.init.uniform_(p, a=-stdv, b=stdv)
        #     else:
        #         self.model.bert.load_state_dict(self.pretrained_model_state_dict)

    def _train(self, criterion, optimizer, train_data_loader, dev_data_loader, fold_number=None):
        max_dev_snem = 0
        global_step = 0
        best_model_path = None
        best_model_filename = None

        # initialize the early_stopping object
        early_stopping = EarlyStopping()

        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0

            # switch model to training mode
            self.model.train()

            # train on batches
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
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
            self.evaluator.log_statistics(dev_stats, "dev during training")

            dev_snem = dev_stats[self.opt.snem]

            early_stopping(dev_snem)
            if early_stopping.flag_has_score_increased_since_last_check:
                logger.info("model yields best performance so far, saving to disk...")
                max_dev_snem = dev_snem

                best_model_filename = '{0}_{1}_val_{2}_{3}_epoch{4}'.format(self.opt.model_name, self.opt.dataset_name,
                                                                            self.opt.snem, round(max_dev_snem, 4),
                                                                            epoch)
                if fold_number is not None:
                    best_model_filename += '_cvf' + str(fold_number)

                pathdir = os.path.join(self.opt.experiment_path, 'state_dict')

                os.makedirs(pathdir, exist_ok=True)
                best_model_path = os.path.join(pathdir, best_model_filename)

                torch.save(self.model.state_dict(), best_model_path)
                logger.info('>> saved: {}'.format(best_model_path))

                # save confusion matrices
                filepath_stats_base = os.path.join(self.opt.experiment_path, 'statistics', best_model_filename)
                if not filepath_stats_base.endswith('/'):
                    filepath_stats_base += '/'
                os.makedirs(filepath_stats_base, exist_ok=True)
                create_save_plotted_confusion_matrix(dev_stats['confusion_matrix'],
                                                     expected_labels=self.sorted_expected_label_values,
                                                     basepath=filepath_stats_base)
                logger.debug("created confusion matrices in path: {}".format(filepath_stats_base))

            if early_stopping.early_stop:
                logger.info("early stopping after {} epochs without improvement, total epochs: {} of {}".format(
                    early_stopping.patience, epoch, self.opt.num_epoch))
                break

        return best_model_path, best_model_filename

    def _evaluate(self, data_loader):
        t_labels_all, t_outputs_all = None, None

        # switch model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_labels = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                if t_labels_all is None:
                    t_labels_all = t_labels
                    t_outputs_all = t_outputs
                else:
                    t_labels_all = torch.cat((t_labels_all, t_labels), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        # softmax: get predictions from outputs
        y_pred = torch.argmax(t_outputs_all, -1).cpu()
        y_true = t_labels_all.cpu()

        stats = self.evaluator.calc_statistics(y_true, y_pred)

        return stats

    def get_normalized_inv_class_frequencies(self):
        inv_freqs = []

        for label_name in self.sorted_expected_label_names:
            inv_freq_of_class = 1.0 / self.testset.label_counter[label_name]
            inv_freqs.append(inv_freq_of_class)

        for i in range(len(inv_freqs)):
            inv_freqs[i] = inv_freqs[i] / sum(inv_freqs)

        return inv_freqs

    def run_crossval(self):
        # Loss and Optimizer
        if self.opt.lossweighting:
            inv_class_freqs = self.get_normalized_inv_class_frequencies()
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
            best_model_path, best_model_filename = self._train(criterion, optimizer, train_data_loader, val_data_loader,
                                                               fid)

            # evaluate the model that performed best during training,
            self.model.load_state_dict(torch.load(best_model_path))
            test_stats = self._evaluate(test_data_loader)
            # append its results to the list of results, which will be aggregated after all folds are completed
            all_test_stats.append(test_stats)

            self.evaluator.log_statistics(test_stats, "evaluation of fold {} on test-set".format(fid))

        logger.info("aggregating performance statistics from all {} folds".format(self.opt.crossval))
        mean_test_stats = self.evaluator.mean_from_all_statistics(all_test_stats)
        self.evaluator.log_statistics(mean_test_stats,
                                      "aggregated evaluation results of all folds on test-set".format(fid))

        logger.info("finished execution of this crossval run. exiting.")

        # print snem value to stdout, for the controller to parse it
        print(mean_test_stats[self.opt.snem])

    def run(self):
        # Loss and Optimizer
        if self.opt.lossweighting:
            inv_class_freqs = self.get_normalized_inv_class_frequencies()
            class_weights = torch.tensor(inv_class_freqs).to(self.opt.device)
        else:
            class_weights = None

        if self.opt.lsr:
            criterion = CrossEntropyLoss_LSR(self.opt.device, para_LSR=0.2, weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        dev_data_loader = DataLoader(dataset=self.devset, batch_size=self.opt.batch_size, shuffle=False)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        logger.info("starting training...")
        time_training_start = time.time()
        best_model_path, best_model_filename = self._train(criterion, optimizer, train_data_loader, dev_data_loader)
        time_training_elapsed_mins = (time.time() - time_training_start) // 60
        logger.info("training finished. duration={}mins".format(time_training_elapsed_mins))

        self.post_training(best_model_path, best_model_filename, test_data_loader)

    def post_training(self, best_model_path, best_model_filename, test_data_loader):
        logger.info("loading model that performed best during training: {}".format(best_model_path))
        self.model.load_state_dict(torch.load(best_model_path))

        logger.info("evaluating best model on test-set")
        # set model into evaluation mode (cf. https://pytorch.org/docs/stable/nn.html#torch.nn.Module.train)
        self.model.eval()

        # do the actual evaluation
        test_stats = self._evaluate(test_data_loader)
        test_snem = test_stats[self.opt.snem]

        self.evaluator.log_statistics(test_stats, "evaluation on test-set")

        # save confusion matrices
        test_confusion_matrix = test_stats['confusion_matrix']
        filepath_stats_prefix = os.path.join(self.opt.experiment_path, 'statistics', best_model_filename)
        os.makedirs(filepath_stats_prefix, exist_ok=True)
        if not filepath_stats_prefix.endswith('/'):
            filepath_stats_prefix += '/'
        create_save_plotted_confusion_matrix(test_confusion_matrix,
                                             expected_labels=self.sorted_expected_label_values,
                                             basepath=filepath_stats_prefix)

        logger.info("finished execution of this run. exiting.")

        # print snem value to stdout, for the controller to parse it
        print(test_snem)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1337, type=int, help='set seed for reproducibility')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    # semantic-relative-distance, see the paper of LCF-BERT model
    parser.add_argument('--SRD', default=3, type=int, help='set SRD')
    parser.add_argument('--snem', default='recall_avg', help='see evaluator.py for valid options')
    parser.add_argument("--devmode", type=str2bool, nargs='?', const=True, default=False,
                        help="devmode, default off, enable by using True")
    parser.add_argument('--experiment_path', default=None, type=str,
                        help='if defined, all data will be read from / saved to a folder in the experiments folder')
    parser.add_argument('--crossval', default=0, type=int,
                        help='if k>0 k-fold crossval mode is enabled. the tool will merge ')
    parser.add_argument('--lossweighting', type=str2bool, nargs='?', const=True, default=False,
                        help="True: loss weights according to class frequencies, False: each class has the same loss per example")
    parser.add_argument("--lsr", type=str2bool, nargs='?', const=True, default=False,
                        help="True: enable label smoothing regularization; False: disable")
    parser.add_argument('--bert_spc_reduction', type=str, default='mean_last_hidden_states')

    opt = parser.parse_args()

    if opt.seed is not None:
        logger.info("setting random seed: {}".format(opt.seed))
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'ram': RAM,

        'aen_bert': AEN_BERT,
        'bert_spc': BERT_SPC,
        'aen_distilbert': AEN_DISTILBERT,
        'distilbert_spc': DISTILBERT_SPC,
    }
    model_name_to_pretrained_model_name = {
        'aen_bert': 'bert-base-uncased',
        'aen_distilbert': 'distilbert-base-uncased',
        'bert_spc': 'bert-base-uncased',
        'distilbert_spc': 'distilbert-base-uncased',
    }
    input_columns = {
        'lstm': ['text_raw_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'atae_lstm': ['text_raw_indices', 'aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'memnet': ['text_raw_without_aspect_indices', 'aspect_indices'],
        'ram': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices',
                   'text_right_with_aspect_indices'],
        'tnet_lf': ['text_raw_indices', 'aspect_indices', 'aspect_in_text'],
        'aoa': ['text_raw_indices', 'aspect_indices'],
        'mgan': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'aen_glove': ['text_raw_indices', 'aspect_indices'],
        'lcf_bert': ['text_bert_indices', 'bert_segments_ids', 'text_raw_bert_indices', 'aspect_bert_indices'],

        'aen_bert': ['text_raw_bert_indices', 'aspect_bert_indices'],
        'aen_distilbert': ['text_raw_bert_indices', 'aspect_bert_indices'],
        'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
        'distilbert_spc': ['text_bert_indices'],
    }
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
    if opt.model_name == 'ram':
        pass
    else:
        opt.pretrained_model_name = model_name_to_pretrained_model_name[opt.model_name]

    if not opt.experiment_path:
        opt.experiment_path = '.'
    if not opt.experiment_path.endswith('/'):
        opt.experiment_path = opt.experiment_path + '/'

    if not opt.dataset_path:
        logger.info("dataset_path not defined, creating from dataset_name...")
        opt.dataset_path = os.path.join('datasets', opt.dataset_name)
        if not opt.dataset_path.endswith('/'):
            opt.dataset_path = opt.dataset_path + '/'
        logger.info("dataset_path created from dataset_name: {}".format(opt.dataset_path))
    else:
        logger.info("dataset_path defined: {}".format(opt.dataset_path))

    opt.inputs_cols = input_columns[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    if torch.cuda.is_available():
        opt.device = torch.device('cuda')
        logger.info("GPGPU enabled. CUDA dev index: {}".format(opt.device.index))
    else:
        opt.device = torch.device('cpu')

    if opt.device.type == 'cuda':
        logger.info('using GPU (cuda memory allocated: {})'.format(torch.cuda.memory_allocated()))
    else:
        logger.info("using CPU (cuda not available)")

    # set dataset_path to include experiment_path
    opt.dataset_path = os.path.join(opt.experiment_path, opt.dataset_path)

    ins = Instructor(opt)

    if opt.crossval > 0:
        ins.run_crossval()
    else:
        ins.run()


if __name__ == '__main__':
    main()
