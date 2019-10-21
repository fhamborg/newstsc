# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import argparse
import math
import os
import random
import time

import numpy
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers import BertModel

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, FXDataset
from fxlogger import get_logger
from models import LSTM, IAN, MemNet, RAM, TD_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, LCF_BERT
from models.aen import AEN_BERT, AEN_GloVe
from models.bert_spc import BERT_SPC
from plotter_utils import create_save_plotted_confusion_matrices

logger = get_logger()


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        logger.info(opt)

        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        logger.info("loading datasets from folder '{}'".format(opt.dataset_name))
        self.trainset = FXDataset(opt.dataset_path + 'train.jsonl', tokenizer)
        self.devset = FXDataset(opt.dataset_path + 'dev.jsonl', tokenizer)
        self.testset = FXDataset(opt.dataset_path + 'test.jsonl', tokenizer)
        logger.info("loaded three datasets")

        self.expected_labels = [0, 1, 2]

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

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, dev_data_loader):
        max_dev_acc = 0
        max_dev_f1 = 0
        global_step = 0
        path = None
        filename = None
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

            dev_acc, dev_f1, dev_confusion_matrix = self._evaluate_acc_f1_confusionmatrix(dev_data_loader)
            logger.info('> dev_acc: {:.4f}, dev_f1: {:.4f}'.format(dev_acc, dev_f1))
            logger.info("confusion matrix")
            logger.info(dev_confusion_matrix)

            if dev_acc > max_dev_acc:
                logger.info("model yields best performance so far, saving to disk...")
                max_dev_acc = dev_acc
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                filename = '{0}_{1}_val_acc{2}'.format(self.opt.model_name, self.opt.dataset_name, round(dev_acc, 4))
                path = 'state_dict/' + filename
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))

                # save confusion matrices
                create_save_plotted_confusion_matrices(dev_confusion_matrix, expected_labels=self.expected_labels,
                                                       basefilename=filename)
                logger.info("created confusion matrices in folder statistics/")

            if dev_f1 > max_dev_f1:
                max_dev_f1 = dev_f1

        return path, filename

    def _evaluate_acc_f1_confusionmatrix(self, data_loader):
        n_correct, n_total = 0, 0
        t_labels_all, t_outputs_all = None, None

        # switch model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_labels = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_labels).sum().item()
                n_total += len(t_outputs)

                if t_labels_all is None:
                    t_labels_all = t_labels
                    t_outputs_all = t_outputs
                else:
                    t_labels_all = torch.cat((t_labels_all, t_labels), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total

        t_predictions_all = torch.argmax(t_outputs_all, -1).cpu()

        logger.debug("labels :     {}".format(t_labels_all))
        logger.debug("predictions: {}".format(t_predictions_all))

        f1 = metrics.f1_score(t_labels_all.cpu(), t_predictions_all, labels=self.expected_labels, average='macro')
        confusion_mat = metrics.multilabel_confusion_matrix(t_labels_all.cpu(), t_predictions_all,
                                                            labels=self.expected_labels)

        return acc, f1, confusion_mat

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
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

        logger.info("loading model that performed best during training: {}".format(best_model_path))
        self.model.load_state_dict(torch.load(best_model_path))

        logger.info("evaluating best model on test-set")
        # set model into evaluation mode (cf. https://pytorch.org/docs/stable/nn.html#torch.nn.Module.train)
        self.model.eval()
        # do the actual evaluation
        test_acc, test_f1, test_confusion_matrix = self._evaluate_acc_f1_confusionmatrix(test_data_loader)
        logger.info("evaluation finished.")
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))

        # save confusion matrices
        create_save_plotted_confusion_matrices(test_confusion_matrix, expected_labels=self.expected_labels,
                                               basefilename=best_model_filename + "_testset")
        logger.info("created confusion matrices in folder statistics/")


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='aen_bert', type=str)
    parser.add_argument('--dataset_name', default='poltsanews', type=str,
                        help='name of the sub-folder in \'datasets\' containing files called [train,dev,test].jsonl')
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
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    # semantic-relative-distance, see the paper of LCF-BERT model
    parser.add_argument('--SRD', default=3, type=int, help='set SRD')
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'aen_glove': AEN_GloVe,
        'lcf_bert': LCF_BERT,
        # default hyper-parameters for LCF-BERT model is as follws:
        # lr: 2e-5
        # l2: 1e-5
        # batch size: 16
        # num epochs: 5
    }
    input_colses = {
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
        'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
        'aen_bert': ['text_raw_bert_indices', 'aspect_bert_indices'],
        'aen_glove': ['text_raw_indices', 'aspect_indices'],
        'lcf_bert': ['text_bert_indices', 'bert_segments_ids', 'text_raw_bert_indices', 'aspect_bert_indices'],
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

    opt.dataset_path = os.path.join('datasets', opt.dataset_name)
    if not opt.dataset_path.endswith('/'):
        opt.dataset_path = opt.dataset_path + '/'

    opt.inputs_cols = input_colses[opt.model_name]
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

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
