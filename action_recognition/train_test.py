#!/usr/bin/env python

# -*- coding: utf-8 -*-
#
# Adapted from https://github.com/lshiwjx/2s-AGCN for BABEL (https://babel.is.tue.mpg.de/)

from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import time
from collections import *
import numpy as np

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

import pdb
import ipdb

# Custom
from class_balanced_loss import CB_loss


# class GradualWarmupScheduler(_LRScheduler):
#       def __init__(self, optimizer, total_epoch, after_scheduler=None):
#               self.total_epoch = total_epoch
#               self.after_scheduler = after_scheduler
#               self.finished = False
#               self.last_epoch = -1
#               super().__init__(optimizer)

#       def get_lr(self):
#               return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

#       def step(self, epoch=None, metric=None):
#               if self.last_epoch >= self.total_epoch - 1:
#                       if metric is None:
#                               return self.after_scheduler.step(epoch)
#                       else:
#                               return self.after_scheduler.step(metric, epoch)
#               else:
#                       return super(GradualWarmupScheduler, self).step(epoch)


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=True,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')

    #training
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    # loss
    parser.add_argument(
        '--loss',
        type=str,
        default='CE',
        help='loss type(CE or focal)')
    parser.add_argument(
        '--label_count_path',
        default=None,
        type=str,
        help='Path to label counts (used in loss weighting)')
    parser.add_argument(
        '---beta',
        type=float,
        default=0.9999,
        help='Hyperparameter for Class balanced loss')
    parser.add_argument(
        '--gamma',
        type=float,
        default=2.0,
        help='Hyperparameter for Focal loss')

    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser


class Processor():
    """
            Processor for Skeleton-based Action Recgnition
    """
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    # answer = input('delete it? y/n:')
                    answer = 'y'
                    if answer == 'y':
                        print('Deleting dir...')
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        # input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_per_class_acc = 0

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_class_weights(self):
        if arg.label_count_path == None:
            raise Exception('No label  count path..!!!')
        with open(arg.label_count_path, 'rb') as f:
            label_count = pickle.load(f)
        img_num_per_cls = []
        # ipdb.set_trace()
        for cls_idx in range(len(label_count)):
            img_num_per_cls.append(int(label_count[cls_idx]))
        self.samples_per_class = img_num_per_cls

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        print(self.model)
        self.loss_type = arg.loss
        if self.loss_type != 'CE':
            self.load_class_weights()

        if self.arg.weights:
            self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, wb_dict, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_value, batch_acc, batch_per_class_acc = [], [], []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print('only train part, require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = True
            else:
                print('only train part, do not require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = False

        nb_classes = self.arg.model_args['num_class']
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        for batch_idx, (data, label, sid, seg_id, chunk_n, anntr_id, index) in enumerate(process):

            self.global_step += 1
            # get data
            data = Variable(data.float().cuda(self.output_device), requires_grad=False)
            label = Variable(label.long().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            # forward
            output = self.model(data)

            if self.loss_type == "CE":
                l_type = nn.CrossEntropyLoss()
                loss = l_type(output, label)
            else:
                loss = CB_loss(label, output,
                               self.samples_per_class,
                               nb_classes, self.loss_type,
                               self.arg.beta,
                               self.arg.gamma,
                               self.arg.device[0]
                              )

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            # Compute per-class acc.
            value, predict_label = torch.max(output.data, 1)
            for t, p in zip(label.view(-1), predict_label.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            # Acc.
            acc = torch.mean((predict_label == label.data).float())
            batch_acc.append(acc.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            # if self.global_step % self.arg.log_interval == 0:
            #         self.print_log(
            #                 '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
            #                         batch_idx, len(loader), loss.data[0], lr))
            timer['statistics'] += self.split_time()

        per_class_acc_vals = confusion_matrix.diag()/confusion_matrix.sum(1)
        per_class_acc =  torch.mean(per_class_acc_vals).float()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log('\tTop-1-norm: {:.3f}%'.format(100*per_class_acc))

        # Log
        wb_dict['train loss'] = np.mean(loss_value)
        wb_dict['train acc'] = np.mean(batch_acc)

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')

        return wb_dict

    @torch.no_grad()
    def eval(self, epoch,
             wb_dict,
             save_score=True,
             loader_name=['test'],
             wrong_file=None,
             result_file=None
             ):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            pred_label_list = []
            step = 0
            nb_classes = self.arg.model_args['num_class']
            confusion_matrix = torch.zeros(nb_classes, nb_classes)
            process = tqdm(self.data_loader[ln])
            for batch_idx, (data, label, sid, seg_id, chunk_n, anntr_id, index) in enumerate(process):
                data = Variable(
                    data.float().cuda(self.output_device),
                    requires_grad=False)
                # volatile=True)
                label = Variable(
                    label.long().cuda(self.output_device),
                    requires_grad=False)
                # volatile=True)
                output = self.model(data)

                if self.loss_type == "CE":
                    l_type = nn.CrossEntropyLoss()
                    loss = l_type(output, label)
                else:
                    loss = CB_loss(label, output,
                                        self.samples_per_class,
                                        nb_classes, self.loss_type,
                                        self.arg.beta,
                                        self.arg.gamma,
                                        self.arg.device[0]
                                        )
                # Store outputs
                logits = output.data.cpu().numpy()
                score_frag.append(logits)
                loss_value.append(loss.data.item())

                _, predict_label = torch.max(output.data, 1)
                pred_label_list.append(predict_label)

                step += 1

                # Compute per-class acc.
                for t, p in zip(label.view(-1), predict_label.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            per_class_acc_vals = confusion_matrix.diag()/confusion_matrix.sum(1)
            per_class_acc =  torch.mean(per_class_acc_vals).float()
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)

            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            topk_scores = { k: self.data_loader[ln].dataset.top_k(score, k) \
                            for k in self.arg.show_topk }

            wb_dict['val loss'] = loss
            wb_dict['val acc'] = accuracy
            wb_dict['val per class acc'] = per_class_acc
            for k in topk_scores:
                wb_dict['val top{0} score'.format(k)] = topk_scores[k]

            if accuracy > self.best_acc:
                self.best_acc = accuracy
            if per_class_acc > self.best_per_class_acc:
                self.best_per_class_acc = per_class_acc

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)
                self.val_writer.add_scalar('per_class_acc', per_class_acc , self.global_step)

            score_dict = list(zip(
                self.data_loader[ln].dataset.label[1],  # sid
                self.data_loader[ln].dataset.sample_name,  # seg_id
                self.data_loader[ln].dataset.label[2],  # chunk_id
                score))

            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            self.print_log('\tTop-1-norm: {:.3f}%'.format(100*per_class_acc))
            for k in topk_scores:
                self.print_log('\tTop{}: {:.3f}%'.format(k, 100*topk_scores[k]))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)
        return wb_dict

    def start(self):
        wb_dict = {}
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):

                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)

                # Wandb logging
                wb_dict = {'lr': self.lr}

                # Train
                wb_dict = self.train(epoch, wb_dict, save_model=save_model)

                # Eval. on val set
                wb_dict = self.eval(
                    epoch,
                    wb_dict,
                    save_score=self.arg.save_score,
                    loader_name=['test'])
                # Log stats. for this epoch
                print('Epoch: {0}\nMetrics: {1}'.format(epoch, wb_dict))

            print('best accuracy: ', self.best_acc, ' model_name: ', self.arg.model_saved_name)

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))

            wb_dict = self.eval(epoch=0, wb_dict=wb_dict,
                                save_score=self.arg.save_score,
                                loader_name=['test'],
                                wrong_file=wf,
                                result_file=rf
                                )
            print('Inference metrics: ', wb_dict)
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    print('BABEL Action Recognition')
    print('Config: ', arg)
    init_seed(0)
    processor = Processor(arg)
    processor.start()
