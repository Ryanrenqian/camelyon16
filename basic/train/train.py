import sys
import os
import argparse
import logging
import json
import time

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.optim import SGD
import tqdm
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from basic.utils import acc_metric,save_hard_example,Counter
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)



def train_epoch(epoch,net,loss_fn,dataloader,optimizer):
    net.train()
    acc = {'correct_pos': Counter(), 'total_pos': Counter(),
           'correct_neg': Counter(), 'total_neg': Counter()}
    losses = Counter()
    time_counter = Counter()
    time_counter.addval(time.time(), key='training epoch start')
    for i, data in enumerate(tqdm.tqdm(dataloader, dynamic_ncols=True, leave=False), 0):
        _input, _labels, path_list = data
        if torch.cuda.is_available():
            _input = Variable(_input.type(torch.cuda.FloatTensor))
        else:
            _input = Variable(_input.type(torch.FloatTensor))
        _output = net(_input).squeeze().cpu()
        #             pdb.set_trace()
        loss = loss_fn(_output, _labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        probs = F.softmax(_output)[:,1].cpu()
        correct_pos, total_pos, correct_neg, total_neg = acc_metric(probs, _labels, 0.5)
        acc['correct_pos'].addval(correct_pos)
        acc['total_pos'].addval(total_pos)
        acc['correct_neg'].addval(correct_neg)
        acc['total_neg'].addval(total_neg)
        losses.addval(loss.item(), len(_output))
    TP = acc['correct_pos'].sum
    total_pos = acc['total_neg'].sum
    TN = acc['correct_neg'].sum
    total_neg = acc['total_pos'].sum
    total = total_pos + total_neg
    time_counter.addval(time.time())
    total_neg = acc['total_pos'].sum
    total = total_pos + total_neg
    total_acc = (TP + TN) / total
    pos_acc = TP / total_pos
    neg_acc = TN / total_neg
    logging.info(
        'train new epoch:%d, lr:%.5f, [total:%.2f-pos:%.2f-neg:%.2f], loss:%.2f,time consume:%.2f s' % (
            epoch, optimizer.state_dict()['param_groups'][0]['lr'],
            total_acc, pos_acc, neg_acc,
            losses.avg,
            time_counter.interval()), '\r')
    return total_acc, pos_acc, neg_acc, losses.avg



def valid_epoch(net,loss_fn,dataloader):
    net.eval()
    acc = {'correct_pos': Counter(), 'total_pos': Counter(),
           'correct_neg': Counter(), 'total_neg': Counter()}
    losses = Counter()
    for i, data in enumerate(tqdm.tqdm(dataloader, dynamic_ncols=True, leave=False), 0):
        input, _labels, path_list = data
        # forward and step
        if torch.cuda.is_available():
            input = Variable(input.type(torch.cuda.FloatTensor))
        else:
            input = Variable(input.type(torch.FloatTensor))
        output = net(input).squeeze().cpu()
        loss = loss_fn(output, _labels)
        probs = net.out_fn(output)
        correct_pos, total_pos, correct_neg, total_neg = acc_metric(probs, _labels, 0.5)
        acc['correct_pos'].addval(correct_pos)
        acc['total_pos'].addval(total_pos)
        acc['correct_neg'].addval(correct_neg)
        acc['total_neg'].addval(total_neg)
        losses.addval(loss.item(), len(output))
    TP = acc['correct_pos'].sum
    total_pos = acc['total_neg'].sum
    TN = acc['correct_neg'].sum
    total_neg = acc['total_pos'].sum
    total = total_pos + total_neg
    total_acc = (TP + TN) / total
    pos_acc = TP / total_pos
    neg_acc = TN / total_neg
    return total_acc, pos_acc, neg_acc, losses.avg

def hard_epoch(net,loss_fn,dataloader,epoch,workspace):
    records = []
    samples = 0
    for i, data in enumerate(dataloader, 0):
        input_imgs, class_ids, patch_names = data
        output = net(input_imgs)
        output = output.cpu()
        output = F.softmax(output)[:, 1]
        correct_pos, total_pos, correct_neg, total_neg = acc_metric(output.cpu(), class_ids, 0.5)
        samples += len(output)
        for i, patch_name in zip(output, patch_names):
            if i > 0.5:
                records.append(patch_name + '\n')

        if samples > 200000:
            break
    save_path=os.path.join(workspace,epoch)
    hard_examples = save_hard_example(save_path, records)
    return save_path