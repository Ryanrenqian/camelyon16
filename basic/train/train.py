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



def train_epoch(epoch,net,loss_fn,out_fn,dataloader,optimizer):
    net.train()
    acc = {'correct_pos': Counter(), 'total_pos': Counter(),
           'correct_neg': Counter(), 'total_neg': Counter()}
    losses = Counter()
    time_counter = Counter()
    time_counter.addval(time.time(), key='training epoch start')
    for i, data in enumerate(tqdm.tqdm(dataloader, dynamic_ncols=True, leave=False), 0):
        inputs, labels, patch_list = data
        inputs,labels=inputs.cuda(),labels.cpu()
        outputs = net(inputs).squeeze().cpu()
        #             pdb.set_trace()
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        probs = out_fn(outputs)
        correct_pos, total_pos, correct_neg, total_neg = acc_metric(probs, labels, 0.5)
        acc['correct_pos'].addval(correct_pos)
        acc['total_pos'].addval(total_pos)
        acc['correct_neg'].addval(correct_neg)
        acc['total_neg'].addval(total_neg)
        losses.addval(loss.item(), len(outputs))
    time_counter.addval(time.time())
    TP = acc['correct_pos'].sum
    total_neg = acc['total_neg'].sum
    TN = acc['correct_neg'].sum
    total_pos = acc['total_pos'].sum
    total = total_pos + total_neg
    total_acc = (TP + TN) / total
    pos_acc = TP / total_pos
    neg_acc = TN / total_neg
    logging.info(f"train new epoch:{epoch},  [total_acc:{total_acc}-pos_acc:{pos_acc}-neg_acc:{pos_acc}], loss_avg:{losses.avg},time consume:{time_counter.interval()}s")
    return total_acc, pos_acc, neg_acc, losses.avg



def valid_epoch(net,loss_fn,out_fn,dataloader):
    net.eval()
    acc = {'correct_pos': Counter(), 'total_pos': Counter(),
           'correct_neg': Counter(), 'total_neg': Counter()}
    losses = Counter()
    for i, data in enumerate(tqdm.tqdm(dataloader, dynamic_ncols=True, leave=False), 0):
        inputs, labels, path_list = data
        # forward and step
        if torch.cuda.is_available():
            inputs = Variable(inputs.type(torch.cuda.FloatTensor))
        else:
            inputs = Variable(inputs.type(torch.FloatTensor))
        outputs = net(inputs).squeeze().cpu()
        loss = loss_fn(outputs, labels)
        probs = out_fn(outputs)
        correct_pos, total_pos, correct_neg, total_neg = acc_metric(probs, labels, 0.5)
        acc['correct_pos'].addval(correct_pos)
        acc['total_pos'].addval(total_pos)
        acc['correct_neg'].addval(correct_neg)
        acc['total_neg'].addval(total_neg)
        losses.addval(loss.item(), len(outputs))
    TP = acc['correct_pos'].sum
    total_pos = acc['total_pos'].sum
    TN = acc['correct_neg'].sum
    total_neg = acc['total_neg'].sum
    total = total_pos + total_neg
    total_acc = (TP + TN) / total
    pos_acc = TP / total_pos
    neg_acc = TN / total_neg
    logging.info(f'total_pos:{total_pos},total_neg:{total_neg}')
    return total_acc, pos_acc, neg_acc, losses.avg,

def hard_epoch(net,out_fn,dataloader,epoch,workspace):
    '''
    采样20W个样本
    '''
    records = []
    samples = 0
    for i, data in enumerate(dataloader, 0):
        inputs, class_ids, patch_names = data
        outputs = net(inputs).squeeze().cpu()
        outputs = out_fn(outputs)
        correct_pos, total_pos, correct_neg, total_neg = acc_metric(outputs, class_ids, 0.5)
        samples += len(outputs)
        for i, patch_name in zip(outputs, patch_names):
            if i > 0.5:
                records.append(patch_name + '\n')
        if samples > 200000:
            break
    save_path=os.path.join(workspace,epoch)
    hard_examples = save_hard_example(save_path, records)
    return save_path