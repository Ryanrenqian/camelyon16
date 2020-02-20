import sys,os,logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from basic.data import *
from basic.models import MODELS,OUT_FN
from basic.train import valid_epoch,train_epoch,Loss,optims,hard_epoch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from basic.utils import Checkpointer,Config
import random
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter
from basic.utils import acc_metric,save_hard_example,Counter

def main():
    # config
    tumor_list=''
    normal_list=''
    batch_size=128
    tumor_dataset=ListDataset(list_file=tumor_list,tif_folder='/root/workspace/dataset/CAMELYON16/training/*',all_class=1)
    normal_dataset=ListDataset(list_file=normal_list,tif_folder='/root/workspace/dataset/CAMELYON16/training/*',all_class=0)
    tumor_dataloader=DataLoader(tumor_dataset,batch_size=batch_size,num_workers=20)
    normal_dataloader=DataLoader(normal_dataset,batch_size=batch_size,num_workers=20)
    model_name='scannet'
    gpus=[0,1,2,3]
    optimizer = optims['SGD']
    net = nn.DataParallel(MODELS[model_name], device_ids=gpus)
    save = 'workspace/train/models'
    out_fn = OUT_FN['inceptionv3']
    max_iteration=10000000
    st_iter = 0
    batch_size=32
    # load
    if not os.path.exists(save):
        os.system(f"mkdir -p {save}")
    ckpter = Checkpointer(save)
    start=0
    ckpt = ckpter.load()
    last_epoch = -1
    if ckpt[0]:
        net.load_state_dict(ckpt[0])
        optimizer.load_state_dict(ckpt[1])
        start= ckpt[2]+1
        last_epoch = ckpt[2]
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        start = ckpt[2]+1
    net.to('cuda')

    # train model
    loss_fn = Loss['CrossEntropy']
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer)
    # visualization
    visualize_path = os.path.join(save, 'train', 'visualization')
    writer = SummaryWriter(visualize_path)
    #writer.add_text('config', str(config.config))
    best_epoch = 0
    best_valid_acc = 0
    # stats

    for i in range(st_iter,max_iteration):
        data_tumor,target_tumor=next(tumor_dataloader)
        data_tumor=Variable(data_tumor)
        target_tumor=Variable(target_tumor)
        data_nomal,target_normal=next(normal_dataloader)
        data_nomal=Variable(data_nomal)
        target_normal=Variable(target_normal)
        idx_rand=Variable(torch.randperm(batch_size * 2))
        data=torch.cat([data_tumor,data_nomal])[idx_rand].cuda()
        target=torch.cat([target_tumor,target_normal])[idx_rand].cpu()
        output = net(data).squeeze().cpu()
        loss=loss_fn(output,target)
        loss.backward()
        optimizer.step()
        probs = out_fn(output)
        correct_pos, total_pos, correct_neg, total_neg = acc_metric(probs, target, 0.5)
        total_acc = (correct_pos+correct_neg)/(total_pos+total_neg)
        pos_acc = correct_pos/total_pos
        neg_acc = correct_neg/total_neg
        writer.add_scalar('loss',loss,i)
        writer.add_scalar('total_acc',total_acc,i)
        writer.add_scalar('pos_acc',pos_acc,i)
        writer.add_scalar('net_acc',neg_acc,i)





