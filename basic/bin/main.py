import sys,os,argparse,json,logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from basic.data.PreData import preData
from basic.data import *
from basic.models import MODELS,OUT_FN
from basic.train import valid_epoch,train_epoch,Loss,optims,hard_epoch
import torch.nn as nn
import torch.optim as optim
from basic.utils import Checkpointer
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

def get_args():
    parse=argparse.ArgumentParser(description="main function for trainning model")
    parse.add_argument('-c','--config',type=str,help="config path")
    return  parse.parse_args()

def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
    args=get_args()
    with open(args.config,'r')as f:
        config = json.load(f)
    # set train_dataset
    if config["dataset"]["train"].get("save_path",None):
        dataset=config["dataset"]["train"]["save_path"]
    else:
        dataset =  os.path.join(config['workspace'],'patch_list')
        config["dataset"]["train"]["save_path"]=dataset
    tumor_list = os.path.join(dataset, 'train_tumor.list')
    normal_list = os.path.join(dataset, 'train_normal.list')
    if not os.path.exists(tumor_list):
        os.system(f'mkdir -p {dataset}')
        tumor_list,normal_list=preData(**config["dataset"]["train"])
    config["dataset"]["train"]['tumor_list']=tumor_list
    config["dataset"]["train"]['normal_list']=normal_list
    # dyanamicloader= DynamicLoader(**config["dataset"]["train"])
    dataloader = DATALOADER[config['dataloader']](**config["dataset"])
    # load model
    model_name=config["model"]
    logging.info(f'loading {model_name}')
    train_config = config['train']
    valid_config = config['valid']
    net = nn.DataParallel(MODELS[model_name], device_ids=train_config['GPU'])
    optimizer = optims[train_config["optimizer"]["optim"]](net.parameters(), lr=train_config["optimizer"]['lr'], momentum=train_config["optimizer"]['momentum'],weight_decay=train_config["optimizer"]['weight_decay'])
    save = os.path.join(config['workspace'], 'train', 'model')
    out_fn=OUT_FN[config["model"]]

    if not os.path.exists(save):
        os.system(f"mkdir -p {save}")
    ckpter = Checkpointer(save)
    ckpt = ckpter.load(train_config['start'])
    last_epoch = -1
    if ckpt[0]:
        net.load_state_dict(ckpt[0])
        optimizer.load_state_dict(ckpt[1])
        train_config['start'] = ckpt[2]+1
        last_epoch = ckpt[2]
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        train_config['start'] = ckpt[2]+1
    net.to('cuda')
    # set train
    loss_fn = Loss[train_config['loss']]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                        train_config['scheduler']['step_size'],
                                        gamma=train_config['scheduler']['gamma'],
                                        last_epoch=last_epoch)
    # visualization
    visualize_path=os.path.join(config['workspace'],'train','visualization')
    writer=SummaryWriter(visualize_path)
    writer.add_text('config',str(config))
    best_epoch = 0
    best_valid_acc = 0
    f_patch_list=[]
    for epoch in range(train_config['start'],train_config['last']):
        # train
        total_acc, pos_acc, neg_acc, loss=train_epoch(epoch, net,loss_fn,out_fn, dataloader.load_train_data(**train_config),  optimizer)
        writer.add_scalar('acc_in_train',total_acc,epoch)
        writer.add_scalar('pos_acc_in_train', pos_acc,epoch)
        writer.add_scalar('neg_acc_in_train', neg_acc,epoch)
        writer.add_scalar('loss_in_train', loss,epoch)
        writer.add_scalar('Lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        scheduler.step()
        # valid
        total_acc, pos_acc, neg_acc, loss = valid_epoch(net,loss_fn,out_fn, dataloader.load_valid_data(**valid_config))
        writer.add_scalar('acc_in_valid', total_acc,epoch)
        writer.add_scalar('pos_acc_in_valid', pos_acc,epoch)
        writer.add_scalar('neg_acc_in_valid', neg_acc,epoch)
        writer.add_scalar('loss_in_valid', loss,epoch)
        state_dict = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "last_epoch": epoch,
        }
        ckpter.save(epoch, state_dict, total_acc)
        if total_acc>best_valid_acc:
            best_epoch = epoch
            best_valid_acc=total_acc

    # use the best for hard minning
    logging.info(f'best_epoch: {best_epoch}, best_valid_acc:{total_acc}')
    best_load=ckpter.load(best_epoch)
    net.load_state_dict(best_load[0])
    hard_config=config['hard']
    save = os.path.join(config['workspace'], 'hard', 'model')
    if not os.path.exists(save):
        os.system(f"mkdir -p {save}")
    ckpter = Checkpointer(save)
    visualize_path = os.path.join(config['workspace'], f'hard_{best_epoch}', 'visualization')
    writer = SummaryWriter(visualize_path)
    optimizer = optims[hard_config["optimizer"]](net.parameters(), lr=hard_config["optimizer"]['lr'])
    for epoch in range(hard_config['epoch']):
        # find hard expamples
        hardlist=hard_epoch(epoch, net, out_fn, dataloader.load_normal_data(**hard_config), optimizer)
        hard_dataloader=HardDataLoader(hardlist,**config['dataset']['hard'])
        # train
        total_acc, pos_acc, neg_acc, loss = train_epoch(epoch,loss_fn,hard_dataloader,optimizer)
        writer.add_scalar('acc_in_hard', total_acc), epoch
        writer.add_scalar('pos_acc_in_hard', pos_acc, epoch)
        writer.add_scalar('neg_acc_in_hard', neg_acc, epoch)
        writer.add_scalar('loss_in_hard', loss, epoch)
        writer.add_scalar('Lr_in_hard ', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        # valid, 使用trainSet中的normal样本检查是否合适
        total_acc, pos_acc, neg_acc, loss = valid_epoch(net, loss_fn, dataloader.load_normal_data(**valid_config))
        writer.add_scalar('acc_in_valid', total_acc,epoch)
        writer.add_scalar('pos_acc_in_valid', pos_acc,epoch)
        writer.add_scalar('neg_acc_in_valid', neg_acc,epoch)
        writer.add_scalar('loss_in_valid', loss,epoch)
        state_dict = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "last_epoch": epoch,
        }
        ckpter.save(epoch, state_dict, total_acc)
        if total_acc>best_valid_acc:
            best_epoch = epoch
            best_valid_acc=total_acc

if __name__=="__main__":
    main()