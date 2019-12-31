import sys,os,argparse,json,logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
from basic.data.PreData import preData
from basic.data.DataLoader import DynamicLoader,ValidDataLoader,HardDataLoader
from basic.models import MODELS
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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args=get_args()
    with open(args.config,'r')as f:
        config = json.load(f)
    # set train_dataset
    dataset =  os.path.join(config['workspace'],'patch_list')
    config["dataset"]["train"]["save_path"]=dataset
    tumor_list = os.path.join(dataset, 'train_tumor.list')
    normal_list = os.path.join(dataset, 'train_normal.list')
    if not os.path.exists(tumor_list):
        os.system(f'mkdir -p {dataset}')
        tumor_list,normal_list=preData(**config["dataset"]["train"])
    config["dataset"]["train"]['tumor_list']=tumor_list
    config["dataset"]["train"]['normal_list']=normal_list
    train_dataloader = DynamicLoader(**config["dataset"]["train"])
    valid_dataloader = ValidDataLoader(**config['dataset']['valid'])

    # load model
    model_name=config["model"]
    logging.info(f'loading {model_name}')
    train_config = config['train']
    valid_config = config['valid']
    net = nn.DataParallel(MODELS[model_name](), device_ids=train_config['GPU'])
    optimizer = optims[train_config["optimizer"]["optim"]](net.parameters(), lr=train_config["optimizer"]['lr'], momentum=train_config["optimizer"]['momentum'],weight_decay=train_config["optimizer"]['weight_decay'])
    save = os.path.join(config['workspace'], 'train', 'model')
    if not os.path.exists(save):
        os.system(f"mkdir -p {save}")
    ckpter = Checkpointer(save)
    ckpt = ckpter.load(train_config['start'])
    if ckpt[0]:
        net.load_state_dict(ckpt[0])
        optimizer.load_state_dict(ckpt[1])
        train_config['start'] = ckpt[2]+1
    # set train
    net=net.to('cuda')
    loss_fn = Loss[train_config['loss']]
    # scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=train_config['scheduler']['star_epoch'],
    #                                                         gamma=train_config['scheduler']['gama'], last_epoch=train_config['scheduler']['last_epoch'])
    # visualization
    visualize_path=os.path.join(config['workspace'],'train','visualization')
    writer=SummaryWriter(visualize_path)
    best_epoch = 0
    best_valid_acc = 0
    for epoch in range(train_config['start'],train_config['last']):
        # train
        total_acc, pos_acc, neg_acc, loss=train_epoch(epoch, net, loss_fn, train_dataloader.load_data(**train_config),  optimizer)
        writer.add_scalar('acc in train',total_acc),epoch
        writer.add_scalar('pos_acc in train', pos_acc,epoch)
        writer.add_scalar('neg_acc in train', neg_acc,epoch)
        writer.add_scalar('loss in train', loss,epoch)
        writer.add_scalar('Lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        # valid
        total_acc, pos_acc, neg_acc, loss = valid_epoch(net, loss_fn, valid_dataloader.load_data(**valid_config))
        writer.add_scalar('acc in valid', total_acc,epoch)
        writer.add_scalar('pos_acc in valid', pos_acc,epoch)
        writer.add_scalar('neg_acc in valid', neg_acc,epoch)
        writer.add_scalar('loss in valid', loss,epoch)
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
        hardlist=hard_epoch(epoch, net, loss_fn, train_dataloader.load_data(**hard_config), optimizer)
        hard_dataloader=HardDataLoader(hardlist,**config['dataset']['hard'])
        # train
        total_acc, pos_acc, neg_acc, loss = train_epoch(epoch,loss_fn,hard_dataloader,optimizer)
        writer.add_scalar('acc in hard', total_acc), epoch
        writer.add_scalar('pos_acc in hard', pos_acc, epoch)
        writer.add_scalar('neg_acc in hard', neg_acc, epoch)
        writer.add_scalar('loss in hard', loss, epoch)
        writer.add_scalar('Lr in hard ', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        # valid
        total_acc, pos_acc, neg_acc, loss = valid_epoch(net, loss_fn, valid_dataloader.load_data(**valid_config))
        writer.add_scalar('acc in valid', total_acc,epoch)
        writer.add_scalar('pos_acc in valid', pos_acc,epoch)
        writer.add_scalar('neg_acc in valid', neg_acc,epoch)
        writer.add_scalar('loss in valid', loss,epoch)
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