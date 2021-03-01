import os
import argparse
#from tools import train_net
from tools.lib import init_lr
import random
import numpy as np
from tools.classification import classification
from tools.classification_multi import classification_multi
import torch

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False

seed_torch(0)

root_path = os.getcwd() #'/data2/lqm/pytorch_interpretable/py_icnn'
parser = argparse.ArgumentParser('parameters')
#info:gpu
parser.add_argument('--gpu_id',type=int,default=0,help='select the id of the gpu')
#info:task
parser.add_argument('--task_name',type=str,default='classification',help='select classification or classification_multi')
parser.add_argument('--task_id',type=int,default=0,help='0,1,2..')
parser.add_argument('--dataset',type=str,default='voc2010_crop',help='select voc2010_crop, helen, cub200,cubsample'
                                                                     'celeba, vocpart, ilsvrcanimalpart')
parser.add_argument('--imagesize',type=int,default=224,help='')
parser.add_argument('--label_name',type=str,default='bird',help='if voc2010_crop, set bird, cat, cow, dog, horse or sheep;'
                                                                'else, it does not matter')
parser.add_argument('--label_num',type=int,default=1,help='keep the same number of label_name')
parser.add_argument('--model',type=str,default='resnet_18',help='select vgg_vd_16, vgg_m, vgg_s, '
                                                                'alexnet, resnet_18, resnet_50, densenet_121')
parser.add_argument('--losstype',type=str,default='logistic',help='select logistic or softmax')
#info:hyper-parameter
parser.add_argument('--batchsize',type=int,default=8,help='select more than 8 may cause out of cuda memory, '
                                                          'when you want to choose different batchsize, you also need to adjust line 94 of /tools/sgd.py at the same time to make them consistent')
parser.add_argument('--dropoutrate',type=int,default=0,help='select the number between 0 and 1')
parser.add_argument('--lr',type=int,default=0,help='see function init_lr in /tools/lib.py for details')
parser.add_argument('--epochnum',type=int,default=0,help='see function init_lr in /tools/lib.py for details')
parser.add_argument('--weightdecay',type=int,default=0.0005,help='0.02,0.002')
parser.add_argument('--momentum',type=int,default=0.09,help='0.02,0.002')


args = parser.parse_args()
args.lr, args.epochnum = init_lr(args.model,args.label_num,args.losstype) #init lr and epochnum
if(args.task_name=='classification'):
    if args.dataset == 'celeba':
        args.label_num = 40
    classification(root_path, args)
else:
    if args.dataset == 'vocpart':
        args.label_name = ['bird','cat','cow','dog','horse','sheep']
        args.label_num = 6
    classification_multi(root_path,args)







