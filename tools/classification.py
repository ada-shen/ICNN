import os
import h5py
from tools.load_data import load_data
from tools.init_model import init_model
from tools.train_model import train_model
from tools.showresult import showresult
from tools.lib import *

def classification(root_path,args):
    # task/classification
    task_path = os.path.join(root_path,'task',args.task_name)
    make_dir(task_path)
    # task/classification/vgg_vd_16
    task_model_path = os.path.join(task_path,args.model)
    make_dir(task_model_path)
    task_model_dataset_path = os.path.join(task_model_path,args.dataset)
    make_dir(task_model_dataset_path)
    if args.dataset!='helen' and args.dataset!='celeba' and args.dataset!='cubsample':
        task_model_dataset_labelname_path = os.path.join(task_model_dataset_path,args.label_name)
        make_dir(task_model_dataset_labelname_path)
    else:
        task_model_dataset_labelname_path = task_model_dataset_path
    task_model_dataset_labelname_taskid_path = os.path.join(task_model_dataset_labelname_path,str(args.task_id))
    make_dir(task_model_dataset_labelname_taskid_path)

    train_dataloader, var_dataloader, density, dataset_length = load_data(root_path, task_model_dataset_labelname_path, args)

    net = init_model(root_path,args)

    max_acc,max_epoch = train_model(task_model_dataset_labelname_taskid_path, args, net, train_dataloader, var_dataloader, density, dataset_length)
    # calculate stability
    '''max_sta = showresult(max_epoch,task_model_dataset_labelname_taskid_path, task_model_dataset_labelname_path, root_path, args)
    with open('train.log','a') as f:
        f.writelines(args.label_name+" "+str(max_acc)+" "+str(max_epoch)+" "+str(max_sta)+"\n")'''
    print("\n")
    print(max_acc)







    
    
    
    




