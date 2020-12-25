import os
import h5py
from tools.load_data import load_data
from tools.init_model import init_model
from tools.train_model import train_model
from tools.showresult import showresult
from tools.lib import *
from tools.load_data_multi import load_data_multi


def classification_multi(root_path,args):

    task_path = os.path.join(root_path, 'task', args.task_name)
    make_dir(task_path)
    task_model_path = os.path.join(task_path, args.model)
    make_dir(task_model_path)
    task_model_dataset_path = os.path.join(task_model_path, args.dataset)
    make_dir(task_model_dataset_path)
    label_name = ""
    for i in range(args.label_num):
        label_name = label_name + args.label_name[i] +"_"
    label_name = label_name[:-1]
    task_model_dataset_labelname_path = os.path.join(task_model_dataset_path, label_name)
    make_dir(task_model_dataset_labelname_path)
    task_model_dataset_labelname_taskid_path = os.path.join(task_model_dataset_labelname_path, str(args.task_id))
    make_dir(task_model_dataset_labelname_taskid_path)

    train_dataloader, var_dataloader, density, dataset_length = load_data_multi(root_path, task_model_dataset_labelname_path, args)

    net = init_model(root_path,args)

    train_model(task_model_dataset_labelname_taskid_path, args, net, train_dataloader, var_dataloader, density, dataset_length)
    # calculate stability
    #showresult(task_model_dataset_labelname_taskid_path, task_model_dataset_labelname_path, root_path, args)




