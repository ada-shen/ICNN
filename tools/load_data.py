import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tools.get_ilsvrimdb import get_ilsvrimdb
from tools.get_vocimdb import get_vocimdb
from tools.get_cubimdb import get_cubimdb
from tools.get_voc2010imdb import get_voc2010imdb
from tools.get_helenimdb import get_helenimdb
from tools.get_celebaimdb import get_celebaimdb
from tools.get_cubsampleimdb import get_cubsampleimdb
from tools.lib import *


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __getitem__(self,index):
        img = torch.from_numpy(self.data['image'][index,:,:,:]).float()
        label = torch.from_numpy(self.data['label'][index,:,:,:]).float()

        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return self.data['label'].shape[0]


def get_density(label):
    if label.shape[1]>1:
        label = torch.from_numpy(label[:,:,0,0])
        density = torch.mean((label>0).float(),0)
    else:
        density = torch.Tensor([0])
    return density


def download_dataset(datasets_path,dataset_path,dataset):
    downloadpath_ilsvrcanimalpart = "https://github.com/zqs1022/detanimalpart.git"
    downloadpath1_vocpart = "http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/trainval.tar.gz"
    downloadpath2_vocpart = "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar"
    downloadpath_cub200 = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"


    if os.path.exists(dataset_path) == False:
        if dataset == "ilsvrcanimalpart":
            os.system(" git clone " + downloadpath_ilsvrcanimalpart + " " + dataset_path)
            os.system(" unzip " + os.path.join(dataset_path, 'detanimalpart-master.zip') + ' -d ' + dataset_path)
        elif dataset == "vocpart":
            os.system(" wget -O " + dataset_path + " --no-check-certificate " + downloadpath1_vocpart)
            os.system(" wget -O " + dataset_path + " --no-check-certificate " + downloadpath2_vocpart)
            os.system(" tar -xvzf "+ dataset_path + '/trainval.tar.gz')
            os.system(" tar -xvf " + dataset_path + '/VOCtrainval_03-May-2010.tar')
        elif dataset == "cub200":
            os.system(" wget -O " + dataset_path + " --no-check-certificate " + downloadpath_cub200)
            os.system(" tar -xvzf "+ dataset_path + '/CUB_200_2011.tgz')
        else:
            print("error: no target dataset!")
            os.exit(0)


def get_imdb(root_path,imdb_path,dataset_path,dataset,imagesize,label_name):
    neg_path = os.path.join(root_path, 'datasets', 'neg')
    imdb_train_path = os.path.join(imdb_path, 'train.mat')
    imdb_val_path = os.path.join(imdb_path, 'val.mat')
    imdb_mean_path = os.path.join(imdb_path, 'mean.mat')
    if os.path.exists(imdb_train_path) == False:
        if (dataset == "ilsvrcanimalpart"):
            imdb_train, imdb_val, imdb_mean = get_ilsvrimdb(dataset_path, neg_path, label_name,imagesize)
            save_imdb(imdb_train_path, imdb_train ,'train') # image: type:numpy size(3596, 3, 224, 224) ; label type:numpy size(3596, 1, 1, 1)
            save_imdb(imdb_val_path, imdb_val, 'val') # image: type:numpy size(404, 3, 224, 224) ; label type:numpy size(404, 1, 1, 1)
            save_imdb(imdb_mean_path, imdb_mean, 'mean') #mean type:numpy size:(224,224,3) ;
        elif dataset == "vocpart":
            imdb_train, imdb_val, imdb_mean = get_vocimdb(root_path, dataset, dataset_path, neg_path, label_name,imagesize)
            save_imdb(imdb_train_path, imdb_train,'train')
            save_imdb(imdb_val_path, imdb_val,'val')
            save_imdb(imdb_mean_path, imdb_mean, 'mean')
        elif dataset == "cub200":
            imdb_train, imdb_val, imdb_mean = get_cubimdb(dataset_path, neg_path, label_name,imagesize)
            save_imdb(imdb_train_path, imdb_train,'train')
            save_imdb(imdb_val_path, imdb_val,'val')
            save_imdb(imdb_mean_path, imdb_mean, 'mean')
        elif dataset == "voc2010_crop":
            imdb_train, imdb_val, imdb_mean = get_voc2010imdb(root_path, dataset, dataset_path, label_name, imagesize)
            save_imdb(imdb_train_path, imdb_train,'train')
            save_imdb(imdb_val_path, imdb_val,'val')
            save_imdb(imdb_mean_path, imdb_mean, 'mean')
        elif dataset == "cubsample":
            imdb_train, imdb_val, imdb_mean = get_cubsampleimdb(root_path, dataset, dataset_path, 'cubsample', imagesize)
            save_imdb(imdb_train_path, imdb_train,'train')
            save_imdb(imdb_val_path, imdb_val,'val')
            save_imdb(imdb_mean_path, imdb_mean, 'mean')
        elif dataset == "helen":
            imdb_train, imdb_val, imdb_mean = get_helenimdb(root_path, dataset, dataset_path, 'helen', imagesize)
            save_imdb(imdb_train_path, imdb_train,'train')
            save_imdb(imdb_val_path, imdb_val,'val')
            save_imdb(imdb_mean_path, imdb_mean, 'mean')
        elif dataset == "celeba":
            imdb_train, imdb_val, imdb_mean = get_celebaimdb(root_path, dataset, dataset_path, imagesize)
            save_imdb(imdb_train_path, imdb_train,'train')
            save_imdb(imdb_val_path, imdb_val,'val')
            save_imdb(imdb_mean_path, imdb_mean, 'mean')
    else:
        imdb_train = load_imdb(imdb_train_path, 'train')
        imdb_val = load_imdb(imdb_val_path, 'val')

    return imdb_train, imdb_val


def load_data(root_path, imdb_path, args):
    datasets_path = os.path.join(root_path,'datasets')
    # Check if you need to download the dataset
    download_dataset(datasets_path,datasets_path,args.dataset)
    # Check if you need to generate the imdb
    imdb_train, imdb_val = get_imdb(root_path,imdb_path,datasets_path,args.dataset,args.imagesize,args.label_name)
    density = get_density(np.concatenate((imdb_train['label'], imdb_val['label']), axis=0))
    train_dataset = MyDataset(imdb_train, transform=None)
    val_dataset = MyDataset(imdb_val, transform=None)
    train_dataloader = DataLoader(train_dataset, args.batchsize, shuffle=True)
    val_dataloader = DataLoader(val_dataset, args.batchsize, shuffle=True)
    dataset_length = {'train': len(train_dataset), 'val': len(val_dataset)}
    return train_dataloader, val_dataloader, density, dataset_length