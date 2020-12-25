import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tools.get_ilsvrimdb import get_ilsvrimdb
from tools.get_vocimdb import get_vocimdb
from tools.get_cubimdb import get_cubimdb
from tools.get_ilsvrimdb import readAnnotation as ilsvr_readAnnotation
from tools.get_cubimdb import readAnnotation as cub_readAnnotation
from tools.get_vocimdb import readAnnotation as voc_readAnnotation
from tools.lib import *
from tools.load_data import download_dataset
from tools.load_data import get_density
from tools.load_data import MyDataset

def get_imdb(root_path,imdb_path,dataset_path,dataset,imagesize,label_name):
    neg_path = os.path.join(root_path, 'datasets', 'neg')
    imdb_train_path = os.path.join(imdb_path, label_name + '_train.mat')
    imdb_val_path = os.path.join(imdb_path, label_name +'_val.mat')
    imdb_mean_path = os.path.join(imdb_path, label_name+ '_mean.mat')
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
            imdb_train, imdb_val, imdb_mean = get_cubimdb(dataset_path, neg_path, label_name, imagesize)
            save_imdb(imdb_train_path, imdb_train, 'train')
            save_imdb(imdb_val_path, imdb_val, 'val')
            save_imdb(imdb_mean_path, imdb_mean, 'mean')

def get_imdb_multi(root_path,dataset_path,imdb_path,dataset,label_num,label_name):
    imdb_train_path = os.path.join(imdb_path, 'train.mat')
    imdb_val_path = os.path.join(imdb_path, 'val.mat')
    imdb_mean_path = os.path.join(imdb_path, 'mean.mat')
    if os.path.exists(imdb_train_path) == False:
        trainRate = 0.9
        if (label_num > 10):
            maxSampleNum = 400
            minSampleNum = 100
        else:
            maxSampleNum = 1000000
            minSampleNum = 1500
        for i in range(label_num):
            tempimdb_train_path = os.path.join(imdb_path, label_name[i] + '_train.mat')
            tempimdb_val_path = os.path.join(imdb_path, label_name[i] + '_val.mat')
            tempimdb_mean_path = os.path.join(imdb_path, label_name[i] + '_mean.mat')
            tempimdb_train = load_imdb(tempimdb_train_path, 'train')
            tempimdb_val = load_imdb(tempimdb_val_path, 'val')
            tempimdb_img = np.concatenate((tempimdb_train['image'],tempimdb_val['image']),axis=0)
            tempimdb_label = np.concatenate((tempimdb_train['label'],tempimdb_val['label']),axis=0)
            tempimdb_mean = load_imdb(tempimdb_mean_path, 'mean')
            tempimdb_mean = np.transpose(tempimdb_mean,(2,1,0))
            tempimdb_mean = tempimdb_mean[np.newaxis,:,:,:]
            if (dataset == "ilsvrcanimalpart"):
                objset = ilsvr_readAnnotation(dataset_path, label_name[i])
            elif dataset == "vocpart":
                objset = voc_readAnnotation(root_path, dataset, dataset_path, label_name)
            elif dataset == "cub200":
                objset = cub_readAnnotation(dataset_path, label_name)
            List = np.where(tempimdb_label == 1)
            List = List[0]
            List = List[0:(len(objset)*2)]
            if(len(List) < minSampleNum):
                List = np.tile(List,int(np.ceil(minSampleNum/len(List))))
                List = List[0:minSampleNum]
            List = List[0:min(len(List),maxSampleNum)]
            tempimdb_img = tempimdb_img[List,:,:,:]
            tempimdb_label = tempimdb_label[List,:,:,:]
            img_num = len(List)
            tempimdb_mean = np.repeat(tempimdb_mean,img_num,axis=0)
            tempimdb_img = tempimdb_img - tempimdb_mean
            if(i==0):
                imdb_img = tempimdb_img
                imdb_label = np.ones((img_num,label_num,1,1)) * (-1)
                imdb_label[:,0,0,0] = tempimdb_label[:,0,0,0]
            else:
                imdb_img = np.concatenate((imdb_img,tempimdb_img),axis=0)
                imdb_label = np.concatenate((imdb_label,np.ones((img_num,label_num,1,1)) * (-1)),axis=0)
                imdb_label[(imdb_label.shape[0]-tempimdb_label.shape[0]):imdb_label.shape[0],i,0,0] = tempimdb_label[:,0,0,0]
        num = imdb_img.shape[0]
        List_train = np.round(np.linspace(0,num-1,round(num*trainRate)))
        set = np.ones((1, num)) * 2
        set[:, List_train.astype(int)] = 1
        dataMean = np.mean(imdb_img[List_train.astype(int), :, :, :], axis=0) # shape = 3,224,224
        imdb_mean = {'mean': np.transpose(dataMean,(2,1,0))}
        dataMean = dataMean[np.newaxis, :, :, :]
        dataMean = np.repeat(dataMean, num, axis=0)
        imdb_img = imdb_img - dataMean
        set = np.squeeze(set, axis=0)

        data_train = imdb_img[np.where(set == 1), :, :, :]
        data_train = np.squeeze(data_train,axis=0)
        label_train = imdb_label[np.where(set == 1), :, :, :]
        label_train = np.squeeze(label_train, axis=0)
        imdb_train = {'image': data_train, 'label': label_train}

        data_val = imdb_img[np.where(set == 2),:, :, :]
        data_val = np.squeeze(data_val,axis=0)
        label_val = imdb_label[np.where(set == 2),:,:,:]
        label_val = np.squeeze(label_val, axis=0)
        imdb_val = {'image': data_val, 'label': label_val}

        save_imdb(imdb_train_path, imdb_train,'train')  # image: type:numpy size(3596, 3, 224, 224) ; label type:numpy size(3596, 1, 1, 1)
        save_imdb(imdb_val_path, imdb_val,'val')  # image: type:numpy size(404, 3, 224, 224) ; label type:numpy size(404, 1, 1, 1)
        save_imdb(imdb_mean_path, imdb_mean, 'mean')  # mean type:numpy size:(224,224,3) ;
    else:
        imdb_train = load_imdb(imdb_train_path, 'train')
        imdb_val = load_imdb(imdb_val_path, 'val')
    return imdb_train, imdb_val


def load_data_multi(root_path, imdb_path, args):
    datasets_path = os.path.join(root_path, 'datasets')
    dataset_path = os.path.join(datasets_path, args.dataset)
    # Check if you need to download the dataset
    download_dataset(datasets_path,dataset_path,args.dataset)
    # Check if you need to generate the imdb
    for i in range(args.label_num):
        get_imdb(root_path,imdb_path,dataset_path,args.dataset,args.imagesize,args.label_name[i])
    imdb_train, imdb_val = get_imdb_multi(root_path,dataset_path,imdb_path,args.dataset,args.label_num,args.label_name)
    density = get_density(np.concatenate((imdb_train['label'], imdb_val['label']), axis=0))
    train_dataset = MyDataset(imdb_train, transform=None)
    val_dataset = MyDataset(imdb_val, transform=None)
    train_dataloader = DataLoader(train_dataset, args.batchsize, shuffle=True)
    val_dataloader = DataLoader(val_dataset, args.batchsize, shuffle=False)
    dataset_length = {'train': len(train_dataset), 'val': len(val_dataset)}
    return train_dataloader, val_dataloader, density, dataset_length