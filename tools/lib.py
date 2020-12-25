import os
import cv2
import torch
import h5py
import pandas as pd
import numpy as np



def init_lr(model,label_num,losstype):
    if model == "alexnet":
        lrMag = 30 * label_num
    elif model == "vgg_vd_16":
        lrMag = label_num
    elif model == "vgg_m":
        lrMag = 30 * label_num
    elif model == "vgg_s":
        lrMag = 10 * label_num
    elif model == "resnet_18":
        lrMag = 10*label_num
    elif model == "resnet_50":
        lrMag = 10*label_num
    elif model == "densenet_121":
        lrMag = 10*label_num
    else:
        print("error: no target model!")
        os.exit(0)

    if label_num == 1:
        lr = np.logspace(-4, -4, 1000)
        lr = lr[0:500]*lrMag
    else:
        lr = np.logspace(-4, -5, 80)*lrMag
    # mag need to be modified for multiple classifications
    if losstype == 'softmax':
        if label_num > 10:
            lr = lr/50
            if model == 'vgg_m':
                lr = lr/10
        else:
            lr = lr/5
    else:
        if label_num > 10:
            lr = lr/10
            if model == 'vgg_m':
                lr = lr*2
    epochnum = len(lr)
    return lr, epochnum


def cv2imread(path):
    I = cv2.imread(path)
    channel_1 = I[:,:,0]
    channel_3 = I[:,:,2]
    I[:,:,2] = channel_1
    I[:,:,0] = channel_3
    return I


def make_dir(dir_path):
    if os.path.exists(dir_path) == False:
        os.mkdir(dir_path)


def save_imdb(imdb_path,imdb,type):
    f = h5py.File(imdb_path, "w")
    if type == 'mean':
        f.create_dataset('mean', data=imdb['mean'])
    else:
        f.create_dataset('image', data=imdb['image'])
        f.create_dataset('label', data=imdb['label'])
    f.close()


def load_imdb(mean_path,type):
    f = h5py.File(mean_path, "r")
    if type == 'mean':
        imdb = f['mean']
    else:
        imdb = f
    return imdb

# load trained model
def load_model(model_path):
    model = torch.load(model_path)
    return model


def load_csv(csv_path):
    data = pd.read_csv(csv_path)
    vector = np.array(data)
    vector = vector.squeeze(1)
    return vector

def load_txt(txt_path,type):
    if type == "int_int":
        res = []
        with open(txt_path,'r') as data:
            for each_line in data:
                temp = each_line.split()
                temp[0] = int(temp[0])
                temp[1] = int(temp[1])
                res.append(temp)
        res = np.array(res)
    elif type == "int_str":
        res = {'id':[],'id_name':[]}
        with open(txt_path,'r') as data:
            for each_line in data:
                temp = each_line.split()
                temp[0] = int(temp[0])
                res['id'].append(temp[0])
                res['id_name'].append(temp[1])
    else:
        res = []
        with open(txt_path, 'r') as data:
            for each_line in data:
                temp = each_line.split()
                temp[0] = float(temp[1])
                temp[1] = float(temp[2])
                temp[2] = float(temp[3])
                temp[3] = float(temp[4])
                res.append(temp[0:4])
        res = np.array(res)
    return res





