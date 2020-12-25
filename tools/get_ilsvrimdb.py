import os
import cv2
import h5py
import matplotlib.image as mpimg
import math
import numpy as np
from tools.lib import cv2imread

# different get+dataset+imdb.py has deffernet readAnnotation getNegObjSet and getI functions


def readAnnotation(dataset_path,categoryName):
    minArea = 2500
    dataset_data_path = dataset_path
    dataset_data_namebatch_bndbox_path = os.path.join(dataset_data_path, categoryName + "_obj", "img", "data.mat")
    dataset_data_namebatch_img_path = os.path.join(dataset_data_path, categoryName + "_obj", "img", "img", "")
    mat = h5py.File(dataset_data_namebatch_bndbox_path, 'r')
    samples = mat['samples']["obj"]
    files = os.listdir(dataset_data_namebatch_img_path)
    objset = []
    for i in range(len(files)):
        xmin = int(mat[(samples[i][0])]['bndbox']['xmin'].value)
        ymin = int(mat[(samples[i][0])]['bndbox']['ymin'].value)
        xmax = int(mat[(samples[i][0])]['bndbox']['xmax'].value)
        ymax = int(mat[(samples[i][0])]['bndbox']['ymax'].value)
        if ((xmax-xmin+1)*(ymax-ymin+1)>=minArea) == False:
            continue
        filename = ("%05d.jpg") % (i+1)
        objset.append({'filename':os.path.join(dataset_data_namebatch_img_path + filename),'bndbox':[xmin,xmax,ymin,ymax],'id':i+1})
    return objset


def getNegObjSet(neg_path):
    MaxObjNum = 1000
    objset_neg = []
    for i in range(MaxObjNum):
        filename = ("%05d.JPEG") % (i+1)
        img = mpimg.imread(os.path.join(neg_path,filename))
        h = img.shape[0]
        w = img.shape[1]
        xmin = 1
        ymin = 1
        xmax = w
        ymax = h
        objset_neg.append({'filename': os.path.join(neg_path,filename), 'bndbox':[xmin,xmax,ymin,ymax],'id': i + 1000000001})
    return objset_neg


def getI(obj,image_size, IsFlip):
    I = cv2imread(obj["filename"])
    if(len(I.shape)==2):
        I = np.expand_dims(I,axis=2)
        I = np.repeat(I,3,axis=2)
    h = I.shape[0]
    w = I.shape[1]
    d = I.shape[2]
    xmin = max(obj['bndbox'][0],1)
    xmax = min(obj['bndbox'][1],w)
    ymin = max(obj['bndbox'][2],1)
    ymax = min(obj['bndbox'][3],h)
    I_patch = I[ymin-1:ymax-1,xmin-1:xmax-1,:]
    I_patch = I_patch.astype(np.float32)
    I_patch = cv2.resize(I_patch,image_size,interpolation=cv2.INTER_LINEAR)
    if(IsFlip==True):
        I_patch = I_patch[:,::-1,:]
    return I_patch, I


def get_ilsvrimdb(dataset_path,neg_path,categoryName,image_size):
    trainRate = 0.9
    posRate = 0.75
    objset = readAnnotation(dataset_path,categoryName)
    objset_neg = getNegObjSet(neg_path)
    num_pos = len(objset)
    num_neg = len(objset_neg)
    tarN = round(posRate/(1-posRate)*num_neg)
    repN = math.ceil(tarN/(num_pos*2))
    list_train_pos = np.round(np.linspace(0,num_pos-1,round(num_pos*trainRate))) #
    list_train_pos = np.append(2*list_train_pos-1, 2*list_train_pos)
    if (repN > 1):
        repN_tmp = np.array(range(repN))
        repN_tmp = np.expand_dims(repN_tmp,axis=1)
        list_train_pos = np.tile(list_train_pos,(repN, 1)) + np.tile(repN_tmp*(num_pos*2),(1,list_train_pos.size))
        list_train_pos = np.reshape(list_train_pos,(1,list_train_pos.size))
        list_train_pos = np.sort(list_train_pos,axis=1)
    list_train_pos = list_train_pos[np.where(list_train_pos<=tarN)]
    list_train_neg = np.round(np.linspace(0,num_neg-1,round(num_neg*trainRate)))
    list_train = np.sort(np.append(list_train_pos, list_train_neg + tarN)) # 1dim
    list_train = list_train-1 # start_index is 0 in python, is 1 in matlab

    data = np.zeros((image_size,image_size,3,num_pos*2))
    data_neg = np.zeros((image_size,image_size,3,num_neg))
    for i in range(num_pos):
        tar=(i+1)*2-2
        IsFlip = False
        I_patch,I = getI(objset[i],(image_size,image_size), IsFlip)
        data[:,:,:,tar]=I_patch
        tar = (i+1)*2-1
        IsFlip = True
        I_patch,I = getI(objset[i],(image_size,image_size), IsFlip)
        data[:,:,:,tar]=I_patch
    data = np.tile(data, (1, 1, 1, repN))
    data = data[:,:,:, 0:tarN]
    for i in range(num_neg):
        tar = i
        IsFlip = False
        I_patch,I = getI(objset_neg[i], (image_size,image_size), IsFlip)
        data_neg[:,:,:,tar]=I_patch

    total_images = tarN + num_neg
    labels = np.ones((1,total_images))*(-1)
    labels[:,0:tarN] = 1
    set = np.ones((1,total_images))*2
    set[:, list_train.astype(int)] = 1
    data = np.concatenate((data,data_neg), axis=3)
    dataMean = np.mean(data[:,:,:,list_train.astype(int)],axis=3) #depend on training data not truth data
    imdb_mean = {'mean': dataMean}
    dataMean = np.expand_dims(dataMean,axis=3)
    dataMean = np.tile(dataMean,(1,1,1,total_images))
    data = data-dataMean
    set = np.squeeze(set,axis=0)

    data_train = data[:, :, :, np.where(set == 1)]
    data_train = np.squeeze(data_train,axis=3)
    data_train = data_train.transpose(3,2,0,1)
    label_train = labels[:, np.where(set == 1)]
    label_train = label_train[np.newaxis,:,:,:]
    label_train = label_train.transpose(3,1,2,0)
    imdb_train = {'image': data_train, 'label': label_train}


    data_val = data[:, :, :, np.where(set == 2)]
    data_val = np.squeeze(data_val, axis=3)
    data_val = data_val.transpose(3,2,0,1)
    label_val = labels[:, np.where(set == 2)]
    label_val = label_val[np.newaxis, :, :, :]
    label_val = label_val.transpose(3, 1, 2, 0)
    imdb_val = {'image': data_val, 'label': label_val}

    return imdb_train, imdb_val, imdb_mean














