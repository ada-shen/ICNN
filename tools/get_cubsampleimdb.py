import os
import cv2
import h5py
import matplotlib.image as mpimg
import math
import numpy as np

# different get+dataset+imdb.py has deffernet readAnnotation getNegObjSet and getI functions

def getI(obj,image_size, IsFlip):
    I = cv2.imread(obj["filename"])
    if(len(I.shape)==2):
        I = np.expand_dims(I,axis=2)
        I = np.repeat(I,3,axis=2)
    I = I.astype(np.float32)
    I = cv2.resize(I,image_size,interpolation=cv2.INTER_LINEAR)
    if(IsFlip==True):
        I = I[:,::-1,:]
    return I

def readAnnotation(root_path, dataset, dataset_path, categoryName):
    dataset_file = os.path.join(dataset_path,dataset, categoryName+'_info.txt')

    dataset_img_path = os.path.join(dataset_path,dataset)
    print(dataset_file)
    print(dataset_img_path)
    f = open(dataset_file,'r')
    objset_train_true=[]
    objset_train_false = []
    objset_val_true = []
    objset_val_false = []

    for line in f.readlines():
        words = line.split(',')
        path = words[1]
        if int(words[2])==0 and int(words[3])==1:
            objset_train_true.append({'filename':dataset_img_path+'/'+path})
        if int(words[2])==0 and int(words[3])==0:
            objset_val_true.append({'filename':dataset_img_path+'/'+path})
        if int(words[2])==1 and int(words[3])==1:
            objset_train_false.append({'filename':dataset_img_path+'/'+path})
        if int(words[2])==1 and int(words[3])==0:
            objset_val_false.append({'filename':dataset_img_path+'/'+path})

    return objset_train_true,objset_train_false,objset_val_true,objset_val_false

def get_cubsampleimdb(root_path, dataset, dataset_path,categoryName,image_size):
    objset_train_true,objset_train_false,objset_val_true,objset_val_false = readAnnotation(root_path, dataset, dataset_path,categoryName)
    train_true_l = len(objset_train_true)
    train_false_l = len(objset_train_false)
    val_true_l = len(objset_val_true)
    val_false_l = len(objset_val_false)


    data_train_true = np.zeros((image_size,image_size,3,train_true_l))
    data_train_false = np.zeros((image_size,image_size,3,train_false_l))
    data_val_true = np.zeros((image_size,image_size,3,val_true_l))
    data_val_false = np.zeros((image_size,image_size,3,val_false_l))

    for i in range(train_true_l):
        tar= i
        IsFlip = False
        I = getI(objset_train_true[i],(image_size,image_size), IsFlip)
        data_train_true[:,:,:,tar]=I

    for i in range(train_false_l):
        tar = i
        IsFlip = False
        I = getI(objset_train_false[i], (image_size,image_size), IsFlip)
        data_train_false[:,:,:,tar]=I

    for i in range(val_true_l):
        tar = i
        IsFlip = False
        I = getI(objset_val_true[i],(image_size,image_size), IsFlip)
        data_val_true[:,:,:,tar]=I

    for i in range(val_false_l):
        tar = i
        IsFlip = False
        I = getI(objset_val_false[i], (image_size,image_size), IsFlip)
        data_val_false[:,:,:,tar]=I

    train_label = np.ones((1,train_true_l+train_false_l))*(-1)
    train_label[:,0:train_true_l] = 1

    val_label = np.ones((1,val_true_l+val_false_l))*(-1)
    val_label[:,0:val_true_l] = 1

    train_data = np.concatenate((data_train_true,data_train_false), axis=3)
    val_data = np.concatenate((data_val_true, data_val_false), axis=3)

    dataMean = np.mean(train_data[:,:,:,:],axis=3)
    imdb_mean = {'mean': dataMean}
    dataMean = np.expand_dims(dataMean, axis=3)
    dataMean_train = np.tile(dataMean,(1,1,1,train_true_l+train_false_l))
    dataMean_val = np.tile(dataMean, (1, 1, 1, val_true_l+val_false_l))
    train_data = train_data-dataMean_train
    val_data = val_data-dataMean_val

    data_train = train_data.transpose(3,2,0,1)
    label_train = train_label[np.newaxis, :, :]
    label_train = label_train[np.newaxis,:,:,:]
    label_train = label_train.transpose(3,1,2,0)
    imdb_train = {'image': data_train, 'label': label_train}

    data_val = val_data.transpose(3,2,0,1)
    label_val = val_label[np.newaxis, :, :]
    label_val = label_val[np.newaxis, :,:, :]
    label_val = label_val.transpose(3, 1, 2, 0)
    imdb_val = {'image': data_val, 'label': label_val}


    return imdb_train, imdb_val, imdb_mean














