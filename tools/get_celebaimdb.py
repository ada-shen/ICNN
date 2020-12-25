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

def readAnnotation(root_path, dataset, dataset_path):
    use_train = 162770
    use_test = 19868

    dataset_file = os.path.join(dataset_path,dataset, 'list_eval_partition.txt')
    label_file = os.path.join(dataset_path,dataset, 'Anno/list_attr_celeba.txt')
    dataset_img_path = os.path.join(dataset_path,dataset,'images')
    f = open(dataset_file,'r')
    g = open(label_file,'r')
    objset_train=[]
    objset_val = []
    train_label = []
    val_label = []
    g.readline()
    g.readline()

    train_num = 0
    test_num = 0
    for line in f.readlines():
        words = line.split()
        line1 = g.readline()
        words1 = line1.split()
        path = words[0]
        if path != words1[0]:
            print('filename error!')
        if int(words[1])==0 and train_num<use_train:
            objset_train.append({'filename':dataset_img_path+'/'+path})
            train_label.append(list(1 if int(i) > 0 else -1 for i in words1[1:]))
            train_num+=1
        if int(words[1])==1 and test_num<use_test:
            objset_val.append({'filename':dataset_img_path+'/'+path})
            val_label.append(list(1 if int(i) > 0 else -1 for i in words1[1:]))
            test_num += 1

    return objset_train,objset_val,train_label,val_label

def get_celebaimdb(root_path, dataset, dataset_path,image_size):
    objset_train,objset_val,train_label,val_label = readAnnotation(root_path, dataset, dataset_path)
    train_l = len(objset_train)
    val_l = len(objset_val)

    train_data = np.zeros((image_size,image_size,3,train_l))
    val_data = np.zeros((image_size,image_size,3,val_l))

    for i in range(train_l):
        tar= i
        IsFlip = False
        I = getI(objset_train[i],(image_size,image_size), IsFlip)
        train_data[:,:,:,tar]=I

    for i in range(val_l):
        tar = i
        IsFlip = False
        I = getI(objset_val[i],(image_size,image_size), IsFlip)
        val_data[:,:,:,tar]=I

    train_label = np.array(train_label)
    val_label = np.array(val_label)


    dataMean = np.mean(train_data[:,:,:,:],axis=3)
    imdb_mean = {'mean': dataMean}
    dataMean = np.expand_dims(dataMean, axis=3)
    dataMean_train = np.tile(dataMean,(1, 1, 1, train_l))
    dataMean_val = np.tile(dataMean, (1, 1, 1, val_l))
    train_data = train_data-dataMean_train
    val_data = val_data-dataMean_val

    data_train = train_data.transpose(3,2,0,1)
    label_train = train_label[np.newaxis, :, :]
    label_train = label_train[np.newaxis,:,:,:]
    label_train = label_train.transpose(2,3,0,1)
    imdb_train = {'image': data_train, 'label': label_train}

    data_val = val_data.transpose(3,2,0,1)
    label_val = val_label[np.newaxis, :, :]
    label_val = label_val[np.newaxis, :, :, :]
    label_val = label_val.transpose(2,3,0,1)
    imdb_val = {'image': data_val, 'label': label_val}


    return imdb_train, imdb_val, imdb_mean














