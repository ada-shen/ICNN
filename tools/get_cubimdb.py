import os
import cv2
import h5py
import matplotlib.image as mpimg
import math
import numpy as np
from tools.lib import cv2imread
from tools.lib import load_txt
from tools.get_ilsvrimdb import getNegObjSet
from tools.get_ilsvrimdb import getI

# different get+dataset+imdb.py has deffernet readAnnotation getNegObjSet and getI functions


def readAnnotation(dataset_path,categoryName):
    minArea = 2500
    dataset_data_path = os.path.join(dataset_path, "CUB_200_2011")
    dataset_data_image_path = os.path.join(dataset_data_path, "images")
    dataset_data_labeltxt_path = dataset_data_path + "/image_class_labels.txt"
    dataset_data_train_test_txt_path = dataset_data_path + "/train_test_split.txt"
    dataset_data_classes_txt_path = dataset_data_path + "/classes.txt"
    dataset_data_images_txt_path = dataset_data_path + "/images.txt"
    dataset_data_boundingboxes_txt_path = dataset_data_path + "/bounding_boxes.txt"

    idClassPair = load_txt(dataset_data_labeltxt_path,"int_int")
    if categoryName in ['1','2','3','4','5','6','7','8','9']:
        imgIds = np.where(idClassPair[:,1] == int(categoryName))
    else:
        imgIds = idClassPair[:,0]-1

    train_test_list = load_txt(dataset_data_train_test_txt_path,"int_int")
    train_list = np.where(train_test_list[:, 1] == 1)

    batchClassnamePair = load_txt(dataset_data_classes_txt_path,"int_str")

    idNamePair = load_txt(dataset_data_images_txt_path,"int_str")
    imgnames = []
    for i in range(len(imgIds)):
        imgnames.append(idNamePair['id_name'][imgIds[i]])

    idBndboxPair = load_txt(dataset_data_boundingboxes_txt_path,"int_int_int_int")
    x = idBndboxPair[imgIds,0]
    y = idBndboxPair[imgIds,1]
    width = idBndboxPair[imgIds,2]
    height = idBndboxPair[imgIds,3]

    objset = []
    for i in range(len(imgIds)):
        xmin = int(x[i])
        ymin = int(y[i])
        xmax = int(x[i]+width[i])
        ymax = int(y[i]+height[i])
        if ((xmax - xmin + 1) * (ymax - ymin + 1) >= minArea) == False:
            continue
        imgnames[i] = imgnames[i].split('.')
        filename = imgnames[i][1] + '.' + imgnames[i][2]
        name = batchClassnamePair["id_name"][int(imgnames[i][0])-1]
        objset.append(
            {'filename': dataset_data_image_path + '/' + imgnames[i][0]+'.' + filename, 'name':name, 'bndbox': [xmin, xmax, ymin, ymax],
             'id': i + 1})
    return objset, train_list

def get_cubimdb(dataset_path,neg_path,categoryName,image_size):

    objset,trainList = readAnnotation(dataset_path,categoryName)
    objset_neg = getNegObjSet(neg_path)
    objset_neg = [val for val in objset_neg for i in range(4)]

    num_pos = len(objset)
    num_neg = len(objset_neg)
    data = np.zeros((image_size, image_size, 3, num_pos))
    data_neg = np.zeros((image_size, image_size, 3, num_neg))


    for i in range(num_pos):
        tar=i
        IsFlip = False
        I_patch,I = getI(objset[i],(image_size,image_size), IsFlip)
        data[:,:,:,tar]=I_patch


    for i in range(num_neg):
        tar = i
        IsFlip = False
        I_patch,I = getI(objset_neg[i], (image_size,image_size), IsFlip)
        data_neg[:,:,:,tar]=I_patch

    total_images = num_pos + num_neg
    labels = np.ones((1,total_images))*(-1)
    labels[:,0:num_pos] = 1

    list_train = np.where(labels==-1)
    tmp = range(round(num_neg * 0.5))
    list_train = list_train[1][tmp]
    list_train = np.append(trainList,list_train)

    set = np.ones((1,total_images))*2
    set[:, list_train] = 1
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














