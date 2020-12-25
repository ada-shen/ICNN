import os
import torch
import h5py
import numpy as np
from tools.getConvNetPara import getConvNetPara
from tools.computeStability import computeStability
from tools.computeStability_multi import computeStability_multi
from tools.lib import*


def step_computeStability(root_path,dataset,dataset_path, truthpart_path, label_name, net, model, layerID, epochnum, partList, partRate, imdb_mean):
    selectPatternRatio = 1.0
    patchNumPerPattern = 100
    if(label_name=='n01443537'):
        partList=[1]
    convnet = getConvNetPara(model)
    stability = computeStability(root_path,dataset,dataset_path, truthpart_path, label_name, net, model, convnet, layerID, epochnum, partList, partRate, imdb_mean, selectPatternRatio, patchNumPerPattern)
    return stability

def step_computeStability_multi(root_path,dataset,dataset_path, truthpart_path, label_name, net, model, layerID, epochnum, partList, imdb_mean):
    selectPatternRatio = 1.0
    patchNumPerPattern = 100

    convnet = getConvNetPara(model)
    for i in range(len(label_name)):
        if (label_name[i] == 'n01443537'):
            partList = [1]
        tmp,tmp_score = computeStability_multi(patchNumPerPattern, root_path,dataset,dataset_path, truthpart_path, label_name[i], net, model, convnet, layerID, epochnum,partList, imdb_mean)
        if i == 0:
            stability = np.zeros((len(tmp),len(label_name)))
            score = np.zeros((len(tmp),len(label_name)))
        stability[:,i] = np.squeeze(tmp,1)
        score[:,i] = tmp_score
    for i in range(len(tmp)):
        idx = np.argmax(score[i,:])
        stability[i, 0] = stability[i,idx]
    stability = stability[:,0]
    selectedPatternNum = round(len(stability) * selectPatternRatio)
    stability = np.sort(stability[np.isnan(stability) == 0])
    stability = np.mean(stability[0:min(selectedPatternNum, len(stability))])
    return stability



def getresult(root_path,dataset,taskid_path, imdb_path, dataset_path, truthpart_path, label_name, model, layerID, epochnum, partList):
    partRate = 1
    imdb_mean_path = os.path.join(imdb_path, 'mean.mat')
    imdb_mean = load_imdb(imdb_mean_path,'mean')
    sta = []

    net_path = os.path.join(taskid_path,"net-" + str(epochnum) +".pkl")
    net = load_model(net_path)
    stability = step_computeStability(root_path,dataset,dataset_path, truthpart_path, label_name, net, model, layerID, epochnum, partList, partRate, imdb_mean)
    return stability

def getresult_multi(root_path,dataset,taskid_path, imdb_path, dataset_path, truthpart_path, label_name, model, layerID, epochnum, partList):
    imdb_mean_path = os.path.join(imdb_path, 'mean.mat')
    imdb_mean = load_imdb(imdb_mean_path, 'mean')
    net_path = os.path.join(taskid_path, "net-" + str(epochnum) + ".pkl")
    net = load_model(net_path)
    stability = step_computeStability_multi(root_path,dataset,dataset_path, truthpart_path, label_name, net, model, layerID, epochnum, partList, imdb_mean)
    return stability

def showresult(epoch_num,taskid_path, imdb_path, root_path, args):
    if args.model in ['alexnet','vgg_m','vgg_s']:
        layerID = 6
    elif args.model in ['vgg_vd_16']:
        layerID = 14
    else:
        print('invalid model name')
        os._exit(1)

    if args.dataset == 'cub200':
        partList=[1, 6, 14]
    elif args.dataset == 'vocpart':
        partList = [1, 2, 3]
    elif args.dataset == 'ilsvrcanimalpart':
        partList = [1, 2]
    else:
        print('invalid dataset name')
        os._exit(1)

    datasets_path = os.path.join(root_path, 'datasets')
    dataset_path = os.path.join(datasets_path, args.dataset)

    truthpart_path = os.path.join(root_path, "data_input", args.dataset)
    if args.task_name == 'classification_multi':
        stability = getresult_multi(root_path,args.dataset,taskid_path, imdb_path, dataset_path, truthpart_path, args.label_name, args.model, layerID, epoch_num, partList)
    else:
        stability = getresult(root_path,args.dataset,taskid_path, imdb_path, dataset_path, truthpart_path, args.label_name, args.model, layerID, epoch_num, partList)
    print(stability)
    return stability




