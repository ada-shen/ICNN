import os
import h5py
import numpy as np


def getDiff(pos,truth,imgID,invalidNum):
    partNum = len(truth)
    patNum = pos.shape[1]
    diff = np.zeros((2,partNum,patNum))
    for i in range(partNum):
        truth_center = truth[i]['truth']['pHW_center']
        pos_truth = truth[i][truth_center[imgID][0]][0]
        if (type(pos_truth) is not np.ndarray or len(pos_truth)>2):
            diff[:,i,:] = invalidNum
        else:
            tmp = np.reshape(pos_truth, (2, 1))
            pos_truth = np.tile(tmp, (1, patNum))
            diff[:,i,:] = np.reshape((pos_truth - pos),(2,patNum))
    return diff

def new_var(array):
    array_mean = np.mean(array)
    res = 0
    for i in range(len(array)):
        res = res + (array[i]-array_mean)*(array[i]-array_mean)
    return res/(len(array)-1)


def getAvgDistSqrtVar(diff,prob,patchNumPerPattern,invalidNum):
    partNum = diff.shape[1]
    patNum = diff.shape[2]
    distSqrtVar = np.zeros((patNum, 1))
    for pat in range(patNum):
        idx = np.argsort(-1 * prob[pat,:]) #1dim
        for partID in range(partNum):
            tmp = np.where(diff[0,partID,pat,:]==invalidNum)
            tmp = np.setdiff1d(idx, tmp, True)
            tmp = tmp[0:min(patchNumPerPattern, len(tmp))]
            if(len(tmp)<2):
                distSqrtVar[pat,0] = np.nan
            else:
                dist = np.reshape(np.sqrt(np.sum((diff[:,partID, pat, tmp]*diff[:,partID, pat, tmp]),axis=0)),(1,len(tmp)))
                distSqrtVar[pat,0] = distSqrtVar[pat,0] + np.sqrt(new_var(dist[0,:]))
        distSqrtVar[pat,0] = distSqrtVar[pat,0]/partNum
    return distSqrtVar

def getDistSqrtVar(truthpart_path, pos, prob, patchNumPerPattern, partList, label_name):
    invalidNum=100000
    partNum=len(partList)
    truth=[]
    for i in range(partNum):
        partID = partList[i]
        file_path = os.path.join(truthpart_path, label_name, "truth_part" + str(0) + str(partID) + '.mat')
        f = h5py.File(file_path, 'r')
        truth.append(f)
    patNum = pos.shape[1]
    imgNum = pos.shape[2]
    diff = np.zeros((2,partNum,patNum,imgNum))
    for imgID in range(imgNum):
        diff[:,:,:,imgID] = getDiff(pos[:,:,imgID],truth,imgID,invalidNum)

    distSqrtVar = getAvgDistSqrtVar(diff,prob,patchNumPerPattern,invalidNum)
    return distSqrtVar