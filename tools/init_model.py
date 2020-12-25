import os
import numpy as np
from model.vgg_vd_16.vgg_vd_16 import vgg_vd_16
from model.alexnet.alexnet import alexnet
from model.vgg_m.vgg_m import vgg_m
from model.vgg_s.vgg_s import vgg_s
from model.resnet_18.resnet_18 import resnet_18
from model.resnet_50.resnet_50 import resnet_50
from model.densenet_121.densenet_121 import densenet_121

def get_net(model, model_path, label_num, dropoutrate, losstype):
    if(model == "vgg_vd_16"):
        pretrain_path = os.path.join(model_path,model+".mat")
        net = vgg_vd_16(pretrain_path, label_num, dropoutrate, losstype)
    elif(model == 'alexnet'):
        pretrain_path = os.path.join(model_path, model + ".mat")
        net = alexnet(pretrain_path, label_num, dropoutrate, losstype)
    elif(model == 'vgg_m'):
        pretrain_path = os.path.join(model_path, model + ".mat")
        net = vgg_m(pretrain_path, label_num, dropoutrate, losstype)
    elif(model == 'vgg_s'):
        pretrain_path = os.path.join(model_path, model + ".mat")
        net = vgg_s(pretrain_path, label_num, dropoutrate, losstype)
    elif (model == 'resnet_18'):
        pretrain_path = os.path.join(model_path, model + ".pth")
        net = resnet_18(pretrain_path, label_num, dropoutrate, losstype)
    elif (model == 'resnet_50'):
        pretrain_path = os.path.join(model_path, model + ".pth")
        net = resnet_50(pretrain_path, label_num, dropoutrate, losstype)
    elif (model == 'densenet_121'):
        pretrain_path = os.path.join(model_path, model + ".pth")
        net = densenet_121(pretrain_path, label_num, dropoutrate, losstype)
    return net


def download_pretrain(model,model_path):
    download_vgg_vd_16_path = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat"
    download_vgg_m_path = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat"
    download_vgg_s_path = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-s.mat"
    download_alexnet_path = "http://www.vlfeat.org/matconvnet/models/imagenet-caffe-alex.mat"
    download_resnet_18_path = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
    download_resnet_50_path = "https://download.pytorch.org/models/resnet50-19c8e357.pth"
    download_densenet_121_path = "https://download.pytorch.org/models/densenet121-a639ec97.pth"

    if (model == "vgg_vd_16"):
        pretrain_path = os.path.join(model_path, model+".mat")
        if os.path.exists(pretrain_path) == False:
            os.system(" wget -O " + pretrain_path + " --no-check-certificate " + download_vgg_vd_16_path)
    elif model == "alexnet":
        pretrain_path = os.path.join(model_path, model + ".mat")
        if os.path.exists(pretrain_path) == False:
            os.system(" wget -O " + pretrain_path + " --no-check-certificate " + download_alexnet_path)
    elif model == "vgg_m":
        pretrain_path = os.path.join(model_path, model + ".mat")
        if os.path.exists(pretrain_path) == False:
            os.system(" wget -O " + pretrain_path + " --no-check-certificate " + download_vgg_m_path)
    elif model == "vgg_s":
        pretrain_path = os.path.join(model_path, model + ".mat")
        if os.path.exists(pretrain_path) == False:
            os.system(" wget -O " + pretrain_path + " --no-check-certificate " + download_vgg_s_path)
    elif model == "resnet_18":
        pretrain_path = os.path.join(model_path, model + ".pth")
        if os.path.exists(pretrain_path) == False:
            os.system(" wget -O " + pretrain_path + " --no-check-certificate " + download_resnet_18_path)
    elif model == "resnet_50":
        pretrain_path = os.path.join(model_path, model + ".pth")
        if os.path.exists(pretrain_path) == False:
            os.system(" wget -O " + pretrain_path + " --no-check-certificate " + download_resnet_50_path)
    elif model == "densenet_121":
        pretrain_path = os.path.join(model_path, model + ".pth")
        if os.path.exists(pretrain_path) == False:
            os.system(" wget -O " + pretrain_path + " --no-check-certificate " + download_densenet_121_path)
    else:
        print("error: no target model!")
        os.exit(0)


def init_model(root_path,args):
    model_path = os.path.join(root_path, 'model', args.model)
    download_pretrain(args.model,model_path)
    net = get_net(args.model, model_path, args.label_num, args.dropoutrate, args.losstype)
    return net


