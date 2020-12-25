import torch
import torch.nn.functional as F
from tools.get_ilsvrimdb import getI

def get_x(net, im, label, Iter, density,model):
    if model == "vgg_vd_16":
        x = net.conv1(im)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool1(x)
        x = net.conv2(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool2(x)
        x = net.conv3(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool3(x)
        x = net.conv4(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool4(x)
        x = net.conv5(x)
        x = net.mask1[0](x, label, Iter, density)
        x = net.maxpool5[0](x)
        x = x.cpu().clone().data.numpy()
    elif model == "alexnet":
        x = net.conv1(im)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool1(x)

        x = net.conv2(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool2(x)

        x = net.conv3(x)
        x = net.mask1[0](x, label, Iter, density)
        x = net.maxpool3(x)
    elif model == "vgg_s":
        x = net.conv1(im)
        x = F.pad(x, (0, 2, 0, 2))
        x = net.maxpool1(x)

        x = net.conv2(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool2(x)

        x = net.conv3(x)
        x = net.mask1[0](x, label, Iter, density)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool3(x)
    elif model == "vgg_m":
        x = net.conv1(im)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool1(x)

        x = net.conv2(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = net.maxpool2(x)

        x = net.conv3(x)
        x = net.mask1[0](x, label, Iter, density)
        x = net.maxpool3(x)
    elif model == "resnet_18":
        x = net.pad2d_3(im)  # new padding
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.pad2d_1(x)
        x = net.maxpool(x)

        x = net.layer1(x)
        x = net.layer2(x)
        x = net.layer3(x)
        x = net.mask1[0](x, label, Iter, density)
        # f_map = x.detach()

    elif model == "resnet_50":
        x = net.pad2d_3(im)  # new padding
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.pad2d_1(x)
        x = net.maxpool(x)

        x = net.layer1(x)
        x = net.layer2(x)
        x = net.layer3(x)
        x = net.mask1[0](x, label, Iter, density)
        # f_map = x.detach()
    return x

def getCNNFeature(dataset_path, obj, net, isFlip, dataMean,epochnum, model):

    if "ilsvrcanimalpart" in dataset_path:
        I = getI(obj, (224,224), isFlip)
    elif "vocpart" in dataset_path:
        I = getI(obj, (224,224), isFlip)
    elif "cub200" in dataset_path:
        I = getI(obj, (224,224), isFlip)

    im = I[0] - dataMean
    im = torch.from_numpy(im).float()
    im = im.unsqueeze(3)
    im = im.permute(3, 2, 0, 1)
    label = torch.ones((1, 1, 1, 1))
    im = im.cuda()
    label = label.cuda()
    net = net.cuda()
    Iter = torch.Tensor([epochnum])
    density = torch.Tensor([0])
    x = get_x(net, im, label, Iter, density,model) # type numpy
    return x, I[0]



