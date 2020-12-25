import torch

class Layer:
    def __init__(self, type, kernel_size, stride, padding):
        self.type = type
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

def vgg_16_vd():
    net = []
    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='pool', kernel_size=2, stride=2, padding=0)
    net.append(layer)

    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='pool', kernel_size=2, stride=2, padding=0)
    net.append(layer)

    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='pool', kernel_size=2, stride=2, padding=0)
    net.append(layer)

    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='pool', kernel_size=2, stride=2, padding=0)
    net.append(layer)

    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv_mask', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='pool', kernel_size=2, stride=2, padding=0)
    net.append(layer)

    layer = Layer(type='conv_mask', kernel_size=7, stride=1, padding=0)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='dropout', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=1, stride=1, padding=0)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='dropout', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=1, stride=1, padding=0)
    net.append(layer)

    return net

def alexnet():
    net = []
    layer = Layer(type='conv', kernel_size=11, stride=4, padding=0)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='lrn', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='pool', kernel_size=3, stride=2, padding=0)
    net.append(layer)

    layer = Layer(type='conv', kernel_size=5, stride=1, padding=2)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='lrn', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='pool', kernel_size=3, stride=2, padding=0)
    net.append(layer)

    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv_mask', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='pool', kernel_size=3, stride=2, padding=0)
    net.append(layer)

    layer = Layer(type='conv_mask', kernel_size=6, stride=1, padding=0)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='dropout', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=1, stride=1, padding=0)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='dropout', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=1, stride=1, padding=0)
    net.append(layer)

    return net

def vgg_m():
    net = []
    layer = Layer(type='conv', kernel_size=7, stride=2, padding=0)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='lrn', kernel_size=0, stride=0, padding=1)
    net.append(layer)
    layer = Layer(type='pool', kernel_size=3, stride=2, padding=0)
    net.append(layer)

    layer = Layer(type='conv', kernel_size=5, stride=2, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='lrn', kernel_size=0, stride=0, padding=1)
    net.append(layer)
    layer = Layer(type='pool', kernel_size=3, stride=2, padding=0)
    net.append(layer)

    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv_mask', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='pool', kernel_size=3, stride=2, padding=0)
    net.append(layer)

    layer = Layer(type='conv_mask', kernel_size=6, stride=1, padding=0)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='dropout', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=1, stride=1, padding=0)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='dropout', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=1, stride=1, padding=0)
    net.append(layer)

def vgg_s():
    net = []
    layer = Layer(type='conv', kernel_size=7, stride=2, padding=0)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='lrn', kernel_size=0, stride=0, padding=1)
    net.append(layer)
    layer = Layer(type='pool', kernel_size=3, stride=3, padding=0)
    net.append(layer)

    layer = Layer(type='conv', kernel_size=5, stride=1, padding=0)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='pool', kernel_size=2, stride=2, padding=0)
    net.append(layer)

    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv_mask', kernel_size=3, stride=1, padding=1)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='pool', kernel_size=3, stride=3, padding=0)
    net.append(layer)

    layer = Layer(type='conv_mask', kernel_size=6, stride=1, padding=0)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='dropout', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=1, stride=1, padding=0)
    net.append(layer)
    layer = Layer(type='relu', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='dropout', kernel_size=0, stride=0, padding=0)
    net.append(layer)
    layer = Layer(type='conv', kernel_size=1, stride=1, padding=0)
    net.append(layer)

def getConvNetPara(model):
    if model == 'alexnet':
        net = alexnet()
    elif model == 'vgg_m':
        net = vgg_m()
    elif model == 'vgg_s':
        net = vgg_s()
    elif model == 'vgg_vd_16':
        net = vgg_16_vd()


    convnet = {'targetLayer': [],
               'targetScale': [],
               'targetStride': [],
               'targetCenter': []}

    convLayers=[]
    for i in range(len(net)):
        if 'conv' in net[i].type:
            convLayers.append(i+1)
            convnet['targetLayer'].append(i+2)
    length = len(convLayers)

    for i in range(length):
        tarLay = convLayers[i]
        pad = net[tarLay-1].padding
        scale = net[tarLay-1].kernel_size
        stride = net[tarLay-1].stride
        if (i == 0):
            convnet['targetStride'].append(stride)
            convnet['targetScale'].append(scale)
            convnet['targetCenter'].append((1+scale-pad*2)/2)
        else:
            IsPool = False
            poolStride = 0
            poolSize = 0
            poolPad = 0
            for j in range(convLayers[i-1]+1,tarLay-1):
                if 'pool' in net[j].type:
                    IsPool = True
                    poolSize = net[j].kernel_size
                    poolStride = net[j].stride
                    poolPad = net[j].padding
            convnet['targetStride'].append((1 + IsPool * (poolStride - 1)) * stride * convnet['targetStride'][i-1])
            convnet['targetScale'].append(convnet['targetScale'][i-1] + IsPool * (poolSize - 1) * convnet['targetStride'][i-1] + convnet['targetStride'][i] * (scale - 1))
            if (IsPool):
                convnet['targetCenter'].append((scale - pad * 2 - 1) * poolStride * convnet['targetStride'][i-1] / 2 + (convnet['targetCenter'][i-1] + convnet['targetStride'][i-1] * (poolSize - 2 * poolPad - 1) / 2))
            else:
                convnet['targetCenter'].append((scale - pad * 2 - 1) * convnet['targetStride'][i-1] / 2 + convnet['targetCenter'][i-1])
    return convnet


