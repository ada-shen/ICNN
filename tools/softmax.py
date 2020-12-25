import torch
import h5py
import numpy as np
from torch.autograd import Function
import torch.autograd.variable as Variable

class softmax_F(Function):
    @staticmethod
    def forward(self, x, c):
        #print('loss_forward')
        (a,tmp) = torch.max(c,1)
        tmp = tmp.unsqueeze(1).float()
        inputSize = np.array([x.size()[0],x.size()[1],x.size()[2],x.size()[3]])
        numPixelsPerImage = np.prod(inputSize[2:4])
        numPixels = numPixelsPerImage * inputSize[0]
        imageVolume = numPixelsPerImage * inputSize[1]
        n = np.array(range(numPixels))
        n = np.reshape(n, (tmp.size()[0],tmp.size()[1],tmp.size()[2],tmp.size()[3]))
        offset = np.mod(n,numPixelsPerImage) + imageVolume * np.fix(n / numPixelsPerImage)
        ci = torch.from_numpy(offset).float().cuda() + numPixelsPerImage * torch.max(tmp,torch.zeros(tmp.size()).cuda())
        (Xmax,b) = torch.max(x,1)
        Xmax = Xmax.unsqueeze(1).float()
        ex = torch.exp(x - Xmax)
        x_line = x.reshape(x.size()[0]*x.size()[1])
        x_ci = torch.zeros(ci.size()).cuda()
        for i in range(x.size()[0]):
            x_ci[i, 0,0,0] = x_line[ci[i, 0, 0, 0].long()]
        t = Xmax + torch.log(torch.sum(ex,1).unsqueeze(1)) - x_ci
        t = torch.sum(t)
        self.save_for_backward(x, tmp)
        return t

    @staticmethod
    def backward(self, grad_output):
        #print('loss_backward')
        x,tmp = self.saved_tensors
        x_grad = tmp_grad = None
        inputSize = np.array([x.size()[0],x.size()[1],x.size()[2],x.size()[3]])
        numPixelsPerImage = np.prod(inputSize[2:4]) ;
        numPixels = numPixelsPerImage * inputSize[0] ;
        imageVolume = numPixelsPerImage * inputSize[1];

        n = np.array(range(numPixels))
        n = np.reshape(n, (tmp.size()[0],tmp.size()[1],tmp.size()[2],tmp.size()[3]))
        offset = np.mod(n,numPixelsPerImage) + imageVolume * np.fix(n / numPixelsPerImage)
        ci = torch.from_numpy(offset).float().cuda() + numPixelsPerImage * torch.max(tmp,torch.zeros(tmp.size()).cuda())
        (Xmax,b) = torch.max(x,1)
        Xmax = Xmax.unsqueeze(1).float()
        ex  = torch.exp(x - Xmax)
        x_grad = ex.div(torch.sum(ex,1).unsqueeze(1))

        x_grad_line = x_grad.reshape(x_grad.size()[0]*x_grad.size()[1])
        for i in range(ci.size()[0]):
            index = ci[i,0,0,0].long()
            x_grad_line[index] = x_grad_line[index] - 1
        x_grad = x_grad_line.reshape(x_grad.size()[0],x_grad.size()[1],x_grad.size()[2],x_grad.size()[3])
        x_grad = grad_output.mul(x_grad)
        x_grad = x_grad * x.size()[1]
        #print('x_grad',x_grad[0,:,:,:])
        return x_grad , tmp_grad



'''
Data = h5py.File('imdb_train_32_mutil.mat')
label = Data['label'][0:8]
label = torch.from_numpy(label)
label = label.reshape(label.size()[0],label.size()[1],1,1).cuda()

data = h5py.File('m_fmap_finally.mat')
x = data['fmap'][:, :, :, :]
x = torch.from_numpy(x).cuda()
x = Variable(x,requires_grad = True)

loss = Our_loss_softmax_F.apply(x,label)
loss.backward()
'''
