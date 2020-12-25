import torch
from torch.autograd import Function


class logistic_F(Function):
    @staticmethod
    def forward(self, x, c):
        #print('loss_forward')
        a = -c.mul(x)
        b = torch.max(a,torch.zeros(a.size()).cuda())
        #b = torch.max(a, torch.zeros(a.size()))
        t = b + torch.log(torch.exp(-b) + torch.exp(a-b))
        t = torch.sum(t)
        #t1 = torch.sum((b>0))
        self.save_for_backward(x, c)
        return t

    @staticmethod
    def backward(self, grad_output):
        #print('loss_backward')
        x,c = self.saved_tensors
        x_grad = c_grad = None
        x_grad = -grad_output*c.div(1+torch.exp(c.mul(x)))
        return x_grad , c_grad




