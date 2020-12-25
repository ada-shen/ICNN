import os
import math
import torch
import numpy as np
from tools.sgd import SGD
import torch.autograd.variable as Variable
from tools.logistic import logistic_F
from tools.softmax import softmax_F
from tensorboardX import SummaryWriter
from tools.lib import *
from torch import nn


def train_model(taskid_path, args, net, train_dataloader, val_dataloader, density, dataset_length):

    log_path = os.path.join(taskid_path,"log")
    make_dir(log_path)
    writer = SummaryWriter(log_path)

    torch.cuda.set_device(args.gpu_id)
    net = net.cuda()

    max_acc = 0
    max_epoch = 0
    judge = 0


    for epoch in range(args.epochnum):
        paras = dict(net.named_parameters())
        paras_new = []
        for k, v in paras.items():
            if 'mask' in k:
                if 'bias' in k:
                    paras_new += [{'params': [v], 'lr': args.lr[epoch] * 2, 'weight_decay': args.weightdecay * 0}]
                if 'mask_weight' in k:
                    paras_new += [{'params': [v], 'lr': args.lr[epoch] * 0.05, 'weight_decay': args.weightdecay * 0}]
                if '.weight' in k:
                    paras_new += [{'params': [v], 'lr': args.lr[epoch] * 1, 'weight_decay': args.weightdecay * 1}]
            if 'line' in k:
                if 'bias' in k:
                    paras_new += [{'params': [v], 'lr': args.lr[epoch] * 2, 'weight_decay': args.weightdecay * 0}]
                if 'weight' in k:
                    paras_new += [{'params': [v], 'lr': args.lr[epoch] * 1, 'weight_decay': args.weightdecay * 1}]
            if 'conv' in k:
                if 'bias' in k:
                    paras_new += [{'params': [v], 'lr': args.lr[epoch] * 1, 'weight_decay': args.weightdecay * 1}]
                if 'weight' in k:
                    paras_new += [{'params': [v], 'lr': args.lr[epoch] * 1, 'weight_decay': args.weightdecay * 1}]
        optimizer = SGD(paras_new, lr=args.lr[epoch], momentum=args.momentum, weight_decay=args.weightdecay)

        # train
        net.train()
        train_loss = []
        train_acc = []
        print('Train: ' + "\n" + 'epoch:{}'.format(epoch + 1))
        for index, (image, label) in enumerate(train_dataloader):
            batch_size = image.shape[0]

            image = Variable(image)
            image = image.cuda()
            label = label.cuda()

            out = net(image, label, torch.Tensor([epoch + 1]), density)

            if args.model == "resnet_18" or args.model == "resnet_50" or args.model == "densenet_121":
                out = torch.unsqueeze(out,2)
                out = torch.unsqueeze(out, 3)
            label = Variable(label)
            if args.losstype == 'logistic':
                loss = logistic_F.apply(out, label)
                train_loss.append(loss.cpu().clone().data.numpy())
                train_correct = label.mul(out)
                train_correct = torch.max(train_correct, torch.zeros(train_correct.size()).cuda())
                train_correct = torch.sum((train_correct > 0))
                train_acc.append(train_correct.cpu().data.numpy())
            if args.losstype == 'softmax':
                loss = softmax_F.apply(out, label)
                train_loss.append(loss.cpu().clone().data.numpy())
                (tmp, out) = torch.sort(out, dim=1, descending=True)
                (tmp, label) = torch.max(label, dim=1)
                label = label.unsqueeze(2)
                error = ~(out == label)
                train_correct = args.batchsize - torch.sum(error[:, 0, 0, 0])
                train_acc.append(train_correct.cpu().data.numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('batch:{}/{}'.format(index + 1, len(train_dataloader)) + " " +
                  'loss:{:.6f}'.format(loss / batch_size) + " " +
                  'acc:{:.6f}'.format(train_correct.cpu().data.numpy()/(batch_size*args.label_num)))

            length = dataset_length['train'] if index + 1 == len(train_dataloader) else args.batchsize * (index + 1)
            if (index + 1) % 10:
                writer.add_scalar('Train/Loss', sum(train_loss)/ length, epoch)
                writer.add_scalar('Train/acc', sum(train_acc)/ (length*args.label_num), epoch)


        # eval

        net.eval()
        with torch.no_grad():
            eval_loss = []
            eval_acc = []
            for index, (image, label) in enumerate(val_dataloader):
                print('Val: ' + "\n" + 'epoch:{}'.format(epoch + 1))
                batch_size = image.shape[0]
                image = Variable(image)
                image = image.cuda()
                label = label.cuda()

                out = net(image, label, torch.Tensor([epoch + 1]), density)
                if args.model == "resnet_18" or args.model == "resnet_50" or args.model == "densenet_121":
                    out = torch.unsqueeze(out, 2)
                    out = torch.unsqueeze(out, 3)
                label = Variable(label)
                if args.losstype == 'logistic':
                    loss = logistic_F.apply(out, label)
                    eval_loss.append(loss.cpu().data.numpy())
                    eval_correct = label.mul(out)
                    eval_correct = torch.max(eval_correct, torch.zeros(eval_correct.size()).cuda())
                    eval_correct = torch.sum((eval_correct > 0))
                    eval_acc.append(eval_correct.cpu().data.numpy())
                if args.losstype == 'softmax':
                    loss = softmax_F.apply(out, label)
                    eval_loss.append(loss.cpu().data.numpy())
                    (tmp, out) = torch.sort(out, dim=1, descending=True)
                    (tmp, label) = torch.max(label, dim=1)
                    label = label.unsqueeze(2)
                    error = ~(out == label)
                    eval_correct = args.batchsize - torch.sum(error[:, 0, 0, 0])
                    eval_acc.append(eval_correct.cpu().data.numpy())
                length = dataset_length['val'] if index + 1 == len(val_dataloader) else args.batchsize * (index + 1)
                print('batch:{}/{}'.format(index + 1, len(val_dataloader)) + " " +
                      'loss:{:.6f}'.format(loss/batch_size) + " " +
                      'acc:{:.6f}'.format(eval_correct.cpu().data.numpy()/(batch_size*args.label_num)))
            print("max_acc:"+str(max_acc))

            if sum(eval_acc)/(length*args.label_num)>max_acc:
                judge=1
                max_acc=sum(eval_acc)/(length*args.label_num)
                print("rightnow max_acc:"+str(max_acc))
                max_epoch=epoch

            writer.add_scalar('Eval/Loss', sum(eval_loss)/ length, epoch)
            writer.add_scalar('Eval/acc', sum(eval_acc)/ (length*args.label_num), epoch)
        if judge==1 or (epoch+1)%50==0:
            # save
            torch.save(net, taskid_path + '/net-' + str(epoch + 1) + '.pkl')
            #torch.save(net.state_dict(), taskid_path + '/net-params-' + str(epoch + 1) + '.pkl')
            judge=0

    return max_acc,max_epoch+1