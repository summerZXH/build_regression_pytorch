import sys
import os
import os.path
import random
import collections
import shutil
import time
import glob
import csv
import numpy as np
import math

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import json
import data_input

from PIL import Image
import cv2

ROOT_DIR = os.getcwd()
DATA_HOME_DIR = ROOT_DIR

# paths
data_path = DATA_HOME_DIR + '/'
train_path = data_path + 'data/insulator_image'
valid_path = data_path + 'data/insulator_image'
test_path = data_path + 'data/insulator_image/'
train_json_path = data_path + 'data/insulator_train.txt'
val_json_path = data_path + 'data/insulator_val.txt'
test_json_path = data_path + 'data/insulator_test.txt'

pretrained_model = '/home/zhuxiuhong/dataset/pretrained_model/pytorch/whole_resnet50_places365.pth'
output_path = ROOT_DIR + '/models/'
best_model_path = output_path + 'model_best.pth.tar'
checkpoint_path = output_path + 'checkpoint.pth.tar'
submission_path = output_path
log_path = output_path + 'log.txt'
log_txt = open(log_path, 'w+')

# data
batch_size = 16
image_size = 224
scale_size = 256
archs = ["resnet152"]

# model
nb_runs = 1
nb_aug = 1
epochs = 100
lr = 1e-4
clip = 0.001

# model_names = ['alexnet', 'densenet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'inception', 'inception_v3', 'resnet', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'squeezenet', 'squeezenet1_0', 'squeezenet1_1', 'vgg', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))
best_prec1 = 0


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    end = time.time()
    start = time.time()

    # switch to train mode
    model.train()

    count = 0
    for i, (images, target) in enumerate(train_loader):
        count = count + 1
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        image_var = torch.autograd.Variable(images)
        # label_var = torch.autograd.Variable(target)
        label_var = torch.autograd.Variable(target.type(torch.FloatTensor)).cuda()

        # compute y_pred
        y_pred = model(image_var)
        # loss = nn.functional.smooth_l1_loss(y_pred, label_var).cuda()
        loss = criterion(y_pred, label_var)

        # measure accuracy and record loss
        # prec1, prec1 = accuracy(y_pred.data, target, topk=(1, 1))
        losses.update(loss.data[0], images.size(0))
        # acc.update(prec1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    print("~~~~~ count = %d | epoch = %d | time = %f~~~~~~~~"%(count, epoch, end-start))


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc1 = AverageMeter()  # top1
    acc3 = AverageMeter()  # top3

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, labels) in enumerate(val_loader):
        labels = labels.cuda(async=True)
        image_var = torch.autograd.Variable(images, volatile=True)
        # label_var = torch.autograd.Variable(labels, volatile=True)
        label_var = torch.autograd.Variable(labels.type(torch.FloatTensor)).cuda()

        # compute y_pred
        y_pred = model(image_var)
        loss = criterion(y_pred, label_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(
    '   * EPOCH {epoch} |Loss: {losses.avg:.3f}'.format(
        epoch=epoch,
        losses=losses))

    s = 'Epoch {0} | Loss: {1}\n'.format(epoch, losses.avg)
    log_txt.write(s)

    return losses.avg

def Conv2Rad(x):
    max = 1.57079632679
    min = 0.0
    item = min + x * (max - min)
    #item = item * 180 / math.pi
    return item

def show_result(imgfile, result):
    scale = 3
    IMG = '/home/zhuxiuhong/src/PycharmProjects/insulator/data/insulator_image'
    im = cv2.imread(IMG + "/" + imgfile)                    
    im = cv2.resize(im, (int(im.shape[1] / scale), int(im.shape[0] / scale)))
    x1 = int(im.shape[1]/2)
    y1 = int(im.shape[0]/2)
    k = math.tan(result[0][0])
    x2 = x1 + int(im.shape[1]/4)
    y2 = y1 + int(im.shape[1]/4 * k)
    cv2.line(im, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.imshow('image', im)
    cv2.waitKey()

def test(test_loader, model):
    result = []

    # switch to evaluate mode
    model.eval()
    nb_aug = 1
    for aug in range(nb_aug):
        print("   * Predicting on test augmentation {}".format(aug + 1))

        for i, (images, filepath) in enumerate(test_loader):

            temp_dict = {}
            print("------- i = %d -------" % i)

            # pop extension, treat as id to map
            filepath = os.path.splitext(os.path.basename(filepath[0]))[0]

            image_var = torch.autograd.Variable(images, volatile=True)
            y_pred = model(image_var)

            # get the index of the max log-probability
            x = Conv2Rad(y_pred.data).tolist()
            print x

            temp_dict['image_id'] = filepath + '.jpg'
            temp_dict['label_id'] = x
            result.append(temp_dict)
            
            show_result(temp_dict['image_id'], temp_dict['label_id'])

    sub_fn = submission_path + '{0}epoch_{1}clip_{2}runs'.format(epochs, clip, nb_runs)

    for arch in archs:
        sub_fn += "_{}".format(arch)

    print("Writing Predictions to JSON...")
    with open(sub_fn + '.json', 'w') as f:
        json.dump(result, f)
        print('write result json, num is %d' % len(result))
    print("Done.")

# test in one image
def ui_test(filename, model):
    result = []

    # switch to evaluate mode
    model.eval()
        
    ui_transofoms = transforms.Compose([transforms.Scale(scale_size),
                        transforms.CenterCrop(image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                      normalize,
                       ])
    image = Image.open(filename)
    image = ui_transforms(image)
        
    image_var = torch.autograd.Variable(image, volatile=True)
    y_pred = model(image_var)

    x = Conv2Rad(y_pred.data).tolist()
    
    temp_dict = {}
    temp_dict['image_id'] = filepath + '.jpg'
    temp_dict['label_id'] = x
                    
    show_result(temp_dict['image_id'], temp_dict['label_id'])

    print("Done.")


def save_checkpoint(state, is_best, filename=checkpoint_path):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_model_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    global lr
    lr = lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def accuracy(y_pred, y_actual, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def shear(img):
    width, height = img.size
    m = random.uniform(-0.05, 0.05)
    xshift = abs(m) * width
    new_width = width + int(round(xshift))
    img = img.transform((new_width, height), Image.AFFINE,
                        (1, m, -xshift if m > 0 else 0, 0, 1, 0),
                        Image.BICUBIC)
    return img

def pre_data():
    traindir = train_path
    valdir = valid_path
    testdir = test_path

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_loader = data.DataLoader(
        data_input.TrainImageFolder(traindir, train_json_path,
                                        transforms.Compose([
                                            #ransforms.Lambda(shear),
                                            transforms.RandomSizedCrop(image_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),                                                normalize,
                                            ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True)

    val_loader = data.DataLoader(
        data_input.TrainImageFolder(valdir, val_json_path,
                                        transforms.Compose([
                                            transforms.Scale(scale_size),
                                             transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                            normalize,
                                        ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True)

    test_loader = data.DataLoader(
        data_input.TestImageFolder(testdir, test_json_path,
                                       transforms.Compose([
                                           #                                 transforms.Lambda(shear),
                                         transforms.Scale(scale_size),
                                         transforms.CenterCrop(image_size),
                                        transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                      normalize,
                                       ])),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False)

    return train_loader, val_loader, test_loader


def main(mode="train", resume=False, filename=None):
    global best_prec1

    for arch in archs:

        # create model
        print("=> Starting {0} on '{1}' model".format(mode, arch))
        print("=> Using pre-trained parameters '{}'".format(arch))
        model = models.__dict__[arch](pretrained=True)
        # Don't update non-classifier learned features in the pretrained networks
        # for param in model.parameters():
        #     param.requires_grad = False
        # Replace the last fully-connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
        # Final dense layer needs to replaced with the previous out chans, and number of classes
        # in this case -- resnet 101 - it's 2048 with two classes (cats and dogs)

        model.fc = nn.Linear(2048, 1)


        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

        # optionally resume from a checkpoint
        if resume:
            if os.path.isfile(resume):
                print("=> Loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            else:
                print("=> No checkpoint found at '{}'".format(resume))

        cudnn.benchmark = True

        # Data loading code
        train_loader, val_loader, test_loader = pre_data()

        if mode == "test":
            filename='./data/insulator_image/000024.jpg'
            ui_test(filename,model)
            test(test_loader, model)
            return

        # define loss function (criterion) and optimizer
        # criterion = nn.CrossEntropyLoss().cuda()
        criterion = nn.SmoothL1Loss().cuda()

        if mode == "validate":
            validate(val_loader, model, criterion, 0)
            return

        # optimizer = optim.Adam(model.module.fc.parameters(), lr=lr, weight_decay=1e-4)
        # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

        for epoch in range(epochs):
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            loss = validate(val_loader, model, criterion, epoch)

            # remember best Accuracy and save checkpoint
            is_best = False
            if loss < 1e-3:
                is_best = true
                best_prec1 = 1.0
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)

        log_txt.close()


if __name__ == '__main__':
    # main(mode="train")
    # main(mode="validate",resume='./model_best.pth.tar')
    main(mode="test", resume='./models/checkpoint.pth.tar')
