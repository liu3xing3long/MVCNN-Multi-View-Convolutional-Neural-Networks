import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel# for multi-GPU training
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from mvcnn import modelnet40_Alex, modelnet40_VGG
from data_processing import Dataset
import argparse
import os
import shutil
import torch.cuda as cuda

LISTS = './list/'
TRAIN = LISTS + 'train/'
TEST = LISTS + 'test/'
CLASSES = './classes.txt'

TRAIN_OUT = './train_lists.txt'
TEST_OUT = './test_lists.txt'
VAL_OUT = './val_lists.txt'

best_prec1 = 0


parser = argparse.ArgumentParser(description='PyTorch MVCNN Training')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--b', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')


# load dataset
def read_lists(list_of_lists_file):
    listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
    listfiles, labels  = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    return listfiles, labels

def main():

    global args, best_prec1
    args = parser.parse_args()

    # Read list of training and validation data
    listfiles_train, labels_train = read_lists(TRAIN_OUT)
    listfiles_val, labels_val = read_lists(VAL_OUT)
    listfiles_test, labels_test = read_lists(TEST_OUT)
    dataset_train = Dataset(listfiles_train, labels_train, subtract_mean=False, V=12)
    dataset_val = Dataset(listfiles_val, labels_val, subtract_mean=False, V=12)
    dataset_test = Dataset(listfiles_test, labels_test, subtract_mean=False, V=12)

    # shuffle data
    dataset_train.shuffle()
    dataset_val.shuffle()
    dataset_test.shuffle()
    tra_data_size, val_data_size, test_data_size= dataset_train.size(), dataset_val.size(), dataset_test.size()
    print 'training size:', tra_data_size
    print 'validation size:', val_data_size
    print 'testing size:', test_data_size

    batch_size = args.b
    print("batch_size is :" + str(batch_size))
    learning_rate = args.lr
    print("learning_rate is :" + str(learning_rate))
    num_cuda = cuda.device_count()
    print("number of GPUs have been detected:"+str(num_cuda))

    # creat model
    print("model building...")
    mvcnn = DataParallel(modelnet40_Alex(num_cuda, batch_size))
    #mvcnn = modelnet40(num_cuda, batch_size, multi_gpu = False)
    mvcnn.cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint'{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            mvcnn.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    #print(mvcnn)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adadelta(mvcnn.parameters(), weight_decay=1e-4)
    # evaluate performance only
    if args.evaluate:
        print 'testing mode ------------------'
        validate(dataset_test, mvcnn, criterion, optimizer, batch_size)
        return

    print 'training mode ------------------'
    for epoch in xrange(args.start_epoch, args.epochs):
        print('epoch:', epoch)

        #adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(dataset_train, mvcnn, criterion, optimizer, epoch, batch_size)

        # evaluate on validation set
        prec1 = validate(dataset_val, mvcnn, criterion, optimizer, batch_size)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
          save_checkpoint({
              'epoch': epoch + 1,
              'state_dict': mvcnn.state_dict(),
              'best_prec1': best_prec1,
          }, is_best, epoch)
        elif epoch % 5 is 0:
          save_checkpoint({
              'epoch': epoch + 1,
              'state_dict': mvcnn.state_dict(),
              'best_prec1': best_prec1,
          }, is_best, epoch)

          #max_index = ttoutputs.max(dim=1)[1]
        #  (max_index == ttargets).sum()

# train
def train(dataset_train, mvcnn, criterion, optimizer, epoch, batch_size):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    mvcnn.train()
    step = 0
    end = time.time()
    for batch_x, batch_y in dataset_train.batches(batch_size):
        # measure data loading time
        data_time.update(time.time() - end)

        batch_x, batch_y = torch.from_numpy(batch_x), torch.from_numpy(batch_y)
        batch_x, batch_y = batch_x.type(torch.FloatTensor), batch_y.type(torch.LongTensor)

        input, target = batch_x.cuda(), batch_y.cuda(async=True)

        input_var, target_var = torch.autograd.Variable(input), torch.autograd.Variable(target)

        # compute output
        output = mvcnn(input_var)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, check_result=False, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print training information
        if step % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, step, dataset_train.size()/batch_size, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
        step += 1
        del loss
        del input_var
        del target_var

def validate(dataset_val, mvcnn, criterion, optimizer, batch_size):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    mvcnn.eval()

    for batch_x, batch_y in dataset_val.batches(batch_size):
        batch_x, batch_y = torch.from_numpy(batch_x), torch.from_numpy(batch_y)
        batch_x, batch_y = batch_x.type(torch.FloatTensor), batch_y.type(torch.LongTensor)
        input, target = batch_x.cuda(), batch_y.cuda(async=True)
        input_var, target_var = torch.autograd.Variable(input, volatile=False), torch.autograd.Variable(target, volatile=False)
        # compute output
        output = mvcnn(input_var)
        loss=criterion(output, target_var)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, check_result=True, topk=(1, 5))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del loss
        print(' Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))


    return top1.avg


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
        self.sum += val * n  #   val*batch_size
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target_var, check_result = False, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = 8

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    if check_result is True:
        print("Prediction:\n")
        print(pred)
        print("Ground_Truth:\n")
        print(target_var.unsqueeze(0).permute(0, 1).contiguous())
    correct = pred.eq(target_var.data.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, epoch, filename='_epoch_checkpoint.pth.tar'):
    torch.save(state, "model"+str(epoch)+filename)
    if is_best:
        shutil.copyfile("model"+str(epoch)+filename, "model"+str(epoch)+'_epoch_model_best.pth.tar')

if __name__ == '__main__':

    main()




















