import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import model.vgg as vgg

model_names = ['vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn']

parser = argparse.ArgumentParser(description='cifar10 training for pytorch')
parser.add_argument('--arch','-a',metavar='ARCH', default='vgg19', 
                    choices=model_names, help='model archiitecture: ' + '|'.join(model_names))
parser.add_argument('-j', '--num_workers', default=4, type=int,metavar='N',help="number of data load workers.")
parser.add_argument('--epoches', default=300, type=int, metavar='N',help='number of total epoch to run.')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='manual epoch num.')
parser.add_argument('-b','--batch_size', default=128, type=int, metavar='N',help='batch size')
parser.add_argument('-lr','--learning_rate',default=0.05,type=float,metavar='LR',help='init learning rate.')
parser.add_argument('--momentum',default=0.9,type=float,metavar='M',help='momentum')
parser.add_argument('--weight_decay',default=5e-4,type=float,metavar='W',help='weight decay')
parser.add_argument('--resume',default='',type=str, metavar='PATH',help='resume path')
parser.add_argument('-e','--evaluate',dest='evaluate',action='store_true',help='evaluate model')
parser.add_argument('--pretrained',dest='pretrained',action='store_true',help='pretrained model')
parser.add_argument('--half',dest='half',action='store_true',help='use 16bit')
parser.add_argument('--save_dir',default='save_tmp',type=str,dest='save_dir',help='path to save model')
parser.add_argument('--print_freq','-p',default=20,type=int,metavar='N',help='print frequency.')


class AverageMeter(object):
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
        

def save_point(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    

def adjust_learning_rate(epoch, optimizer):
    lr = args.learning_rate * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = output.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))
        
    return res


best_prec1 = 0
def main():
    global args, best_prec1
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    model = vgg.__dict__[args.arch]()
    
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
            
        else:
            print("=> no checkpoint found at {}".format(args.resume))
            
    cudnn.benchmark = True
    
    normalize = transforms.Normalize(mean=[0.485,0.456,0,406],std=[0.229,0.224,0.225])
    
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data',
                                                                train=True,
                                                                transform=transforms.Compose([
                                                                   transforms.RandomHorizontalFlip(),
                                                                   transforms.RandomCrop(32,4),
                                                                   transforms.ToTensor(),normalize]),
                                                                download=True),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', 
                                                              train=False, 
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  normalize]),
                                                              download=True),
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True)
    
    
    criterion = nn.CrossEntropyLoss().cuda()
    
    if args.half:
        model.half()
        criterion.half()
            
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        
    if args.evaluate:
        validate(val_loader, model, criterion)
        return
        
    for epoch in range(args.start_epoch, args.epoches):
        adjust_learning_rate(epoch,optimizer) 
        train(train_loader, model, criterion, optimizer, epoch)
        if (epoch + 1) % 20 == 0:
            prec1 = validate(val_loader, model, criterion)    
            best_prec1 = max(prec1,best_prec1)
            save_point({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1},
                        filename=os.path.join(args.save_dir, 'checkpoints_{}.tar'.format(epoch)))      

        
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        target = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)
        
        if args.half:
            input_var = input_var.half()
        
        output = model(input_var)
        loss = criterion(output, target_var)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        output = output.float()
        loss = loss.float()
        
        prec1 = accuracy(output,target_var)[0]
        losses.update(loss.item(),input.size(0))
        top1.update(prec1, input.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target)
        
            if args.half:
                input_val = input_val.half()
            
            output = model(input_var)
            loss = criterion(output, target_var)
        
            prec1 = accuracy(output,target_var)[0]
            losses.update(loss,input.size(0))
            top1.update(prec1,input.size(0))
        
            batch_time.update(time.time()-end)
            end = time.time()
        
            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, top1=top1))
            
        print('* prec@1 {top1.avg:.3f}'.format(top1=top1))
        
    return top1.avg

if __name__ == '__main__':
	main()
