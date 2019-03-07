import torch
import torch.nn as nn
import torch.nn.init as init

import math

__all__=['VGG','vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg_19','vgg19_bn']

cfgs = {
    'A': [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'B': [64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'D': [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'E': [64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
}

def make_layer(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    
    return nn.Sequential(layers)

class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512,num_classes)
        )
        self.init_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        
        return x
    
    def init_weight(self):
        for m in self.modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1)
                m.bias.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.normal_(0,0.01)
                m.bias.zero_()
    
def vgg11(**kwargs):
    return VGG(make_layer(cfgs['A']),**kwargs)

def vgg11_bn(**kwargs):
    return VGG(make_layer(cfgs['A'],batch_norm=True),**kwargs)

def vgg13(**kwargs):
    return VGG(make_layer(cfgs['B']),**kwargs)

def vgg13_bn(**kwargs):
    return VGG(make_layer(cfgs['B'],batch_norm=True),**kwargs)

def vgg16(**kwargs):
    return VGG(make_layer(cfgs['D']),**kwargs)

def vgg16_bn(**kwargs):
    return VGG(make_layer(cfgs['D'],batch_norm=True),**kwargs)

def vgg19(**kwargs):
    return VGG(make_layer(cfgs['E']),**kwargs)

def vgg19_bn(**kwargs):
    return VGG(make_layer(cfgs['E'],batch_norm=True),**kwargs)


