import torch
import torch.nn as nn
import math

__all__ = ['ResNet','resnet18','resnet34','resnet50','resnet101','resnet152']


def con3x3(in_planes, out_planes, stride=1):
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride = 1, downsample=None):
    super(BasicBlock, self).__init__()

    self.conv1 = con3x3(in_planes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)

    self.conv2 = con3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)

    self.relu = nn.ReLU(inplace=True)

    self.downsample = downsample

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out

class Bottlenet(nn.Module):
  expansion = 4

  def __init__(self, in_planes, planes, stride=1, downsample=None):
    super(Bottlenet, self).__init__()

    self.conv1 = con3x3(in_planes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)

    self.conv2 = con3x3(planes,planes)
    self.bn2 = nn.BatchNorm2d(planes)

    self.conv3 = con3x3(planes,planes * self.expansion)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion)

    self.downsample = downsample
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    super(ResNet,self).__init__()
    
    self.in_planes = 64
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(self.in_planes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size = 3, stride =2, padding =1)

    self.layer1 = self.make_layers(block, 64, layers[0])
    self.layer2 = self.make_layers(block, 128, layers[1], stride=2)
    self.layer3 = self.make_layers(block, 256, layers[2], stride=2)
    self.layer4 = self.make_layers(block, 512, layers[3], stride=2)

    self.avgpool = nn.AvgPool2d(7, stride=1)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    self._init_weight()

  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.maxpool(out)

    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)

    out = self.avgpool(out)
    out = out.view(out.size(0),-1)
    out = self.fc(out)

    return out

  def _init_weight(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
          m.bias.data.zero_()

      if isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

      if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

  def make_layers(self, block, planes, layer, stride=1):

    layers = []
    downsample = None

    if stride != 1 or self.in_planes != planes * block.expansion:
       downsample = nn.Sequential(nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride = stride, bias=False), nn.BatchNorm2d(planes * block.expansion))

    layers += [block(self.in_planes, planes, stride, downsample)]
    self.in_planes = planes * block.expansion
    for i in range(1, layer):
      layers += [block(self.in_planes, planes)]

    return nn.Sequential(*layers)


def resnet18(**kwargs):
  return ResNet(BasicBlock,[2,2,2,2])

def resnet34(**kwargs):
  return ResNet(BasicBlock,[3,4,6,3])

def resnet50(**kwargs):
  return ResNet(Bottlenet, [3,4,6,3])

def resnet101(**kwargs):
  return ResNet(Bottlenet, [3,4,23,3])

def resnet152(**kwargs):
  return ResNet(Bottlenet, [3,8,36,3])


