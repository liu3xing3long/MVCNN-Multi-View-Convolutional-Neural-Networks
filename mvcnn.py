import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = self.make_layers()
    def make_layers(self):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M']
        layers = []
        in_channels = 3
        for index, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.BatchNorm2d(cfg[index-1]),
                           nn.MaxPool2d(kernel_size=2, stride=2),
                           nn.Dropout(p=0.2)]

            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3),
                           nn.ELU(inplace=True)]
                in_channels = x

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)

        return out

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        return x

class modelnet40_VGG(nn.Module):
    def __init__(self, num_cuda, batch_size, multi_gpu=True):
        super(modelnet40_VGG, self).__init__()
        self.view_00 = VGG()
        self.view_01 = VGG()
        self.view_02 = VGG()
        self.view_03 = VGG()
        self.view_04 = VGG()
        self.view_05 = VGG()
        self.view_06 = VGG()
        self.view_07 = VGG()
        self.view_08 = VGG()
        self.view_09 = VGG()
        self.view_10 = VGG()
        self.view_11 = VGG()
        if multi_gpu:
            self.batch_size = batch_size/num_cuda
        else:
            self.batch_size = batch_size
        self.reduce_dim_1 = torch.nn.Conv2d(3072, 512,  kernel_size=1)
        self.reduce_dim_2 = torch.nn.Conv2d(128, 40,  kernel_size=1)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(128)
        self.ELU = nn.ELU(inplace=True)
        self.logsoftmax = nn.Softmax()
        self.drop = nn.Dropout(p=0.2)
        self.global_pooling = nn.AvgPool2d(kernel_size=19)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0


    def forward(self, x):
        # x: (nL, 12L, 3L, 227L, 227L)
        x = x.permute(1, 0, 2, 3, 4).contiguous()

        # x: (12L, nL, 3L, 227L, 227L)
        split_x = torch.split(x, 1, dim=0)
        # x: (1L, nL, 3L, 227L, 227L)
        # cnn1 (multi-view)

        #each_x = dict()
        #for i in range(0, 12):
        #    each_x[i] = self.view[i].forward(split_x[i].view(self.batch_size, 3, 227, 227))
        x00 = self.view_00.forward(split_x[0].view(self.batch_size, 3, 227, 227))
        x01 = self.view_01.forward(split_x[1].view(self.batch_size, 3, 227, 227))
        x02 = self.view_02.forward(split_x[2].view(self.batch_size, 3, 227, 227))
        x03 = self.view_03.forward(split_x[3].view(self.batch_size, 3, 227, 227))
        x04 = self.view_04.forward(split_x[4].view(self.batch_size, 3, 227, 227))
        x05 = self.view_05.forward(split_x[5].view(self.batch_size, 3, 227, 227))
        x06 = self.view_06.forward(split_x[6].view(self.batch_size, 3, 227, 227))
        x07 = self.view_07.forward(split_x[7].view(self.batch_size, 3, 227, 227))
        x08 = self.view_08.forward(split_x[8].view(self.batch_size, 3, 227, 227))
        x09 = self.view_09.forward(split_x[9].view(self.batch_size, 3, 227, 227))
        x10 = self.view_10.forward(split_x[10].view(self.batch_size, 3, 227, 227))
        x11 = self.view_11.forward(split_x[11].view(self.batch_size, 3, 227, 227))

        # append and shrink down number of features
        #merge_result = torch.cat([each_x[1], each_x[2], each_x[3], each_x[4], each_x[5],
        #                          each_x[6], each_x[7], each_x[8], each_x[9], each_x[10],
        #                          each_x[11], each_x[0]], 1)
        merge_result = torch.cat([x00,x01,x02,x03,x04,x05,x06,x07,x08,x09,x10,x11], 1)
        merge_result = self.drop(merge_result)
        merge_result = self.ELU(self.reduce_dim_1(merge_result))
        # cnn2
        merge_result = self.ELU(self.bn1(self.conv1(merge_result)))
        merge_result = self.drop(merge_result)
        merge_result = self.ELU(self.bn2(self.conv2(merge_result)))
        merge_result = self.reduce_dim_2(merge_result)
        merge_result = self.global_pooling(merge_result)
        merge_result = merge_result.view(merge_result.size(0), -1)
        merge_result = self.logsoftmax(merge_result)
        return merge_result


class modelnet40_Alex(nn.Module):
    def __init__(self, num_cuda, batch_size, multi_gpu=True):
        super(modelnet40_Alex, self).__init__()

        self.view_00 = AlexNet()
        self.view_01 = AlexNet()
        self.view_02 = AlexNet()
        self.view_03 = AlexNet()
        self.view_04 = AlexNet()
        self.view_05 = AlexNet()
        self.view_06 = AlexNet()
        self.view_07 = AlexNet()
        self.view_08 = AlexNet()
        self.view_09 = AlexNet()
        self.view_10 = AlexNet()
        self.view_11 = AlexNet()

        if multi_gpu:
            self.batch_size = batch_size/num_cuda
        else:
            self.batch_size = batch_size
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 40),
        )
        self.logsoftmax = nn.Softmax()
        self.reduce_dim_1 = torch.nn.Conv1d(12, 1,  kernel_size=1)
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0


    def forward(self, x):
        # x: (nL, 12L, 3L, 227L, 227L)
        x = x.permute(1, 0, 2, 3, 4).contiguous()

        # x: (12L, nL, 3L, 227L, 227L)
        split_x = torch.split(x, 1, dim=0)
        # x: (1L, nL, 3L, 227L, 227L)
        # cnn1 (multi-view)

        #each_x = dict()
        #for i in range(0, 12):
        #    each_x[i] = self.view[i].forward(split_x[i].view(self.batch_size, 3, 227, 227))
        x00 = self.view_00.forward(split_x[0].view(self.batch_size, 3, 227, 227)).unsqueeze(1)
        x01 = self.view_01.forward(split_x[1].view(self.batch_size, 3, 227, 227)).unsqueeze(1)
        x02 = self.view_02.forward(split_x[2].view(self.batch_size, 3, 227, 227)).unsqueeze(1)
        x03 = self.view_03.forward(split_x[3].view(self.batch_size, 3, 227, 227)).unsqueeze(1)
        x04 = self.view_04.forward(split_x[4].view(self.batch_size, 3, 227, 227)).unsqueeze(1)
        x05 = self.view_05.forward(split_x[5].view(self.batch_size, 3, 227, 227)).unsqueeze(1)
        x06 = self.view_06.forward(split_x[6].view(self.batch_size, 3, 227, 227)).unsqueeze(1)
        x07 = self.view_07.forward(split_x[7].view(self.batch_size, 3, 227, 227)).unsqueeze(1)
        x08 = self.view_08.forward(split_x[8].view(self.batch_size, 3, 227, 227)).unsqueeze(1)
        x09 = self.view_09.forward(split_x[9].view(self.batch_size, 3, 227, 227)).unsqueeze(1)
        x10 = self.view_10.forward(split_x[10].view(self.batch_size, 3, 227, 227)).unsqueeze(1)
        x11 = self.view_11.forward(split_x[11].view(self.batch_size, 3, 227, 227)).unsqueeze(1)

        # each is[n, 6*6*256] features

        # append and shrink down number of features
        merge_result = torch.cat([x00,x01,x02,x03,x04,x05,x06,x07,x08,x09,x10,x11], 1)
        merge_result = self.reduce_dim_1(merge_result)
        merge_result = merge_result.view(merge_result.size(0), -1)
        merge_result = self.classifier(merge_result)
        # now it is [n, 6*6*256*12] features

        merge_result = self.logsoftmax(merge_result)
        return merge_result

























