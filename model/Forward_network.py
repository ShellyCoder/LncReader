import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):

    def __init__(self):
        super(Mish,self).__init__()

    def forward(self,x):
        return x * torch.tanh(F.softplus(x))

class MLE(nn.Module):

    def __init__(self, in_channels, num_classes, drop_p=0.2):
        super().__init__()
        self.layer1 = nn.Sequential(*[
            nn.Linear(in_channels, 256, bias=True),
            nn.BatchNorm1d(256, eps=1e-3, momentum=0.01),
            nn.Softplus(),
            nn.Dropout(drop_p),
        ])
        self.layer2 = nn.Sequential(*[
            nn.Linear(256, 512, bias=True),
            nn.BatchNorm1d(512, eps=1e-3, momentum=0.01),
            nn.Softplus(),
            nn.Dropout(drop_p),
        ])
        self.layer3 = nn.Sequential(*[
            nn.Linear(512, 256, bias=True),
            nn.BatchNorm1d(256, eps=1e-3, momentum=0.01),
            nn.Softplus(),
            nn.Dropout(drop_p),
        ])
        self.layer4 = nn.Sequential(*[
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.01),
            nn.Softplus(),
            nn.Dropout(drop_p),
        ])
        self.layer5 = nn.Linear(128, num_classes)


    def forward(self,x):
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return self.layer5(l4)