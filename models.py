from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn



class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(128, 64, kernel_size=3)
        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.l0 = nn.Linear(32 * 22 * 22 + 64, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = h.view(y.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(192, 512, kernel_size=1)
        self.c1 = nn.Conv2d(512, 512, kernel_size=1)
        self.c2 = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)


class PriorDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(64, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(3, 64, kernel_size=4, stride=1)
        self.c1 = nn.Conv2d(64, 128, kernel_size=4, stride=1)
        self.c2 = nn.Conv2d(128, 256, kernel_size=4, stride=1)
        self.c3 = nn.Conv2d(256, 512, kernel_size=4, stride=1)
        self.l1 = nn.Linear(512*20*20, 64)

        self.b1 = nn.BatchNorm2d(128)
        self.b2 = nn.BatchNorm2d(256)
        self.b3 = nn.BatchNorm2d(512)
        

    def forward(self, x):
        h = F.relu(self.c0(x))
        features = F.relu(self.b1(self.c1(h)))
        h = F.relu(self.b2(self.c2(features)))
        h = F.relu(self.b3(self.c3(h)))
        encoded = self.l1(h.view(x.shape[0], -1))
        return encoded, features



class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 15)
        self.bn1 = nn.BatchNorm1d(15)
        self.fc2 = nn.Linear(15, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(10, 10)
        self.bn3 = nn.BatchNorm1d(10)

    def forward(self, x):
        encoded, _ = x[0], x[1]
        x = F.relu(self.bn1(self.fc1(encoded)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))

        return x


class ClassificationModel(nn.Module):
    def __init__(self, pretrain, fine_tune):
        super().__init__()
        self.encoder = Encoder()
        self.classifier = Classifier()

        if not fine_tune:
            self.freeze_encoder()

        if pretrain:
            self.load_pretrained(pretrain)


    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        return
    
    def load_pretrained(self, pretrain_pth):
        weights_dict = torch.load(pretrain_pth)
        model_dict = self.state_dict()
        
        pretrained_dict = {k: v for k, v in weights_dict.items() if
                        k in model_dict and weights_dict[k].shape == model_dict[k].shape}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        
        return


    def forward(self, x):
        z, features = self.encoder(x)
        # z = z.detach()
        return self.classifier((z, features))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


