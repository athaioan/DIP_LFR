import torch
from models import Encoder, GlobalDiscriminator, LocalDiscriminator, PriorDiscriminator
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
import statistics as stats
import argparse
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils import read_imagenet_tiny, seed_everything

class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        self.global_d = GlobalDiscriminator()
        self.local_d = LocalDiscriminator()
        self.prior_d = PriorDiscriminator()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y, M, M_prime):

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf

        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand(-1, -1, 26, 26)

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha

        prior = torch.rand_like(y)

        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma

        return LOCAL + GLOBAL + PRIOR


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepInfomax pytorch')
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size

    # image size 3, 32, 32
    # batch size must be an even number
    # shuffle must be True
    # cifar_10_train_dt = CIFAR10('./data/cifar10',  download=True, transform=ToTensor())
    # cifar_10_train_l = DataLoader(cifar_10_train_dt, batch_size=batch_size, shuffle=True, drop_last=True,
    #                               pin_memory=torch.cuda.is_available())


    trainset = read_imagenet_tiny(data_path = "/home/ioaat57/projects/DIP_LFR/data/imagenet_tiny/image_tensor.bin")
    trainset = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True,
                                  pin_memory=torch.cuda.is_available())

    encoder = Encoder().to(device)
    loss_fn = DeepInfoMaxLoss().to(device)
    optim = Adam(encoder.parameters(), lr=1e-4)
    loss_optim = Adam(loss_fn.parameters(), lr=1e-4)

    epoch_restart = 0
    root = None

    for epoch in range(epoch_restart + 1, 20):
        batch = tqdm(trainset, total=len(trainset) // batch_size)
        train_loss = []
        for x in batch:
            x = x.to(device)

            optim.zero_grad()
            loss_optim.zero_grad()
            y, M = encoder(x)
            # rotate images to create pairs for comparison
            M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
            loss = loss_fn(y, M, M_prime)
            train_loss.append(loss.item())
            batch.set_description(str(epoch) + ' Loss: ' + str(stats.mean(train_loss[-20:])))
            loss.backward()
            optim.step()
            loss_optim.step()

    ## saving the encoder
    if not os.path.isdir("./stored"):
        os.mkdir("./stored")
    torch.save(encoder.state_dict(), os.path.join("./stored","DIM"+".pth"))
