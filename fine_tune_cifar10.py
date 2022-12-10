import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from torch.optim import Adam, SGD 
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import statistics as stats
import numpy as np
from models import ClassificationModel, Net
from utils import *

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('--device', default= 'cuda' if torch.cuda.is_available() 
                        else 'cpu', type=str, help='device')
    parser.add_argument('--lr', default=1e-3, help='learning rate')
    parser.add_argument('--n_epoch', default=20, type=int, help='epochs')
    # parser.add_argument('--pretrain', default='/home/ioaat57/projects/DIP_LFR/stored/dAE.pth',
    # parser.add_argument('--pretrain', default='/home/ioaat57/projects/DIP_LFR/stored/DIM.pth',
    parser.add_argument('--pretrain', default=None,
                        help='use pretrain weights')
    parser.add_argument('--fine_tune', default=True, type=bool, 
                        help='fine tune backbone')
    parser.add_argument('--use_subset', default=False , type=bool, 
                        help='use 5% of the available training data')
    parser.add_argument('--seed', default=0, type=int, 
                        help='seed')
    parser.add_argument('--save_dir', default='./stored', type=str, 
                        help='weight store directory')
    parser.add_argument('--name', default='no_pretrain_no_fine_tune', type=str, 
                        help='name the model')
    
    args = parser.parse_args()
    
    trainset = CIFAR10('./data/cifar10',  train=True, download=True, 
                       transform=transforms.ToTensor())

    if args.use_subset:
        subset_indices = torch.LongTensor(np.random.choice(len(trainset), len(trainset) // 20, replace=False)) 
        trainset.data = trainset.data[subset_indices]
        trainset.targets = [trainset.targets[index] for index in subset_indices]


    # trainset = read_imagenet_tiny(data_path = "/home/ioaat57/projects/DIP_LFR/data/imagenet_tiny/image_tensor.bin")

    testset = CIFAR10('./data/cifar10', train=False, download=True,
                                             transform=transforms.ToTensor())

                       
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                  pin_memory=torch.cuda.is_available())
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                  pin_memory=torch.cuda.is_available())


    model = ClassificationModel(pretrain=args.pretrain, fine_tune=args.fine_tune).to(args.device)
    optim = Adam(model.parameters(), lr=args.lr)

    
    criterion  = torch.nn.CrossEntropyLoss()   
    ### Training 
    model.train()
    flag = True
    for epoch in range(args.n_epoch):


        batch = tqdm(trainloader, total=len(trainloader.dataset) // args.batch_size)
        train_loss = []
        model.train()

        for x, target in batch:


            # import matplotlib.pyplot as plt

            # plt.imsave("img_cifar.png",x.data.cpu().numpy()[0].transpose(1,2,0))

            x = x.to(args.device)
            target = target.to(args.device)

            optim.zero_grad()

            out = model(x)
            # pred = torch.argmax(out, dim=1)

            loss = criterion(out, target)
            train_loss.append(loss.item())
            
            loss.backward()
            optim.step()
            # print("epoch: {} train_loss: {}".format(epoch, stats.mean(train_loss[-20:])))
            batch.set_description(str(epoch) + ' Loss: ' + str(stats.mean(train_loss[-20:])))


        acc_test = test_model(model, testloader, args.batch_size, args.device)
        print("test_acc:", acc_test)


        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)

    torch.save(model.state_dict(), os.path.join(args.save_dir,args.name+".pth"))
