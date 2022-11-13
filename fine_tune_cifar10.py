import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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



def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def test_model(model, cifar_10_train_t):

    model.eval()
    batch = tqdm(cifar_10_train_t, total=len(cifar_10_train_t.dataset) // args.batch_size)
    conf = np.zeros((10, 10))
    acc = 0
    with torch.no_grad():
        for x, target in batch:
            x = x.to(args.device)
            target = target.to(args.device)

            out = model(x)

            pred = torch.argmax(out, dim=1)
 
            acc += sum((pred - target)==0)

    return acc/len(cifar_10_train_t.dataset)


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('--device', default= 'cuda' if torch.cuda.is_available() 
                        else 'cpu', type=str, help='device')
    parser.add_argument('--lr', default=1e-3, help='learning rate')
    parser.add_argument('--n_epoch', default=10, type=int, help='epochs')
    parser.add_argument('--pretrain', default=None,
                        help='use pretrain weights')
    parser.add_argument('--fine_tune', default=True, type=bool, 
                        help='fine tune backbone')
    parser.add_argument('--seed', default=0, type=int, 
                        help='fine tune backbone')
    parser.add_argument('--save_dir', default='./stored', type=str, 
                        help='weight store directory')
    parser.add_argument('--name', default='no_pretrain_fine_tune', type=str, 
                        help='name the model')
    
    args = parser.parse_args()
    
    seed_everything(args.seed)

    trainset = CIFAR10('./data/cifar10',  train=True, download=True, 
                       transform=transforms.Compose([transforms.ToTensor(), 
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    testset = CIFAR10('./data/cifar10', train=False, download=True,
                      transform=transforms.Compose([transforms.ToTensor(), 
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
                       
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True,
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


        acc_test = test_model(model, testloader)
        print("test_acc:", acc_test)


        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)

    torch.save(model.state_dict(), os.path.join(args.save_dir,args.name+".pth"))
