import torch
from tqdm import tqdm
import numpy as np
import random, os


def read_imagenet_tiny(data_path):

    data = np.fromfile(data_path, dtype=np.uint8).astype(np.float32)/255
    data = data.reshape(100000,3,32,32).transpose(0,1,3,2)

    return data



def test_model(model, testloader, batch_size, device):

    model.eval()
    batch = tqdm(testloader, total=len(testloader.dataset) // batch_size)
    conf = np.zeros((10, 10))
    acc = 0
    with torch.no_grad():
        for x, target in batch:
            x = x.to(device)
            target = target.to(device)

            out = model(x)

            pred = torch.argmax(out, dim=1)
 
            acc += sum((pred - target)==0)

    return acc/len(testloader.dataset)


def seed_everything(seed: int):

    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
