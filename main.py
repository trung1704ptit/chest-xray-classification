import os
import torch
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from multiprocessing import Process, freeze_support
from tqdm.notebook import tqdm

import variables

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_transforms = transforms.Compose([transforms.Resize((224,224),interpolation=Image.NEAREST),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(*stats,inplace=True)
                                   ])
test_transforms = transforms.Compose([transforms.Resize((224,224),interpolation=Image.NEAREST),
                                      transforms.ToTensor(),
                                      transforms.Normalize(*stats)
                                  ])

dataset = datasets.ImageFolder(variables.TRAIN_DIR, transform=train_transforms)
test_data = datasets.ImageFolder(variables.VALID_DIR, transform=test_transforms)

print(dataset.classes)
print(len(dataset))

random_seed = 42
torch.manual_seed(random_seed)

val_pct = 0.1
val_size = int(val_pct * len(dataset))
train_size = len(dataset) - val_size


train_data, valid_data = random_split(dataset, [train_size, val_size])
len(train_data), len(valid_data)

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()
print(device)

train_dl = DataLoader(train_data, batch_size=variables.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
valid_dl = DataLoader(valid_data, batch_size=variables.BATCH_SIZE * 2, num_workers=2, pin_memory=True)
test_dl = DataLoader(test_data, batch_size=variables.BATCH_SIZE, num_workers=2, pin_memory=True)


def decode_label(label_number):
    if label_number==0:
        return "NORMAL"
    return "PNEUMONIA"

# def show_image(img_tuple):
#     plt.imshow(img_tuple[0].permute(1,2,0))
#     print("Label: ",decode_label(img_tuple[1]))
#
#
# show_image(dataset[0])

# def show_batch(dl, invert=False):
#     for images, labels in dl:
#         fig, ax = plt.subplots(figsize=(16, 8))
#         ax.set_xticks([]); ax.set_yticks([])
#         data = 1-images if invert else images
#         ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))
#         break
#
# show_batch(train_dl)
    
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def accuracy(outputs, labels):
    preds = [1 if pred>0.5 else 0 for pred in outputs]
    preds = torch.as_tensor(preds, dtype=torch.float32, device=device)
    preds = preds.view([torch.tensor(preds.shape).item(), 1])
    acc = torch.tensor(torch.sum(preds == labels).item() / len(preds), device=device)
    return acc

class BinaryClassificationBase(nn.Module):
    def training_step(self, batch):
        images, targets = batch 
        targets = torch.tensor(targets.clone().detach(), dtype=torch.float32, device=device)
        targets = targets.view([torch.tensor(targets.shape).item(),1])
        out = self(images)
        loss = F.binary_cross_entropy(out, targets) 
        return loss
    
    def validation_step(self, batch):
        images, targets = batch 
        targets = torch.tensor(targets.clone().detach(), dtype=torch.float32, device=device)
        targets = targets.view([torch.tensor(targets.shape).item(),1]) 
        out = self(images)
        loss = F.binary_cross_entropy(out, targets)
        acc = accuracy(out, targets)
        return {'val_loss': loss.detach(), 'val_acc': acc }
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        batch_scores = [x['val_acc'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()      
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_score.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

class PneumoniaCnnModel(BinaryClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet50(pretrained=True)
        n_features = self.network.fc.in_features
        self.network.fc = nn.Linear(n_features, 1)
    
    def forward(self, xb):        
        return torch.sigmoid(self.network(xb))

trn_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(valid_dl, device)
tst_dl = DeviceDataLoader(test_dl, device)


model = to_device(PneumoniaCnnModel(), device)


def try_batch(dl):
    for images, labels in dl:
        print('images.shape:', images.shape)
        out = model(images)
        print('out.shape:', out.shape)
        print('out[0]:', out[0])
        break

try_batch(trn_dl)


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, max_lr, model, train_loader, val_loader, weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            sched.step()
        
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


@torch.no_grad()
def predict_dl(dl, model):
    torch.cuda.empty_cache()
    batch_probs = []
    for xb, _ in tqdm(dl):
        probs = model(xb)
        batch_probs.append(probs.cpu().detach())
    batch_probs = torch.cat(batch_probs)
    return batch_probs

def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')

def training():
    history = [evaluate(model, val_dl)]

    num_epochs = 20
    max_lr = 1e-2
    opt_func = torch.optim.Adam
    grad_clip = 0.1
    weight_decay = 1e-4

    history += fit(num_epochs, max_lr, model, trn_dl, val_dl, weight_decay=weight_decay, grad_clip=grad_clip,
                   opt_func=opt_func)

    torch.save(model.state_dict(), 'chest-x-ray-resnet50-model.pth')


def predict():
    torch.multiprocessing.freeze_support()
    test_predictions = predict_dl(tst_dl, model)

    test_labels = []
    for _, label in test_data:
        test_labels.append(label)
    test_labels = torch.as_tensor(test_labels, dtype=torch.float32, device=device)
    test_labels = test_labels.view([torch.tensor(test_labels.shape).item(), 1])

    test_accuracy = accuracy(test_predictions, test_labels)
    print("Test Accuracy: ", test_accuracy.item())


def main():
    training()

if __name__ == '__main__':
    freeze_support()
    main()

