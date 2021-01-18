import torch
from torch.utils.data import random_split, DataLoader
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.nn as nn
from visualization import plot_lrs, plot_losses, plot_accuracies
from device_data_loader import DeviceDataLoader
from helper import to_device, get_default_device
import variables

train_transforms = transforms.Compose([transforms.Resize((512, 512), interpolation=Image.NEAREST),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(*variables.STATS, inplace=True)
                                   ])
test_transforms = transforms.Compose([transforms.Resize((512, 512), interpolation=Image.NEAREST),
                                      transforms.ToTensor(),
                                      transforms.Normalize(*variables.STATS)
                                  ])

dataset = datasets.ImageFolder(variables.TRAIN_DIR, transform=train_transforms)
test_data = datasets.ImageFolder(variables.VALID_DIR, transform=test_transforms)

torch.manual_seed(variables.RANDOM_SEED)

val_size = int(variables.VAL_PCT * len(dataset))
train_size = len(dataset) - val_size


train_data, valid_data = random_split(dataset, [train_size, val_size])
print(len(train_data), len(valid_data))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

train_dl = DataLoader(train_data, batch_size=variables.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
valid_dl = DataLoader(valid_data, batch_size=variables.BATCH_SIZE * 2, num_workers=8, pin_memory=True)
test_dl = DataLoader(test_data, batch_size=variables.BATCH_SIZE, num_workers=8, pin_memory=True)


def accuracy(outputs, labels):
    preds = [1 if pred > 0.5 else 0 for pred in outputs]
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
        for batch in train_loader:
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
    for xb, _ in dl:
        probs = model(xb)
        batch_probs.append(probs.cpu().detach())
    batch_probs = torch.cat(batch_probs)
    return batch_probs

def training():
    history = [evaluate(model, val_dl)]
    opt_func = torch.optim.Adam

    history += fit(variables.NUM_EPOCHS, variables.MAX_LR, model, trn_dl, val_dl, weight_decay=variables.WEIGHT_DECAY, grad_clip=variables.GRAD_CLIP,
                   opt_func=opt_func)

    plot_lrs(history)
    plot_accuracies(history)
    plot_losses(history)

    torch.save(model.state_dict(), variables.PATH)


def predict():
    model.load_state_dict(torch.load(variables.PATH))
    test_predictions = predict_dl(tst_dl, model)

    test_labels = []
    for _, label in test_data:
        test_labels.append(label)
    test_labels = torch.as_tensor(test_labels, dtype=torch.float32, device=device)
    test_labels = test_labels.view([torch.tensor(test_labels.shape).item(), 1])

    print(test_predictions, test_labels)

    test_accuracy = accuracy(test_predictions, test_labels)
    print("Test Accuracy: ", test_accuracy.item())


if __name__ == '__main__':
    training()
    # predict()

