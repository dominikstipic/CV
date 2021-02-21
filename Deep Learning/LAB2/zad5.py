import torch
from torch import nn
from zad3 import Flatten
import utils
from collections import OrderedDict
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import SGD
from zad4 import Vizual_Trainer


def multiclass_hinge_loss(logits, targets, delta):
     """
     Args:
        logits: torch.Tensor with shape (B, C), where B is batch size, and C is number of classes.
        target: torch.LongTensor with shape (B, ) representing ground truth labels.
     Returns:
        Loss as scalar torch.Tensor.
     """
     one_hot_targets = utils.one_hot(targets, logits.shape)
     true_logits = torch.masked_select(logits, one_hot_targets==1).view(-1,1)
     other_logits = torch.masked_select(logits, one_hot_targets==0).view(-1, one_hot_targets.shape[1]-1)
     losses = other_logits-true_logits+delta
     zeros = torch.zeros_like(losses)
     losses = torch.max(losses, zeros)
     losses = losses.sum(axis=1)
     return losses.mean()
             
model = nn.Sequential(OrderedDict([
              ("conv1", nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)),
              ("relu1", nn.ReLU()),
              ("pool1", nn.MaxPool2d(kernel_size=3, stride=2)),
              ("conv2", nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)),
              ("relu2", nn.ReLU()),
              ("pool2", nn.MaxPool2d(kernel_size=3, stride=2)),
              ("flatter", Flatten()),
              ("fc1", nn.Linear(in_features=1568, out_features=256)),
              ("relu3", nn.ReLU()),
              ("fc2", nn.Linear(in_features=256, out_features=128)),
              ("relu4", nn.ReLU()),
              ("fc3", nn.Linear(in_features=128, out_features=10))
              ]))

config = {}
config['max_epochs'] = 3
config['batch_size'] = 50
config['weight_decay'] = 1e-3
config['lr'] = 0.1
config["data_dir"] ='./datasets/CIFAR'
config["gamma"] = 0.9
config["delta"] = 1

train_dl, valid_dl, test_dl = utils.get_cifar_data(config)
optimizer = SGD
scheduler = ExponentialLR
loss = lambda logits, targets : multiclass_hinge_loss(logits, targets, delta=config["delta"])

model = Vizual_Trainer(model, (train_dl, valid_dl), optimizer, scheduler, loss, config).train()
#test_loss, (test_acc, test_precissions, test_recalls) = Vizual_Trainer.get_loss_acc(test_dl)  
#print(f"test_accuracy={test_acc}, precissions={test_precissions}, recalls={test_recalls}") 
