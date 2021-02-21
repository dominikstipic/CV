import torch
import numpy as np
import torch.nn as nn
from zad3 import Flatten
import utils
from collections import OrderedDict
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import SGD
from pt_train import Trainer
import torchvision


def create_confusion_matrix(y_pred, y_true, C):
    confusion_matrix = np.zeros((C,C))
    for model_class in range(C):
        for true_class in range(C):
            xs = y_pred == model_class
            ys = y_true == true_class
            confusion_matrix[true_class, model_class] = np.logical_and(xs,ys).sum()   
    return confusion_matrix

def evaluate(y_pred, y_true):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    confusion_matrix = create_confusion_matrix(y_pred, y_true, y_true.max()+1)
    acc = np.diag(confusion_matrix).sum()/confusion_matrix.sum()
    precissions, recalls = [],[]
    C = y_true.max()+1
    for i in range(C):
        y_true_i = (y_true == i)*1
        y_pred_i = (y_pred == i)*1
        matrix_i = create_confusion_matrix(y_pred_i, y_true_i, 2)
        precission_denominator = matrix_i[1,1]+matrix_i[0,1]
        recall_denominator = matrix_i[1,1]+matrix_i[1,0]
        if precission_denominator == 0:
            precissions.append(0.)
        else:    
            p_i = matrix_i[1,1]/precission_denominator
            precissions.append(p_i)
        if recall_denominator == 0:
            recalls.append(0.)
        else:    
            r_i = matrix_i[1,1]/recall_denominator
            recalls.append(r_i)
    return acc, np.array(precissions), np.array(recalls)


cifar_clasess_code = {0 : "airplane", 
                      1 : "automobile",
                      2 : "bird",
                      3 : "cat",
                      4 : "deer",
                      5 : "dog",
                      6 : "frog",
                      7 : "horse",
                      8 : "ship",
                      9 : "truck"}

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
config['max_epochs'] = 10
config['batch_size'] = 50
config['weight_decay'] = 1e-3
config['lr'] = 0.1
config["data_dir"] ='./datasets/CIFAR'
config["gamma"] = 0.9

train_dl, valid_dl, test_dl = utils.get_cifar_data(config)
optimizer = SGD
scheduler = ExponentialLR
loss = nn.CrossEntropyLoss()

class Vizual_Trainer(Trainer):
    def get_loss_acc(dl):
        loss_avg = 0
        y_pred, y_true = torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long) 
        for inputs,targets in dl:
            logits = model(inputs)
            loss_val = loss(logits, targets)
            loss_avg += loss_val
            y = logits.argmax(axis=1).flatten()
            y_pred = torch.cat((y_pred, y))
            y_true = torch.cat((y_true, targets.flatten()))
        loss_avg /= len(dl)
        performance = evaluate(y_pred, y_true)
        return loss_avg, performance
    
    def epoch_vizulization(self, writer, epoch):
        with torch.no_grad():
            model, train_dl, valid_dl, optimizer, scheduler, loss = self.ml
            filters = list(model[0].parameters())[0].detach()
            img_grid = torchvision.utils.make_grid(filters)
            
            validation_loss, (valid_acc, valid_precissions, valid_recalls) = Vizual_Trainer.get_loss_acc(valid_dl)  
            train_loss, (train_acc, train_precissions, train_recalls) = Vizual_Trainer.get_loss_acc(train_dl)  
            
            writer.add_scalars("Metrics/Cross-entropy loss", {"Train" : train_loss,
                                                              "Valid" : validation_loss}, epoch-1)
            writer.add_scalar("Metrics/Learning rate", scheduler.get_lr()[0], epoch-1)
            writer.add_scalars("Metrics/Avarage class accuracy", {"Train" : train_acc,
                                                                 "Valid" : valid_acc}, epoch-1)
            writer.add_scalars("Metrics/Avarage valid precission-recall", {"Precission" : valid_precissions.mean(),
                                                                           "Recall" : valid_recalls.mean()}, epoch-1)
            writer.add_image(f'Filters', img_grid)
    
    def iteration_vizualization(self, writer, epoch):
        pass
    
    
    
if __name__ == "__main__":
    model = Vizual_Trainer(model, (train_dl, valid_dl), optimizer, scheduler, loss, config).train()
    #test_loss, (test_acc, test_precissions, test_recalls) = Vizual_Trainer.get_loss_acc(test_dl)  
    #print(f"test_accuracy={test_acc}, precissions={test_precissions}, recalls={test_recalls}") 
    ix = utils.get_worst_classifications(model, loss, test_dl, 5)
    worst_images = utils.get_indexed_images(model, test_dl, ix)
    
    for img, y in worst_images:
        print(f"predvidena klasa={cifar_clasess_code[y.item()]}")
        utils.draw_image(img.copy(), 93.2067, 65.91576)
