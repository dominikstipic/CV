import torch.nn as nn
import torch
import torchvision
from torch.optim import SGD, lr_scheduler
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split

class Flatten(nn.Module):
    def forward(self, input):        
        N,C,H,W = input.shape
        return input.view(N,C*H*W)

class ConvolutionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size = 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size = 2, stride=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features = 32*7*7, out_features = 512),
            nn.Linear(in_features = 512, out_features = 10))
        self.reset_parameters()
        
    def forward(self, x):
        return self.model(x)
    
    def reset_parameters(self):
        modules = list(self.model)
        for m in modules:
          if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
          elif isinstance(m, nn.Linear) and m is not modules[-1]:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        modules[-1].reset_parameters()

def get_mnist_dataloader(config):
    batch_size  = config["batch_size"]
    save_dir    = config["save_dir"]
    mnist       = torchvision.datasets.MNIST(save_dir, train=True, download=True, transform=transforms.ToTensor())
    valid_size  = int(0.083333 * len(mnist))
    train_size  = len(mnist) - valid_size
    mnist_train, mnist_valid = random_split(mnist, [train_size, valid_size])
    mnist_test  = torchvision.datasets.MNIST(save_dir, train=False, download=True, transform=transforms.ToTensor())
    train_dl    = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_dl     = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
    valid_dl    = DataLoader(mnist_valid, batch_size=batch_size, shuffle=True)
    return train_dl, valid_dl, test_dl
  
def evaluate(name, dl, model, loss, config):
    print(f"** {name} **")
    batch_size = config["batch_size"]
    num_examples = len(dl)*batch_size 
    cnt_correct = 0 
    loss_avg = 0
    model.eval()
    with torch.no_grad():
        for inputs,targets in dl:
            logits = model(inputs)
            yp = logits.argmax(axis = 1)
            cnt_correct += (yp == targets).sum()
            loss_val = loss(logits, targets)
            loss_avg += loss_val
        acc = cnt_correct.item()/num_examples
        loss_avg /= batch_size
        print(f"accuracy={acc*100}, avg_loss={loss_avg}")
        return loss_avg.item()
            
            
def train(model, dataloaders, optimizer, loss, config, log=True):
    optimizer = optimizer(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]) 
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    train_dl, valid_dl = dataloaders
    num_examples = len(train_dl)*config["batch_size"]
    writer = SummaryWriter()
    k = 0
    
    epochs = config["max_epochs"]
    for epoch in range(1,epochs+1):
        model.train()
        print(f"   *lr={round(scheduler.get_lr()[0],4)}")
        training_loss = 0
        cnt_correct   = 0 
        for i, (input_batch, target_batch) in enumerate(train_dl):
            logits = model(input_batch)
            loss_val = loss(logits, target_batch)
            yp = logits.argmax(axis=1)
            cnt_correct += (yp == target_batch).sum().item()
            training_loss +=  loss_val.item()
            loss_val.backward()
            optimizer.step()
            model.zero_grad()
            if i % 10 == 0:
                print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*len(input_batch), num_examples, loss_val))
            if i % 100 == 0 and log:
                writer.add_scalar("Running batch train loss", loss_val, k)
                k += 1
                filters = list(model.model[0].parameters())[0].detach()
                img_grid = torchvision.utils.make_grid(filters)
                writer.add_image(f'Filters_running', img_grid)
            if i > 0 and i % 50 == 0:
                batch_size = config["batch_size"]
                train_acc = cnt_correct/((i+1)*batch_size)
                print("   Train accuracy = %.2f" % (train_acc*100))
        print("Train accuracy=%.2f" % (cnt_correct/num_examples*100))
        validation_loss = evaluate("Validation", valid_dl, model, loss, config)
        scheduler.step(epoch)
        training_loss /= config["batch_size"]
        if log:
            filters = list(model.model[0].parameters())[0].detach()
            img_grid = torchvision.utils.make_grid(filters)
            writer.add_image(f'Filters', img_grid)
            writer.add_scalars("Errors", {"train" : training_loss,
                                         "valid" : validation_loss}, epoch-1)
    writer.close()
    return model

###################################

if __name__ == "__main__":
    SAVE_DIR = Path(__file__).parent / 'out'
    config = {}
    config['max_epochs'] = 8
    config['batch_size'] = 50
    config['save_dir'] = SAVE_DIR
    config['weight_decay'] = 1e-3
    config['lr'] = 0.1
        
    model = ConvolutionalModel()
    optimizer = SGD
    loss = nn.CrossEntropyLoss()
    train_dl, valid_size, test_dl = get_mnist_dataloader(config)
    train(model, (train_dl, valid_size), optimizer, loss, config, log=True)
    evaluate("Test", test_dl, model, loss, config)

