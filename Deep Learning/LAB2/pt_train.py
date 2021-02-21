from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod


class Trainer(ABC):
    def __init__(self, model, dataloaders, optimizer, scheduler, loss, config, evaluate=None, log=True):
        optimizer = optimizer(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        scheduler = scheduler(optimizer, gamma=config["gamma"])
        train_dl, valid_dl = dataloaders
        self.ml = [model, train_dl, valid_dl, optimizer, scheduler, loss]
        self.config = config
    
    @abstractmethod
    def iteration_vizualization(self, writer, epoch):
        pass
    
    @abstractmethod
    def epoch_vizulization(self, writer, epoch):
        pass
    
    def train(self):
        model, train_dl, valid_dl, optimizer, scheduler, loss = self.ml
        num_examples = len(train_dl)*self.config["batch_size"]
        writer = SummaryWriter()
        epochs = self.config["max_epochs"]
        for epoch in range(1,epochs+1):
            model.train()
            print(f"   *lr={round(scheduler.get_lr()[0],4)}")
            self.training_loss = 0
            cnt_correct   = 0 
            for i, (input_batch, target_batch) in enumerate(train_dl):
                logits = model(input_batch)
                loss_val = loss(logits, target_batch)
                yp = logits.argmax(axis=1)
                cnt_correct += (yp == target_batch).sum().item()
                self.training_loss +=  loss_val.item()
                loss_val.backward()
                optimizer.step()
                model.zero_grad()
                if i % 10 == 0:
                    print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*len(input_batch), num_examples, loss_val))
                if i % 100 == 0 :
                    self.iteration_vizualization(writer, epoch)
                if i > 0 and i % 50 == 0:
                    batch_size = self.config["batch_size"]
                    train_acc = cnt_correct/((i+1)*batch_size)
                    print("   Train accuracy = %.2f" % (train_acc*100))
            print("Train accuracy=%.2f" % (cnt_correct/num_examples*100))
            self.training_loss /= self.config["batch_size"]
            self.epoch_vizulization(writer, epoch)
            scheduler.step(epoch)
        writer.close()
        return model
        
