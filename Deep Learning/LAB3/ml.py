import torch
import numpy as np
from torch.utils.data import DataLoader
from dataLoaders import NLPDataset, pad_collate_fn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn

def evaluate(loader, model, loss, config):
  model.eval()
  with torch.no_grad():
    valid_loss = 0
    y_pred, targets = np.array([]), np.array([])
    for i, (input_batch, target_batch, lens) in enumerate(loader):
        logits = model(input_batch).squeeze(1)
        #logits = model(input_batch,lens).squeeze(1)
        #loss_val = loss(logits, target_batch.float())
        loss_val = loss(logits, target_batch)
        valid_loss +=  loss_val.item()
        #yp = (1/(1+torch.exp(-logits)) > 0.5)*1
        yp = torch.argmax(F.softmax(logits,dim=1),dim=1)
        y_pred = np.append(y_pred, yp)
        targets = np.append(targets, target_batch.numpy())
    valid_loss /= len(loader)
    tp = np.logical_and((y_pred == 1), (targets == 1)).sum()
    tn = np.logical_and((y_pred == 0), (targets == 0)).sum()
    fn = np.logical_and((y_pred == 0), (targets == 1)).sum()
    fp = np.logical_and((y_pred == 1), (targets == 0)).sum()
    acc = (tp+tn)/(tp+tn+fp+fn)
    precission = tp/(tp+fp)
    recall = tp/(tp+fn)
    f = 2*(precission*recall)/(precission+recall)
    d = {"loss" : valid_loss, 
         "accuracy" : acc,
         "precission" : precission,
         "recall" : recall,
         "f" : f}
    # keys : loss, accuracy, precission, recall, f
    return d
    
def train(model, train_dl, valid_dl, loss, optimizer, config, log=True):
    #optimizer = optimizer(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]) 
    optimizer = optimizer(model.parameters(), lr=config["lr"]) 
    epochs = config["max_epochs"]
    error_dict = {"valid" : [], "train" : []}
    for epoch in range(1,epochs+1):
        model.train()
        training_loss = 0
        cnt_correct   = 0 
        for i, (input_batch, target_batch, lens) in enumerate(train_dl):
            #optimizer.zero_grad()
            model.zero_grad()
            logits = model(input_batch).squeeze(1)
            #batch_loss = loss(logits, target_batch.float())
            #y_pred = (1/(1+torch.exp(-logits)) > 0.5)*1
            batch_loss = loss(logits, target_batch)
            y_pred = torch.argmax(F.softmax(logits,dim=1),dim=1)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            cnt_correct += (y_pred == target_batch).sum().item()
            training_loss +=  batch_loss.item()
            if i % config["print_time"] == 0 and log:
                progress = round(((i+1)*config["train_batch_size"])/(config["train_batch_size"]*len(train_dl)),2)
                print(f"    epoch %d, progress %.2f, batch loss = %.2f" % (epoch, progress*100, batch_loss))
        training_loss /= len(train_dl)
        training_acc   = cnt_correct / (len(train_dl)*config["train_batch_size"])
        metrics = evaluate(valid_dl, model, loss, config)
        if log:
            print(f"*** epoch %d, train_loss=%.2f, train_acc=%.2f, %s" % (epoch, training_loss, training_acc, metrics))    
        # keys : loss, accuracy, precission, recall, f
        error_dict["valid"].append((metrics["loss"], metrics["accuracy"], metrics["precission"], metrics["recall"], metrics["f"]))
        error_dict["train"].append((training_loss, training_acc))
    return model, error_dict

def load_dataset(config):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    train_dataset = NLPDataset(type="train", min_freq = config["min_freq"], rand=config["rand_embedding"])
    test_dataset = NLPDataset(type="test", min_freq = config["min_freq"], rand=config["rand_embedding"])
    valid_dataset = NLPDataset(type="valid", min_freq = config["min_freq"], rand=config["rand_embedding"])
    train = DataLoader(dataset=train_dataset, batch_size=config["train_batch_size"], collate_fn=pad_collate_fn, shuffle=True)
    test = DataLoader(dataset=test_dataset, batch_size=config["test_batch_size"], collate_fn=pad_collate_fn, shuffle=True)
    valid = DataLoader(dataset=valid_dataset, batch_size=config["valid_batch_size"], collate_fn=pad_collate_fn, shuffle=True)
    return train, test, valid, train_dataset.get_embeddings()

def vizualization_routine(metrics):
    # keys : loss, accuracy, precission, recall, f
    train = metrics["train"]
    valid = metrics["valid"]
    train = list(zip(*train))
    valid = list(zip(*valid))
    
    plt.plot(train[0], label = "train_error")
    plt.plot(valid[0], label = "valid_error")
    plt.legend()
    plt.show()
    
    plt.plot(train[1], label = "train_acc")
    plt.plot(valid[1], label = "valid_acc")
    plt.legend()
    plt.show()
    
    plt.plot(valid[2], label = "valid_precission")
    plt.plot(valid[3], label = "valid_recall")
    plt.legend()
    plt.show()
    
    plt.plot(valid[4], label = "F score")
    plt.legend()
    plt.show()
    

        
class SentiModel(nn.Module):
    def __init__(self, cell, input_size, hidden_size, rnn_layers, embedding_matrix, attention=False):
        super(SentiModel, self).__init__()
        self.attention_flag = attention
        self.embedding_matrix = embedding_matrix
        self.rnn = cell(input_size, hidden_size, rnn_layers)
        if attention:
            self.attention = AttentionHead(hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)
        
    def get_attentions(self, input_batch):
        embeddings = self.embedding_matrix(input_batch)
        embeddings = embeddings.permute(1,0,2)
        if isinstance(self.rnn, nn.LSTM):
            output, (hn, cn) = self.rnn(embeddings)
        else:
            output, hn = self.rnn(embeddings)
       
        output = output.permute(1,0,2)
        distrib = self.attention.distribution(output)
        return distrib
        
    def forward(self, input_batch):
        # embeddings : batch_size, sequence_len, embedding_vec 
        embeddings = self.embedding_matrix(input_batch)
        # embeddings : sequence_len, batch_size, embedding_vec 
        embeddings = embeddings.permute(1,0,2)
        
        # output : sequence_len, batch_size, hidden_size
        # hn     : num_layers, batch_size, hidden_size
        if isinstance(self.rnn, nn.LSTM):
            output, (hn, cn) = self.rnn(embeddings)
        else:
            output, hn = self.rnn(embeddings)
       
        if self.attention_flag:
            output = output.permute(1,0,2)
            hn = self.attention(output)
        else:
            hn = hn.permute(1,0,2)
            # hn : batch_size, hidden_size
            hn = hn.mean(axis=1)
        
        h1 = self.linear1(hn)
        h1 = nn.ReLU()(h1)
        h2 = self.linear2(h1)
        return h2

class AvgPool(nn.Module):
    def __init__(self, embedding):
        super(AvgPool, self).__init__()
        self.E = embedding

    def forward(self, instances):
        representations = self.E(instances)
        return representations.mean(axis=1).float()

class AttentionHead(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionHead, self).__init__()
        self.linear1  = nn.Linear(hidden_size, hidden_size//2, bias=False)
        self.linear2  = nn.Linear(hidden_size//2, 1, bias=False)

    def distribution(self, hidden_states):
        h = self.linear1(hidden_states)
        attn = self.linear2(h).squeeze(2) 
        probs = F.softmax(attn, dim=1)
        return probs

    def forward(self, hidden_states):
        # hidden_states : batch_size, seq_length, wordvec
        h = self.linear1(hidden_states)
        attn = self.linear2(h).squeeze(2) 
        probs = F.softmax(attn, dim=1)
        probs = probs.reshape(probs.shape[0],probs.shape[1],1)
        hidden = (probs*hidden_states).sum(dim=1)
        return hidden    
    
