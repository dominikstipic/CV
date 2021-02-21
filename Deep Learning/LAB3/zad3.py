import torch
from torch import nn
from torch import optim
from ml import load_dataset, evaluate, train, vizualization_routine


config = {"lr" : 1e-4,
          "max_epochs" : 5,
          "train_batch_size" : 10, 
          "valid_batch_size" : 32,
          "test_batch_size" : 32,
          "print_time" : 10,
          "seed" :  7052020,
          "grad_clip" : 0.25,
          "min_freq" : 1}

    
class SentiModel(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_layers, embedding_matrix):
        super(SentiModel, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=rnn_layers, dropout=0.2, bidirectional=False)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)

    def forward(self, input_batch):
        # embeddings : batch_size, sequence_len, embedding_vec 
        embeddings = self.embedding_matrix(input_batch)
        # embeddings : sequence_len, batch_size, embedding_vec 
        embeddings = embeddings.permute(1,0,2)
        
        # output : sequence_len, batch_size, hidden_size
        # hn     : num_layers, batch_size, hidden_size
        output, hn = self.rnn(embeddings)
        
        # hn : batch_size, num_layers, hidden_size
        hn = hn.permute(1,0,2)
        # hn : batch_size, hidden_size
        hn = hn.mean(axis=1)
        h1 = self.linear1(hn)
        h1 = nn.ReLU()(h1)
        h2 = self.linear2(h1)
        return h2
        


train_dl, test_dl, valid_dl, embedding_matrix = load_dataset(config)
model = SentiModel(300, 150, 2, embedding_matrix)
optimizer = optim.Adam
#loss = torch.nn.BCEWithLogitsLoss()
loss = torch.nn.CrossEntropyLoss()
trained_model, metrics = train(model, train_dl, valid_dl, loss, optimizer, config)
vizualization_routine(metrics)
print(evaluate(test_dl, trained_model, loss, config))

