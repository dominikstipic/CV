import torch
from torch import nn
from torch import optim
from ml import evaluate, train, load_dataset, vizualization_routine, AvgPool
        
config = {"lr" : 1e-4,
          "weight_decay" : 0,
          "max_epochs" : 5,
          "train_batch_size" : 10, 
          "valid_batch_size" : 32,
          "test_batch_size" : 32,
          "valid_ratio" : 0.1,
          "test_ratio" : 0.3,
          "print_time" : 10,
          "seed" :  7052020,
          "grad_clip" : 1,
          "min_freq" : 0}

train_dl, test_dl, valid_dl, embedding_matrix = load_dataset(config)
model = nn.Sequential(AvgPool(embedding_matrix),
                      nn.Linear(300, 150),
                      nn.ReLU(),
                      nn.Linear(150, 150),
                      nn.ReLU(),
                      nn.Linear(150, 2))
optimizer = optim.Adam
#loss = torch.nn.BCEWithLogitsLoss()
loss = torch.nn.CrossEntropyLoss()
trained_model, metrics = train(model, train_dl, valid_dl, loss, optimizer, config)
vizualization_routine(metrics)
print(evaluate(test_dl, trained_model, loss, config))



