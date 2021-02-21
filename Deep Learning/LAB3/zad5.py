import torch
from torch import optim
from torch import nn
from ml import load_dataset, evaluate, train, vizualization_routine, SentiModel
import pandas as pd
from dataLoaders import NLPDataset


def build_senti_model(config, embedding_matrix):
    cell_type = "nn." + config["cell"].upper()
    cell = lambda i, h, l : eval(cell_type)(i, h, l, dropout=config["dropout"], bidirectional=config["bidirectional"])
    model = SentiModel(cell=cell, 
                   input_size=300, 
                   hidden_size=config["hidden_state"], 
                   rnn_layers=config["num_layers"], 
                   embedding_matrix=embedding_matrix,
                   attention = config["attention"])
    return model

config = {"lr" : 1e-4,
          "max_epochs" : 10,
          "train_batch_size" : 10, 
          "valid_batch_size" : 32,
          "test_batch_size" : 32,
          "print_time" : 100,
          "seed" :  7052020,
          "grad_clip" : 0.25,
          "min_freq" : 0, 
          "dropout" : 0.2,
          "bidirectional" : False, 
          "hidden_state" : 150,
          "num_layers" : 2,
          "cell" : "rnn",
          "attention" : True}

optimizer = optim.Adam
loss = torch.nn.CrossEntropyLoss()
seeds = [555,321,532,87,4]

# =============================================================================
# for cell in ["gru","rnn","lstm"]:
#     config["cell"] = cell
#     m = []
#     for s in seeds:
#         config["seed"] = s   
#         train_dl, test_dl, valid_dl, embedding_matrix = load_dataset(config)
#         senti_model = build_senti_model(config, embedding_matrix)
#         trained_model,_ = train(senti_model, train_dl, valid_dl, loss, optimizer, config, log=False)
#         metrics = evaluate(test_dl, trained_model, loss, config)
#         print(metrics)
#         m.append(metrics)
#     df = pd.DataFrame(m)
#     df.to_csv(f"attention_{cell}.csv")
# =============================================================================

config["attention"] = False
for cell in ["gru","rnn","lstm"]:
    config["cell"] = cell
    m = []
    for s in seeds:
        config["seed"] = s   
        train_dl, test_dl, valid_dl, embedding_matrix = load_dataset(config)
        senti_model = build_senti_model(config, embedding_matrix)
        trained_model,_ = train(senti_model, train_dl, valid_dl, loss, optimizer, config, log=False)
        metrics = evaluate(test_dl, trained_model, loss, config)
        print(metrics)
        m.append(metrics)
    df = pd.DataFrame(m)
    df.to_csv(f"{cell}.csv")



##GRAPHS

# =============================================================================
# config = {"lr" : 1e-4,
#           "max_epochs" : 5,
#           "train_batch_size" : 10, 
#           "valid_batch_size" : 32,
#           "test_batch_size" : 32,
#           "print_time" : 100,
#           "seed" :  7052020,
#           "grad_clip" : 0.25,
#           "min_freq" : 1, 
#           "dropout" : 0.1,
#           "bidirectional" : False, 
#           "hidden_state" : 100,
#           "num_layers" : 3,
#           "cell" : "lstm",
#           "attention" : True}
# 
# train_dl, test_dl, valid_dl, embedding_matrix = load_dataset(config)
# senti_model = build_senti_model(config, embedding_matrix)
# 
# trained_model,_ = train(senti_model, train_dl, valid_dl, loss, optimizer, config, log=True)
# torch.save(trained_model.state_dict(),"model")
# =============================================================================

# =============================================================================
# senti_model.load_state_dict(torch.load("model"))
# train_dataset = NLPDataset(type="train")
# sen = list(zip(*train_dataset.instances[0:5]))[0]
# 
# for text in sen:
#     print(text)
#     
# xs = []
# for i in range(5):
#     v,_ = train_dataset[i]
#     xs.append(v)
# 
# A = []
# with torch.no_grad():
#     for x in xs:
#         x = x.reshape(1,-1)
#         attn = senti_model.get_attentions(x).numpy()
#         A.append(attn[0])
# 
# import numpy as np
# import matplotlib.pyplot as plt
# for text,a in zip(sen,A):
#     xs = np.arange(len(a))
#     plt.bar(xs,a)
#     plt.xticks(xs, text)
#     plt.xticks(rotation=90)
#     plt.show()
# =============================================================================

