import torch
from torch import optim
from torch import nn
from ml import load_dataset, evaluate, train, vizualization_routine, SentiModel, AvgPool
import pandas as pd


def build_senti_model(config, embedding_matrix):
    cell_type = "nn." + config["cell"].upper()
    cell = lambda i, h, l : eval(cell_type)(i, h, l, dropout=config["dropout"], bidirectional=config["bidirectional"])
    model = SentiModel(cell=cell, 
                   input_size=300, 
                   hidden_size=config["hidden_state"], 
                   rnn_layers=config["num_layers"], 
                   embedding_matrix=embedding_matrix)
    return model

def build_baseline(config, embedding_matrix):
    pipeline = [AvgPool(embedding_matrix)]
    last_output = 300
    for i in range(config["num_layers"]-1):
        linear = nn.Linear(last_output, config["hidden_state"])
        dropout = nn.Dropout(config["dropout"])
        relu   = nn.ReLU()
        pipeline.append(linear)
        pipeline.append(dropout)
        pipeline.append(relu)
        last_output = config["hidden_state"]
    last_linear = nn.Linear(last_output, 2)
    pipeline.append(last_linear)
    model = nn.Sequential(*pipeline)
    return model



def get_hiperparams(config):
    file = HIPER
    df = pd.read_csv(file)
    keys = list(df.keys())
    for i in range(len(df)):
        for key in keys:
            value = df[key][i]
            config[key] = value.tolist()
        yield config
    
config = {"lr" : 1e-4,
          "max_epochs" : 5,
          "train_batch_size" : 10, 
          "valid_batch_size" : 32,
          "test_batch_size" : 32,
          "print_time" : 100,
          "seed" :  7052020,
          "grad_clip" : 0.25,
          "min_freq" : 1, 
          
          "dropout" : 0.2,
          "bidirectional" : False, 
          "hidden_state" : 150,
          "num_layers" : 2,
          "cell" : "lstm"}

#train_dl, test_dl, valid_dl, embedding_matrix = load_dataset(config)
optimizer = optim.Adam
loss = torch.nn.CrossEntropyLoss()
HIPER = "./hiperparams1.csv"
############## HIPERPARAMETER OPTIMIZATION TASK1 ##############
# =============================================================================
# df = pd.read_csv(HIPER)
# df["acc"] = None
# df["precission"] = None
# df["recall"] = None
# df["F"] = None
# df["test_error"] = None
# for cell in ["rnn", "lstm", "gru"]:
#     config["cell"] = cell
#     for i, model_config in enumerate(get_hiperparams(config)):
#         train_dl, test_dl, valid_dl, embedding_matrix = load_dataset(config)
#         model = build_senti_model(model_config, embedding_matrix)
#         trained_model, train_metrics = train(model, train_dl, valid_dl, loss, optimizer, config)
#         #vizualization_routine(train_metrics)
#         metrics = evaluate(test_dl, trained_model, loss, config)
#         print(metrics)
#         test_loss, acc, p, r, f = metrics["loss"], metrics["accuracy"], metrics["precission"], metrics["recall"], metrics["f"]
#         df.loc[i,"test_error"] = test_loss
#         df.loc[i,"acc"] = acc
#         df.loc[i,"precission"] = p
#         df.loc[i,"recall"] = r
#         df.loc[i,"F"] = f
#     df.to_csv(f"hiperparams_{cell}.csv")
# =============================================================================

############## HIPERPARAMETER OPTIMIZATION TASK2 ##############
# =============================================================================
# df = pd.read_csv("./hiperparams.csv")
# df["acc"],df["precission"],df["recall"],df["F"],df["test_error"] = None, None, None, None, None
# config["cell"] = "lstm"
# for i, model_config in enumerate(get_hiperparams(config)):
#     model = build_senti_model(model_config, embedding_matrix)
#     trained_model, train_metrics = train(model, train_dl, valid_dl, loss, optimizer, config)
#     metrics = evaluate(test_dl, trained_model, loss, config)
#     print(metrics)
#     test_loss, acc, p, r, f = metrics["loss"], metrics["accuracy"], metrics["precission"], metrics["recall"], metrics["f"]
#     df.loc[i,"test_error"] = test_loss
#     df.loc[i,"acc"] = acc
#     df.loc[i,"precission"] = p
#     df.loc[i,"recall"] = r
#     df.loc[i,"F"] = f
# df.to_csv(f"results.csv")
# =============================================================================

############## HIPERPARAMETER OPTIMIZATION BASELINE ##############
# =============================================================================
# df = pd.read_csv("./hiperparams.csv")
# df["acc"],df["precission"],df["recall"],df["F"],df["test_error"] = None, None, None, None, None
# for i, model_config in enumerate(get_hiperparams(config)):
#     print(model_config)
#     baseline = build_baseline(config, embedding_matrix)
#     tm, train_metrics = train(baseline, train_dl, valid_dl, loss, optimizer, config)
#     metrics = evaluate(test_dl, tm, loss, config)
#     print(metrics)
#     test_loss, acc, p, r, f = metrics["loss"], metrics["accuracy"], metrics["precission"], metrics["recall"], metrics["f"]
#     df.loc[i,"test_error"] = test_loss
#     df.loc[i,"acc"] = acc
#     df.loc[i,"precission"] = p
#     df.loc[i,"recall"] = r
#     df.loc[i,"F"] = f
# df.to_csv(f"results_baseline.csv")
# =============================================================================


##### STATISTICAL SIGNIFICANCE #####
config = {"lr" : 1e-4,
          "max_epochs" : 5,
          "train_batch_size" : 10, 
          "valid_batch_size" : 32,
          "test_batch_size" : 32,
          "print_time" : 10,
          "seed" :  7052020,
          "grad_clip" : 0.25,
          "min_freq" : 1, 
          "dropout" : 0.6,
          "bidirectional" : True, 
          "hidden_state" : 100,
          "num_layers" : 3,
          "cell" : "lstm",
          "rand_embedding" : True}

seeds = [555,321,532,87,4]
m = []
for s in seeds:
    config["seed"] = s   
    train_dl, test_dl, valid_dl, embedding_matrix = load_dataset(config)
    senti_model = build_senti_model(config, embedding_matrix)
    trained_model,_ = train(senti_model, train_dl, valid_dl, loss, optimizer, config, log=True)
    metrics = evaluate(test_dl, trained_model, loss, config)
    print(metrics)
    m.append(metrics)
df = pd.DataFrame(m)
df.to_csv("x.csv")


# =============================================================================
# config = {"lr" : 1e-3,
#           "max_epochs" : 5,
#           "train_batch_size" : 10, 
#           "valid_batch_size" : 32,
#           "test_batch_size" : 32,
#           "print_time" : 10,
#           "seed" :  7052020,
#           "grad_clip" : 0.25,
#           "min_freq" : 3, 
#           "dropout" : 0.1,
#           "bidirectional" : True, 
#           "hidden_state" : 300,
#           "num_layers" : 3,
#           "cell" : "lstm",
#           "rand_embedding" : True}
# 
# m = []
# for s in [555,321,532,87,4]:
#     config["seed"] = s   
#     train_dl, test_dl, valid_dl, embedding_matrix = load_dataset(config)
#     baseline = build_baseline(config, embedding_matrix)
#     tm,_ = train(baseline, train_dl, valid_dl, loss, optimizer, config, log=True)
#     metrics = evaluate(test_dl, tm, loss, config)
#     print(metrics)
#     m.append(metrics)
# df = pd.DataFrame(m)
# df.to_csv("y.csv")
# =============================================================================





# =============================================================================
# config = {"lr" : 1e-4,
#           "max_epochs" : 5,
#           "train_batch_size" : 10, 
#           "valid_batch_size" : 32,
#           "test_batch_size" : 32,
#           "print_time" : 100,
#           "seed" :  7052020,
#           "grad_clip" : 0.25,
#           "min_freq" : 0, 
#           "dropout" : 0.1,
#           "bidirectional" : True, 
#           "hidden_state" : 100,
#           "num_layers" : 3,
#           "cell" : "rnn"}
# print("--------------")
# for s in seeds:
#     config["seed"] = s   
#     train_dl, test_dl, valid_dl, embedding_matrix = load_dataset(config)
#     senti_model = build_senti_model(config, embedding_matrix)
#     trained_model,_ = train(senti_model, train_dl, valid_dl, loss, optimizer, config, log=False)
#     metrics = evaluate(test_dl, trained_model, loss, config)
#     print(metrics)
# 
# 
# config = {"lr" : 1e-4,
#           "max_epochs" : 5,
#           "train_batch_size" : 10, 
#           "valid_batch_size" : 32,
#           "test_batch_size" : 32,
#           "print_time" : 100,
#           "seed" :  7052020,
#           "grad_clip" : 0.25,
#           "min_freq" : 0, 
#           "dropout" : 0.1,
#           "bidirectional" : True, 
#           "hidden_state" : 100,
#           "num_layers" : 4,
#           "cell" : "lstm"}
# print("--------------")
# for s in seeds:
#     config["seed"] = s   
#     train_dl, test_dl, valid_dl, embedding_matrix = load_dataset(config)
#     senti_model = build_senti_model(config, embedding_matrix)
#     trained_model,_ = train(senti_model, train_dl, valid_dl, loss, optimizer, config, log=False)
#     metrics = evaluate(test_dl, trained_model, loss, config)
#     print(metrics)
# =============================================================================
