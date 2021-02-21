import torch
import pandas as pd
from collections import Counter
import itertools 
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class Vocab():
    def __init__(self, max_size=-1, min_freq=0, type="text", random_embedding=False):
        # Voacabular is build only for train data
        xs = list(pd.read_csv("./data/sst_train_raw.csv", names=["text","label"])[type])
        if type == "text":
            xs = [x.split() for x in xs]
            xs = list(itertools.chain(*xs))
        c = Counter(xs)
        most_common = c.most_common()
        xs = filter(lambda x : x[1] > min_freq, most_common)
        self.itos = list(map(lambda x : x[0].strip(), xs))
        if type == "text":
            self.itos = ["<PAD>", "<UNK>"] + self.itos
        if max_size != -1:
            self.itos = self.stoi[:max_size]
        self.stoi = dict(zip(self.itos, range(len(self.itos))))
        if type=="text":
            self.embedding = self.embedding_matrix(random_embedding)
            N,d = self.embedding.shape
            E = nn.Embedding(N, d)
            self.embedding = E.from_pretrained(torch.tensor(self.embedding), padding_idx = 0, freeze=False)
        
    def embedding_matrix(self, random_embedding=False):
        #embedding = np.random.normal(loc=0, scale=1, size=(len(self.itos), 300))
        embedding = torch.randn(len(self.itos), 300).numpy()
        if not random_embedding:
            d = {}
            with open("./data/sst_glove_6b_300d.txt","r") as f:
                for line in f:
                    xs = line.split()
                    word = xs[0]
                    del xs[0]
                    d[word] = np.array(xs)
            embedding[0] = np.zeros_like(embedding[0]) 
            for idx, token in enumerate(self.itos):
                vector = d.get(token)
                if vector is not None:
                    embedding[idx] = vector
        return embedding
                
    def encode(self, tokens):
        #codes = [self.stoi[token] for token in tokens.split() if token in self.stoi.keys()]
        codes = []
        for token in tokens.split():
            if token in self.stoi.keys():
                codes.append(self.stoi[token])
            else:
                codes.append(self.stoi["<UNK>"])
        return torch.tensor(codes)
    
    def decode(self, codes):
        tokens = [self.itos[code] for code in codes]
        return torch.tensor(tokens)



class NLPDataset(torch.utils.data.Dataset):
    def __init__(self, type, min_freq=0, max_size=-1, rand=False):
        path = f"./data/sst_{type}_raw.csv"            
        df = pd.read_csv(path, names=["text","label"])
        text,labels = [text.split() for text in list(df["text"])], list(df["label"])
        self.instances = list(zip(text, labels))
        self.text_vocab = Vocab(max_size=max_size, min_freq=min_freq, random_embedding=rand)
        self.label_vocab = Vocab(type="label")
                
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        text, label  = self.instances[idx]
        text = self.text_vocab.encode(" ".join(text))
        label = self.label_vocab.encode(label)
        return text,label
    
    def get_embeddings(self):
        return self.text_vocab.embedding

def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch) 
    lengths = torch.tensor([len(text) for text in texts]) 
    texts = pad_sequence(texts).T
    return texts, torch.tensor(labels), lengths

if __name__ == "__main__":
    train_dataset = NLPDataset(type="train")
    instance_text, instance_label = train_dataset.instances[3]
    print(f"Text: {instance_text}")
    print(f"Label: {instance_label}")
    numericalized_text, numericalized_label = train_dataset[3]
    print(f"Numericalized text: {numericalized_text}")
    print(f"Numericalized label: {numericalized_label}")
    
    batch_size = 2 # Only for demonstrative purposes
    shuffle = False # Only for demonstrative purposes
    train_dataset = NLPDataset(type="train")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                  shuffle=shuffle, collate_fn=pad_collate_fn)
    texts, labels, lengths = next(iter(train_dataloader))
    print(f"Texts: {texts}, shape:{texts.shape}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")
    

