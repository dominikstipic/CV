import torch
import os
import pickle
import numpy as np
import skimage as ski
import skimage.io
from torch.utils.data import DataLoader

def one_hot(y, shape):
    one_hot_targets = torch.zeros(*shape)
    one_hot_targets = one_hot_targets.scatter(1, y.view(-1, 1), 1)
    return one_hot_targets

def draw_image(img, mean, std):
  img = img.transpose(1,2,0)
  img *= std
  img += mean
  img = img.astype(np.uint8)
  ski.io.imshow(img)
  ski.io.show()

def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict

def get_worst_classifications(model, loss, dl, k):
    with torch.no_grad():
        losses = torch.empty(0)
        for inputs,targets in dl:
            logits = model(inputs)
            batch_losses = torch.tensor([loss(logits[i].view(1,-1), targets[i].view(1)).item() for i in range(len(targets))])
            losses = torch.cat((losses, batch_losses))
        _,ix = torch.topk(losses, k)
        return ix
    
def get_indexed_images(model, dl, ixs):
    imgs = []
    ixs = [ix.item() for ix in ixs]
    s = set(ixs)
    for i, (inputs,targets) in enumerate(dl):
        current_set = set([i for i in range(len(targets)*i,len(targets)*(i+1))])
        intersection = s.intersection(current_set)
        if intersection:
            xs = [(inputs[k-i*len(targets)].numpy(), model(inputs[k-i*len(targets)].view(1,3,32,32)).argmax()) for k in list(intersection)]
            for x in xs:
                imgs.append(x)
    return imgs
    
def get_cifar_data(config):
    DATA_DIR = config["data_dir"]
    img_height = 32
    img_width = 32
    num_channels = 3
    
    train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
    train_y = []
    for i in range(1, 6):
      subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
      train_x = np.vstack((train_x, subset['data']))
      train_y += subset['labels']
    train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1)
    train_y = np.array(train_y, dtype=np.int32)
    
    subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
    test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)
    
    valid_size = 5000
    train_x, train_y = shuffle_data(train_x, train_y)
    valid_x = train_x[:valid_size, ...]
    valid_y = train_y[:valid_size, ...]
    train_x = train_x[valid_size:, ...]
    train_y = train_y[valid_size:, ...]
    data_mean = train_x.mean((0,1,2))
    data_std = train_x.std((0,1,2))
    
    train_x = (train_x - data_mean) / data_std
    valid_x = (valid_x - data_mean) / data_std
    test_x = (test_x - data_mean) / data_std
    
    train_x = train_x.transpose(0,3,1,2)
    valid_x = valid_x.transpose(0,3,1,2)
    test_x = test_x.transpose(0,3,1,2)
    
    train_y = torch.tensor(train_y)
    valid_y = torch.tensor(valid_y)
    test_y  = torch.tensor(test_y)
    train_x = torch.tensor(train_x)
    valid_x = torch.tensor(valid_x)
    test_x  = torch.tensor(test_x)
    
    train_y = train_y.long()
    valid_y = valid_y.long()
    test_y  = test_y.long()
    
    train_dl = DataLoader(dataset=list(zip(train_x,train_y)), batch_size=config["batch_size"])
    test_dl  = DataLoader(dataset=list(zip(test_x,test_y)), batch_size=config["batch_size"])
    valid_dl = DataLoader(dataset=list(zip(valid_x,valid_y)), batch_size=config["batch_size"]) 
    return train_dl, valid_dl, test_dl
    

    
    
    
    

