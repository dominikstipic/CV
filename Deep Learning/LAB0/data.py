import numpy as np
import matplotlib.pyplot as plt

class Random2DGaussian:
  """Random bivariate normal distribution sampler

  Hardwired parameters:
      d0min,d0max: horizontal range for the mean
      d1min,d1max: vertical range for the mean
      scalecov: controls the covariance range 

  Methods:
      __init__: creates a new distribution

      get_sample(n): samples n datapoints

  """

  d0min=0 
  d0max=10
  d1min=0 
  d1max=10
  scalecov=5
  
  def __init__(self):
    dw0,dw1 = self.d0max-self.d0min, self.d1max-self.d1min
    mean = (self.d0min,self.d1min)
    mean += np.random.random_sample(2)*(dw0, dw1)
    eigvals = np.random.random_sample(2)
    eigvals *= (dw0/self.scalecov, dw1/self.scalecov)
    eigvals **= 2
    theta = np.random.random_sample()*np.pi*2
    R = [[np.cos(theta), -np.sin(theta)], 
         [np.sin(theta), np.cos(theta)]]
    Sigma = np.dot(np.dot(np.transpose(R), np.diag(eigvals)), R)
    self.get_sample = lambda n: np.random.multivariate_normal(mean,Sigma,n)
    

def graph_surface(function, rect, offset=0.5, width=256, height=256):
  """Creates a surface plot (visualize with plt.show)

  Arguments:
    function: surface to be plotted
    rect:     function domain provided as:
              ([x_min,y_min], [x_max,y_max])
    offset:   the level plotted as a contour plot

  Returns:
    None
  """

  lsw = np.linspace(rect[0][1], rect[1][1], width) 
  lsh = np.linspace(rect[0][0], rect[1][0], height)
  xx0,xx1 = np.meshgrid(lsh, lsw)
  grid = np.stack((xx0.flatten(),xx1.flatten()), axis=1)

  #get the values and reshape them
  values=function(grid).reshape((width,height))
  
  # fix the range and offset
  delta = offset if offset else 0
  maxval=max(np.max(values)-delta, - (np.min(values)-delta))
  
  # draw the surface and the offset
  plt.pcolormesh(xx0, xx1, values, 
     vmin=delta-maxval, vmax=delta+maxval)
    
  if offset != None:
    plt.contour(xx0, xx1, values, colors='black', levels=[offset])

def graph_data(X,Y_, Y, special=[]):
  """Creates a scatter plot (visualize with plt.show)

  Arguments:
      X:       datapoints
      Y_:      groundtruth classification indices
      Y:       predicted class indices
      special: use this to emphasize some points

  Returns:
      None
  """
  # colors of the datapoint markers
  palette=([0.5,0.5,0.5], [1,1,1], [0.2,0.2,0.2])
  colors = np.tile([0.0,0.0,0.0], (Y_.shape[0],1))
  for i in range(len(palette)):
    colors[Y_==i] = palette[i]

  # sizes of the datapoint markers
  sizes = np.repeat(20, len(Y_))
  sizes[special] = 40
  
  # draw the correctly classified datapoints
  good = (Y_==Y)
  plt.scatter(X[good,0],X[good,1], c=colors[good], 
              s=sizes[good], marker='o', edgecolor="k", alpha=0.7)

  # draw the incorrectly classified datapoints
  bad = (Y_!=Y)
  plt.scatter(X[bad,0],X[bad,1], c=colors[bad], 
              s=sizes[bad], marker='s', edgecolor="k", alpha=0.7)

def class_to_onehot(Y):
  Yoh=np.zeros((len(Y),max(Y)+1))
  Yoh[range(len(Y)),Y] = 1
  return Yoh

def eval_perf_binary(Y,Y_):
    TP = np.logical_and((Y == 1), (Y_ == 1)).sum()
    TN = np.logical_and((Y == 0), (Y_ == 0)).sum()
    FN = np.logical_and((Y == 0), (Y_ == 1)).sum()
    FP = np.logical_and((Y == 1), (Y_ == 0)).sum()
    
    acc = (TP+TN)/len(Y)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    return acc,precision,recall

def eval_perf_multi(Y, Y_):
    C = np.zeros((max(Y_)+1,max(Y_)+1))
    for i in range(len(Y)):
        y_pred = Y[i]
        y      = Y_[i]
        C[y_pred,y] += 1
    
    matrices = []
    classes = max(Y_)+1
    for i in range(classes):
        C_i = np.zeros((2,2))
        C_i[0,0] = C[i,i]
    
        horizontal = C[i, :]
        C_i[0,1] = np.delete(horizontal,i).sum()
        
        vertical = C[:,i]
        C_i[1,0] = np.delete(vertical,i).sum()
        
        A = np.delete(C, i, 0)
        A = np.delete(A, i, 1)    
        C_i[1,1] = A.sum()
        matrices.append(C_i)
        
    precisions = []
    recalls    = []

    for C_i in matrices:
        p = C_i[0,0]/(C_i[0,0]+C_i[0,1])
        r = C_i[0,0]/(C_i[0,0]+C_i[1,0])
        precisions.append(p)
        recalls.append(r)
        
    acc = (C[0,0] + C[1,1] + C[2,2])/C.sum()
    return C, acc, np.array(precisions), np.array(recalls)
  

def eval_AP(ranked_labels):
  """
      Recovers AP from ranked labels. 
      Algorithm assumes that label are sorted according to ascending scores
      
      Arguments:
          * ranked_labels : ascendingly sorted labels, np.array, 1xD
         
      Returns:
          * AP metric evaluation
  """
  ix = range(1,len(ranked_labels)+1)
  xs = np.cumsum(ranked_labels)
  xs[ranked_labels == 0] = 0
  return sum((xs/ix))/sum(ranked_labels)
  
def sample_gauss_2d(C, N):
    Gs = [Random2DGaussian() for i in range(C)]
    X = np.vstack([G.get_sample(N) for G in Gs])
    Y_ = np.hstack([[y]*N for y in range(C)])
    return X,Y_

def sample_gmm_2d(ncomponents, nclasses, nsamples):
  # create the distributions and groundtruth labels
  Gs=[]
  Ys=[]
  for i in range(ncomponents):
    Gs.append(Random2DGaussian())
    Ys.append(np.random.randint(nclasses))

  # sample the dataset
  X = np.vstack([G.get_sample(nsamples) for G in Gs])
  Y_= np.hstack([[Y]*nsamples for Y in Ys])
  
  return X,Y_

def myDummyDecision(X):
  scores = X[:,0] + X[:,1] - 5
  return scores

if __name__=="__main__":
  np.random.seed(100)
  
  # get data
  X,Y_ = sample_gmm_2d(4, 2, 30)
  # X,Y_ = sample_gauss_2d(2, 100)

  # get the class predictions
  Y = myDummyDecision(X)>0.5  

  # graph the decision surface
  rect=(np.min(X, axis=0), np.max(X, axis=0))
  graph_surface(myDummyDecision, rect, offset=0)
  
  # graph the data points
  graph_data(X, Y_, Y, special=[])

  plt.show()
    