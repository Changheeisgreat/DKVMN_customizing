import json
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn.init
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def varible(tensor, gpu):
    if gpu >= 0:
        return torch.autograd.Variable(tensor)
    else:
        return torch.autograd.Variable(tensor)

def to_scalar(var):
    return var.view(-1).data.tolist()[0]

def save_data(data,filename):
    with open(filename+'.pickle', 'wb') as f:
        pickle.dump(data, f)
        
def save_pred(pred,filename):
    pd.DataFrame(pred,columns=['correct']).to_csv(filename+'.csv',
                                                  index = False)
    
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def trimf (x, params):
    if not isinstance(params, list):
        raise TypeError ("Params should be a list with three values.")
    a = params[0]
    b = params[1]
    c = params[2]
    result = 0.0

    if x <= a:
        result = 0.0,0

    elif a <= x <= b:
        result = (x-a)/(b-a),1

    elif b <= x <= c:
        result = (c-x)/(c-b),2

    elif c <= x:
        result = 0,0
    return result

def pca_kmeans_tri(tensor):
    batch=[]
    for i in range(tensor.shape[0]):
        array=tensor[i].detach().numpy() 
        pca=PCA(n_components=1)
        pca.fit(array)
        array=pca.transform(array)
        
        kmeans=KMeans(n_clusters=3,random_state=0)
        kmeans.fit(array)
        centers=kmeans.cluster_centers_
        
        params=sorted(centers.reshape(1,-1).tolist()[0])
        tmp=np.empty((0),dtype=int)
        for j in range(array.shape[0]):
            tmp=np.concatenate((tmp,(trimf(array[j],params))),axis=0)
        batch.append(tmp.reshape(1,-1))
        
        x=np.vstack(batch).astype(np.float)

    return torch.from_numpy(x).float() #(32,1090)
