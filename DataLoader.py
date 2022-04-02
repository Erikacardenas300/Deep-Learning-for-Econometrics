'''
xs         : numpy array of xs
ys         : numpy array of ys
batch_size : int - batch_size
'''
def get_batch(xs, ys, batch_size):
  import random
  import torch
  import numpy as np
  rand_idxs = random.sample(range(len(xs)), batch_size)
  x_batch, y_batch = [], []
  for idx in rand_idxs:
    x_batch.append(xs[idx])
    y_batch.append(ys[idx])
  
  x_batch = torch.from_numpy(np.array(x_batch))
  y_batch = torch.from_numpy(np.array(y_batch))
  return x_batch, y_batch
    
