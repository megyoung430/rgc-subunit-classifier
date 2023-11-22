#!/usr/bin/env python
# coding: utf-8

# # **Exploration of Neural Network Models for Identification of RGC Receptive Field Subunits**

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os
from matplotlib import pyplot as plt
import numpy as np
from scipy import io as spio
from scipy.stats.mstats import mquantiles
from scipy import optimize
from tqdm.notebook import trange
from google.colab import files
import h5py
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
os.chdir('/content/drive/MyDrive/Colab Notebooks')
from fitellipse import fitellipse
import random
import math


# In[ ]:


pip install shapely


# In[ ]:


import shapely.geometry as shpgeom
import shapely.ops as shpops
from shapely import centroid


# In[ ]:


# Only needed for stimulus generation
os.chdir('/')
get_ipython().run_line_magic('run', 'setup.py build_ext --inplace')
from randpy import ranb


# In[ ]:


'/content/drive/MyDrive/Colab Notebooks/cell_data_01_NC.mat' #data path


# In[ ]:


'/content/drive/MyDrive/Colab Notebooks/Stim Data' #stim path for both regular and frozen stimuli


# In[ ]:


'/content/drive/MyDrive/Colab Notebooks/STA & Crop' #save path


# In[ ]:


STA_filename = 'Cell ' + str(cell_num) + ' Uncropped Spatial STA' + '.h5'


# # **Section 0: Main**

# In[ ]:


def subunit_model(cell_num, ver, data_path, stim_path, stim_save_path=None, STA_path=None,
                  subunits_save_path=None, sigma=3, node_num=60,
                  batch_size=25, learn_rate=0.5, num_epochs=100, stop_thr=0, L1_reg=0):

  """
  This is the main function that runs the entire pipeline. For a given cell, the function:
    1. Calculates the spatial and temporal STAs, both normalized to the Euclidian norm.
    2. Crops the spatial STA.
    3. Generates the stimulus ensembles for every frame as well as the associated label (spike vs. no spike)
    4. Divides the stimulus ensembles into a training set, a validation set, and a testing set.
    5. Creates a sampler to oversample the spike-generating ensembles when creating batches to address the class imbalance between the spikes and no spikes.
    6. Trains and then tests the neural network.
    7. Extracts the weights of the nodes.
    8. Builds a subunit model and evaluates its performance on the frozen noise section of the spatiotemporal white noise images.

  Inputs:
    cell_num (int): cell of interest
    ver (int): number of the version (when doing repeated trainings)
    data_path (str): filepath where data is stored
    stim_path (str): filepath where the stimulus "images" are stored (see: gen_STA)
    stim_save_path (str): filepath where the stimulus "images" should be saved (see: gen_STA)
    STA_path (str): filepath where the saved STA is 
    sigma (float): (see: crop_STA)
    node_num (int): number of nodes in the hidden layer of the network (see: SpikeClassifier). Default is 60 nodes.
    batch_size (int): batch size used during training (see: run_model)
    learn_rate (float): learning rate used to train the model (see: run_model). Default is 0.5.
    num_epochs (int): number of epochs for training (see: run_model). Default is 100 epochs.
    stop_thr (float): threshold for minimum improvement in the loss for terminating training before reaching num_epochs.
    L1_reg (float): coefficient for L1 regularization (see: run_model). Default is no L1 regularization (coefficient of 0).

  Outputs:
    model: trained model
    subunits (matrix of shape node_num x pixel_num): 
    crop (slice): indices used to crop both the STA and the STEs
    cropped_STA.shape[0] (int): size of the width of the cropped image
    cropped_STA.shape[1] (int): size of the height of the cropped image
  """

  # Screen resolution
  screen_x = 800
  screen_y = 600

  # Stimulus pixel ("stixel") size, i.e. how many screen pixels are one stixel
  stixel_w = 4
  stixel_h = 4

  # Actual stimulus dimensions
  x = int(np.ceil(screen_x / stixel_w))
  y = int(np.ceil(screen_y / stixel_h))

  # Stimulus update rate
  refresh_rate = 30  # Hz
  dt = 1 / refresh_rate  # Seconds

  # Random seed to generate the stimulus sequence
  running_seed = -10000

  # The length of the STA filter
  filter_length_sec =  0.67  # 600 ms
  filter_length = int(filter_length_sec / dt)  # Number of frames

  # Retrieve the spiking data
  spikecounts = spio.loadmat(data_path)['spk1']
  num_cells, num_frames, num_trials = spikecounts.shape

  # Select one cell
  spikecounts = spikecounts[cell_num]
  
  # Generate the spatial and temporal STA
  spatial_STA_filename = 'Cell ' + str(cell_num) + ' Uncropped Spatial STA' + '.h5'
  temp_STA_filename = 'Cell ' + str(cell_num) + ' Uncropped Temp STA' + '.h5'
  STA_filename = 'Cell ' + str(cell_num) + ' Uncropped STA.h5'
  STA, spatial_STA, temp_STA = gen_STA(x, y, spikecounts, cell_num=cell_num, filter_length=filter_length, num_frames=num_frames, num_trials=num_trials, stim_path=stim_path, STA_path=STA_path, STA_filename=STA_filename, spatial_STA_filename=spatial_STA_filename, temp_STA_filename=temp_STA_filename)

  # Crop the spatial STA
  cropped_STA, crop, ell_pixels = crop_STA(spatial_STA=spatial_STA, sigma_coeff=sigma)

  # Generate the stimulus ensembles and the associated label (i.e., 0 for no spike and 1 for spike)
  SEs, labels = gen_SEs(x, y, spikecounts, temp_STA=temp_STA, STA_cropped=cropped_STA, crop=crop, ell_pixels=ell_pixels, filter_length=filter_length, num_frames=num_frames, num_trials=num_trials)

  # Divide the SEs into a training set, a validation set, and a testing set
  x_train, x_test, y_train, y_test = train_test_split(SEs,labels)
  x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train)

  # Format the three sets for the neural network
  traindataset = MyDataset(x_train,y_train)
  testdataset = MyDataset(x_test,y_test)
  validdataset = MyDataset(x_valid,y_valid)

  # Create the neural network
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  if torch.cuda.is_available():
    device_name = torch.device("cuda")
  else:
      device_name = torch.device('cpu')
  print("Using {}.".format(device_name))

  model = Spike_Classifier(pixel_num=SEs.shape[1], node_num=node_num, p=dropout_p)
  model = model.to(device)

  # Reset weights if training multiple times
  for layer in model.children():
    if hasattr(layer,'reset_parameters'):
      layer.reset_parameters()

  # Weight the classes appropriately
  num_no_spikes = np.where(y_train==0)[0].shape[0]
  num_spikes = np.where(y_train==1)[0].shape[0]
  class_counts = [num_no_spikes, num_spikes]
  num_samples = sum(class_counts)

  class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
  weights = [class_weights[y_train[i]] for i in range(int(num_samples))]
  sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

  # Train the model
  model, losses, accuracies = run_model(model, running_mode='train',
                                        train_set=traindataset, valid_set=validdataset, test_set=None, sampler=sampler,
                                        batch_size=batch_size, learning_rate=learn_rate,
                                        n_epochs=num_epochs, stop_thr=stop_thr,
                                        L1_coeff=L1_reg, L2_coeff=L2_reg,
                                        shuffle=False,
                                        dynamic_lr=dynamic_lr, scheduler_step=scheduler_step, scheduler_gamma=scheduler_gamma, print_output=True)
  
  # Test the model
  test_loss, test_accuracy = run_model(model, running_mode='test', train_set=None, valid_set=None, test_set=testdataset, sampler=None,
                                       batch_size=batch_size, learning_rate=learn_rate,
                                       n_epochs=num_epochs, stop_thr=stop_thr,
                                       L1_coeff=L1_reg, L2_coeff=L2_reg)

  for name, param in model.named_parameters():
    if name == 'layer1.weight':
      recon_param = np.zeros(shape=(node_num,cropped_STA.shape[0]*cropped_STA.shape[1]))
      param = param.cpu().detach().numpy()
      recon_param[:,ell_pixels.astype(bool).flatten()] = param
  node_weights = recon_param

  subunits = get_subunits(node_weights, cropped_STA.shape[0], cropped_STA.shape[1])

  if subunits_save_path is not None:
    os.chdir(subunits_save_path)
    if ver == 0:
      filename = "Cell " + str(cell_num) + ", " + str(node_num) + " Nodes, " + str(learn_rate) + " LR, " + str(L1_reg) + " L1 Coeff, " + str(num_epochs) + " Epochs Subunits" + '.h5'
    elif ver != 0:
      filename = "Cell " + str(cell_num) + ", " + str(node_num) + " Nodes, " + str(learn_rate) + " LR, " + str(L1_reg) + " L1 Coeff, " + str(num_epochs) + " Epochs Subunits (" + str(ver) + ").h5"
    with h5py.File(filename, mode='w') as f:
        f.create_dataset('subunits', data=subunits, compression=3)

  STAc = STA.reshape(x,y,-1)[crop]
  STAc = STAc.reshape(cropped_STA.shape[0]*cropped_STA.shape[1],-1)
  params_STA = gen_STA_model(x, y, spikecounts, crop, cropped_STA.shape[0], cropped_STA.shape[1], filter_length, num_frames, num_trials, STAc, stim_path=stim_path)
  preds_STA, corr_coef_STA, actual_fr, time = get_STA_predictions(cell_num, params_STA, STAc, crop, cropped_STA.shape[0], cropped_STA.shape[1], stim_frz_path=stim_path)

  params_subunit, weights = gen_subunit_model(cell_num, subunits, x, y, spikecounts, filter_length, num_frames, num_trials, spatial_STA, temp_STA, crop, cropped_STA.shape[0], cropped_STA.shape[1])
  preds_subunit, corr_coef_subunit, actual_fr, time = get_subunit_predictions(cell_num, params_subunit, weights, num_frames, temp_STA, crop, cropped_STA.shape[0], cropped_STA.shape[1], stim_frz_path=stim_path)

  return model, subunits, crop, cropped_STA.shape[0], cropped_STA.shape[1]


# # **Section 1: Implement Main**
# 

# In[ ]:


data_path = '/content/drive/MyDrive/Colab Notebooks/cell_data_01_NC.mat'
stim_path = '/content/drive/MyDrive/Colab Notebooks/Stim Data'
# subunits_save_path = '/content/drive/MyDrive/Colab Notebooks/Subunits/All Cells'
subunits_save_path = None
STA_path = '/content/drive/MyDrive/Colab Notebooks/STA & Crop'
# node_nums = [7, 9, 10, 15, 20, 25, 30, 40, 50, 70, 75, 80, 90, 100]
node_num = 60
# learn_rates = [0.025, 0.05, 0.075, 0.1, 0.25, 0.75, 1]
learn_rate = 0.5
# Relevant range: 2.5e-4, 5e-4, 7.5e-4
# L1_regs = [1e-6, 2.5e-6, 5e-6, 7.5e-6, 1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4]
L1_reg = 1e-4
# curr_L1_regs = [7.5e-5, 1e-4, 2.5e-4, 5e-4, 5e-5]
batch_size = 25
# num_epochs = [5, 10, 15, 20, 25, 30, 40, 50, 75, 125, 150, 175, 200]
num_epoch = 100
ver = 0

"""
for cell in range(36,39):
# for node_num in node_nums:
  for ver in range(0,20):
# for L1_reg in curr_L1_regs:
    try:
      os.chdir('/content/drive/MyDrive/Colab Notebooks/Subunits/All Cells')
      if ver == 0:
        filename = "Cell " + str(cell) + ", 60 Nodes, 0.5 LR, 0.0001 L1 Coeff, 100 Epochs Subunits.h5"
      else:
        filename = "Cell " + str(cell) + ", 60 Nodes, 0.5 LR, 0.0001 L1 Coeff, 100 Epochs Subunits (" + str(ver) + ").h5"
    
      with h5py.File(filename, mode='r') as f:
        subunits = f['subunits'][:,:]
    except:
      model, recon_param, subunits, crop, crop_x, crop_y = subunit_model(cell_num=cell, ver=ver, data_path=data_path, stim_path=stim_path, 
                                                                      STA_path=STA_path, subunits_save_path=subunits_save_path, sigma=3,
                                                                      node_num=node_num, batch_size=batch_size, learn_rate=learn_rate, 
                                                                      num_epochs=num_epoch, L1_reg=L1_reg, dynamic_lr=False)
    else:
      continue
"""

cell = 0
model, recon_param, subunits, crop, crop_x, crop_y = subunit_model(cell_num=cell, ver=ver, data_path=data_path, stim_path=stim_path, 
                                                                      STA_path=STA_path, subunits_save_path=subunits_save_path, sigma=3,
                                                                      node_num=node_num, batch_size=batch_size, learn_rate=learn_rate, 
                                                                      num_epochs=num_epoch, L1_reg=L1_reg, dynamic_lr=False)


# In[ ]:


fig, axs = plt.subplots(math.ceil(node_num/2),2)
counter1 = 0
counter2 = 0
for i in range(node_num):
    mm = np.abs(recon_param[i,:]).max()
    image = axs[counter1,counter2].imshow(recon_param[i,:].reshape(crop_x,crop_y).T,origin='lower',cmap='bwr', vmin=-mm, vmax=mm)
    if counter2 == 1:
      counter1 = counter1 + 1
      counter2 = 0
    else:
      counter2 = counter2 + 1
fig.set_figheight(100)
fig.set_figwidth(15)
os.chdir('/content/drive/MyDrive/Colab Notebooks')
figname = "Cell 0 All Nodes (1).pdf"
plt.savefig(figname)
files.download(figname)


# In[ ]:


num_subunits = subunits.shape[0]
for i in range(num_subunits):
  curr = i
  curr_subunit = subunits[curr,:].reshape(crop_x,crop_y)
  from fitellipse import fitellipse
  ell = fitellipse(curr_subunit, sigma=1.5)
  y_max = round(max(ell[1]))
  y_min = round(min(ell[1]))
  x_max = round(max(ell[0]))
  x_min = round(min(ell[0]))

  """
  ell_ref = np.zeros(shape=(ell[0].shape[0],2))
  ell_ref[:,0] = np.round(ell[0]-x_min)
  ell_ref[:,1] = np.round(ell[1]-y_min)
  ell_ref = ell_ref.astype(int)

  ell_pixels = np.zeros(shape=(crop_x,crop_y))
  y_range = np.unique(ell_ref[:,1])
  for i in range(max(ell_ref[:,1])):
    exists = i in y_range
    if exists:
      x_coords = np.where(ell_ref[:,1]==i)[0]
      curr_min_x = min(ell_ref[x_coords,0])
      curr_max_x = max(ell_ref[x_coords,0])
      ell_pixels[curr_min_x:curr_max_x,i] = 1
    else:
      continue
  for i in range(max(ell_ref[:,1])):
    if i == 0 or i == max(ell_ref[:,1])-1:
      continue
    if np.all(ell_pixels[:,i] == 0):
      ell_pixels[:,i] = ell_pixels[:,i-1]

  num_pixels = np.where(ell_pixels.flatten() == 1)[0].shape[0]
  print(num_pixels)
  """

  mm = np.abs(curr_subunit).max()
  #fig = plt.figure()
  plt.imshow(curr_subunit.T, origin='lower', cmap='bwr', vmin=-mm, vmax=mm)
  plt.plot(ell[0],ell[1])
  figname = "Cell " + str(cell) + " All Subunits (1).pdf"
  #figname = "Cell " + str(cell) + ", " + str(node_num) + " Nodes, " + str(learn_rate) + " LR, " + str(L1_reg) + " L1 Coeff  Subunit " + str(curr) + " (1).pdf"
  #figname = "Cell " + str(cell_num) + ", " + str(node_num) + " Nodes, " + str(learn_rate) + " LR, " + str(L1_reg) + " L1 Coeff, " + str(num_epochs) + " Epochs All Subunits (0).pdf"
  plt.savefig(figname)
  files.download(figname)


# # **Aside: Half and Half**

# In[ ]:


cell_num = 0
data_path = '/content/drive/MyDrive/Colab Notebooks/Lab Rotation/cell_data_01_NC.mat'
stim_path = '/content/drive/MyDrive/Colab Notebooks/Lab Rotation/Stim Data'
subunits_save_path = '/content/drive/MyDrive/Colab Notebooks/Lab Rotation/Subunits'
node_num = 60
learn_rate = 0.5
L1_reg = 1e-4
batch_size = 25
num_epochs = 100
sigma = 3

# Screen resolution
screen_x = 800
screen_y = 600

# Stimulus pixel ("stixel") size, i.e. how many screen pixels are one stixel
stixel_w = 4
stixel_h = 4

# Actual stimulus dimensions
x = int(np.ceil(screen_x / stixel_w))
y = int(np.ceil(screen_y / stixel_h))

# Stimulus update rate
refresh_rate = 30  # Hz
dt = 1 / refresh_rate  # Seconds

# Random seed to generate the stimulus sequence
running_seed = -10000

# The length of the STA filter
filter_length_sec =  0.67  # 600 ms
filter_length = int(filter_length_sec / dt)  # Number of frames

# Retrieve the spiking data
spikecounts = spio.loadmat(data_path)['spk1']
num_cells, num_frames, num_trials = spikecounts.shape

# Select one cell
spikecounts = spikecounts[cell_num]

# Generate the spatial and temporal STA
spatial_STA, temp_STA = gen_STA(x, y, spikecounts, cell_num=cell_num, filter_length=filter_length, num_frames=num_frames, num_trials=num_trials, stim_path=stim_path)

# Crop the spatial STA
cropped_STA, crop, ell_pixels = crop_STA(spatial_STA=spatial_STA, sigma_coeff=sigma)

# Generate the stimulus ensembles and the associated label (i.e., 0 for no spike and 1 for spike)
SEs, labels = gen_SEs(x, y, spikecounts, temp_STA=temp_STA, STA_cropped=cropped_STA, crop=crop, ell_pixels=ell_pixels, filter_length=filter_length, num_frames=num_frames, num_trials=num_trials)


# In[ ]:


total = (num_frames-filter_length)*num_trials
all_ind = np.asarray(list(range(0,total)))
firsthalf_ind = random.sample(range(0,total),math.floor(total/2.5))
firsthalf_ind.sort()
firsthalf_ind = np.asarray(firsthalf_ind)
secondhalf_ind = np.setdiff1d(all_ind, firsthalf_ind)
firsthalf_SEs = SEs[firsthalf_ind,:]
firsthalf_labels = labels[firsthalf_ind]
secondhalf_SEs = SEs[secondhalf_ind,:]
secondhalf_labels = labels[secondhalf_ind]


# In[ ]:


for k in range(3,20):
  total = (num_frames-filter_length)*num_trials
  all_ind = np.asarray(list(range(0,total)))
  firsthalf_ind = random.sample(range(0,total),math.floor(total/4))
  firsthalf_ind.sort()
  firsthalf_ind = np.asarray(firsthalf_ind)
  secondhalf_ind = np.setdiff1d(all_ind, firsthalf_ind)
  firsthalf_SEs = SEs[firsthalf_ind,:]
  firsthalf_labels = labels[firsthalf_ind]
  secondhalf_SEs = SEs[secondhalf_ind,:]
  secondhalf_labels = labels[secondhalf_ind]
  half_SEs = firsthalf_SEs
  half_labels = firsthalf_labels

  # Divide the SEs into a training set, a validation set, and a testing set
  x_train, x_test, y_train, y_test = train_test_split(half_SEs,half_labels)
  x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train)

  # Format the three sets for the neural network
  # traindataset = MyDataset(half_SEs,half_labels)
  traindataset = MyDataset(x_train,y_train)
  testdataset = MyDataset(x_test,y_test)
  validdataset = MyDataset(x_valid,y_valid)

  # Create the neural network
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = Spike_Classifier(pixel_num=half_SEs.shape[1], node_num=node_num, p=0)
  model = model.to(device)

  # Reset weights if training multiple times
  for layer in model.children():
    if hasattr(layer,'reset_parameters'):
      layer.reset_parameters()

  # Weight the classes appropriately
  num_no_spikes = np.where(y_train==0)[0].shape[0]
  num_spikes = np.where(y_train==1)[0].shape[0]
  class_counts = [num_no_spikes, num_spikes]
  num_samples = sum(class_counts)

  class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
  weights = [class_weights[y_train[i]] for i in range(int(num_samples))]
  sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

  # Train the model
  model, losses, accuracies = run_model(model, running_mode='train',
                                        train_set=traindataset, valid_set=validdataset, test_set=testdataset, sampler=sampler,
                                        batch_size=batch_size, learning_rate=learn_rate,
                                        n_epochs=num_epochs, stop_thr=1e-5,
                                        L1_coeff=L1_reg, L2_coeff=0,
                                        shuffle=False,
                                        dynamic_lr=False, scheduler_step=0, scheduler_gamma=0, print_output=True)

  for name, param in model.named_parameters():
    if name == 'layer1.weight':
      recon_param = np.zeros(shape=(node_num,cropped_STA.shape[0]*cropped_STA.shape[1]))
      param = param.cpu().detach().numpy()
      recon_param[:,ell_pixels.astype(bool).flatten()] = param
  node_weights = recon_param

  subunits = get_subunits(node_weights, cropped_STA.shape[0], cropped_STA.shape[1], 0.25)

  if subunits_save_path is not None:
    os.chdir(subunits_save_path)
    filename = "Cell " + str(cell_num) + ", " + str(node_num) + " Nodes, " + str(learn_rate) + " LR, " + str(L1_reg) + " L1 Coeff Subunits -- FIRST QUARTER (" + str(k) + ").h5"
    with h5py.File(filename, mode='w') as f:
        f.create_dataset('subunits', data=subunits, compression=3)


# In[ ]:


fig, axs = plt.subplots(math.ceil(node_num/2),2)
counter1 = 0
counter2 = 0
for i in range(node_num):
    mm = np.abs(recon_param[i,:]).max()
    image = axs[counter1,counter2].imshow(recon_param[i,:].reshape(cropped_STA.shape[0],cropped_STA.shape[1]).T,origin='lower',cmap='bwr', vmin=-mm, vmax=mm)
    if counter2 == 1:
      counter1 = counter1 + 1
      counter2 = 0
    else:
      counter2 = counter2 + 1
fig.set_figheight(15)
fig.set_figwidth(10)


# In[ ]:


num_subunits = subunits.shape[0]
print(num_subunits)


# In[ ]:


for i in range(num_subunits):
  curr = i
  curr_subunit = subunits[curr,:].reshape(cropped_STA.shape[0],cropped_STA.shape[1])

  from fitellipse import fitellipse
  ell = fitellipse(curr_subunit, sigma=1.5)

  """
  y_max = round(max(ell[1]))
  y_min = round(min(ell[1]))
  x_max = round(max(ell[0]))
  x_min = round(min(ell[0]))

  ell_ref = np.zeros(shape=(ell[0].shape[0],2))
  ell_ref[:,0] = np.round(ell[0]-x_min)
  ell_ref[:,1] = np.round(ell[1]-y_min)
  ell_ref = ell_ref.astype(int)

  ell_pixels = np.zeros(shape=(cropped_STA.shape[0],cropped_STA.shape[1]))
  y_range = np.unique(ell_ref[:,1])
  for i in range(max(ell_ref[:,1])):
    exists = i in y_range
    if exists:
      x_coords = np.where(ell_ref[:,1]==i)[0]
      curr_min_x = min(ell_ref[x_coords,0])
      curr_max_x = max(ell_ref[x_coords,0])
      ell_pixels[curr_min_x:curr_max_x,i] = 1
    else:
      continue
  for i in range(max(ell_ref[:,1])):
    if i == 0 or i == max(ell_ref[:,1])-1:
      continue
    if np.all(ell_pixels[:,i] == 0):
      ell_pixels[:,i] = ell_pixels[:,i-1]

  num_pixels = np.where(ell_pixels.flatten() == 1)[0].shape[0]
  """

  mm = np.abs(curr_subunit).max()
  plt.imshow(curr_subunit.T, origin='lower', cmap='bwr', vmin=-mm, vmax=mm)
  plt.plot(ell[0],ell[1])

# figname = "Cell 0 Two Fifth Data Example.pdf"
# plt.savefig(figname)
# files.download(figname)


# In[ ]:


if subunits_save_path is not None:
  os.chdir(subunits_save_path)
  filename = "Cell " + str(cell_num) + ", " + str(node_num) + " Nodes, " + str(learn_rate) + " LR, " + str(L1_reg) + " L1 Coeff Subunits -- FIRST TWO FIFTHS (21)" + '.h5'
  with h5py.File(filename, mode='w') as f:
      f.create_dataset('subunits', data=subunits, compression=3)


# # **Section 2: STA**

# In[ ]:


def gen_STA(x, y, spikecounts, cell_num, filter_length, num_frames, num_trials, STA_path=None, STA_filename=None, 
            spatial_STA_filename=None, temp_STA_filename=None, stim_gen=False, stim_path=None, running_seed=None, 
            save_stim=False, save_stim_path=None, save_STA=False, save_STA_path=None):

  """
  This function generates the temporal and spatial STAs for a particular cell.
  Both the temporal and spatial STAs are normalized to the Euclidean norm in this function.

  Inputs:
    x: Stimulus x dimension.
    y: Stimulus y dimension.
    spikecounts: Spiking data for the cell of interest.
    cell_num: number of the cell of interest
    filter_length: length of the STA filter
    num_frames: number of stimulus frames
    num_trials: number of trials
    stim_gen: Set to true if you want to generate the stimulus from scratch (i.e., with ranb). 
              Set to false if you have saved the stimulus data and would like to import it instead (much faster).
    stim_path: Only needed if stim_gen=False. Path name to where the stimulus data is saved.
    running_seed: Only need if stim_gen=True. Random seed needed for ranb to generate the stimulus.
    save_stim: Set to true if you want to save the stimulus after you generate it. stim_gen must also be set to true.
    save_stim_path: Only needed if save_stim_path=True. Path name to where you would like the stimulus saved.
    save_STA: Set to true if you want to save both the temporal and spatial STAs.
    save_STA_path: Only needed if save_STA=True. Path name to where you would like the temporal and spatial STAs saved.

  Outputs:
    STA: full STA
    spatial_STA: spatial STA normalized to Euclidean norm
    temp_STA: temporal STA normalized to Euclidean norm
  """
  
  if STA_path is not None:
    os.chdir(STA_path)
    
    with h5py.File(spatial_STA_filename, mode='r') as f:
        spatial_STA = f['spatial_STA'][:,:]

    with h5py.File(temp_STA_filename, mode='r') as f:
        temp_STA = f['temp_STA'][:]
    
    with h5py.File(STA_filename, mode='r') as f:	
        STA = f['STA'][:]
  else:
    if stim_gen:
      # The random seed is updated after each iteration
      seed = running_seed
    ST_sums = np.zeros(shape=(x*y,filter_length,num_trials))
    total_spikes = 0

    # Iterate over the stimulus blocks
    for trial in range(num_trials):
        
        if stim_gen:
          # Recreate the stimulus
          stim, seed = ranb(seed, x * y * num_frames)
          stim = np.asarray(stim, dtype='int8')  # Turn boolean to integers
          stim = stim.reshape((y*x, num_frames), order='F')  # Re-shape array
          stim[stim == 0] = -1  # Change values to 1 and -1
        else:
          os.chdir(stim_path)
          filename = 'stim' + str(trial).zfill(4) + '.h5'
          with h5py.File(filename, mode='r') as f:
              stim = f['stim'][:,:]

        if stim_gen and save_stim:
          # Save the stimulus
          filename = 'stim' + str(trial).zfill(4) + '.h5'
          os.chdir(save_stim_path)
          with h5py.File(filename, mode='w') as f:
              f.create_dataset('stim', data=stim, compression=3)
        
        curr_spikecounts = spikecounts[:,trial]
        spike_idx = np.where(curr_spikecounts != 0)[0]
        spike_vals = curr_spikecounts[spike_idx]

        ST = np.zeros(shape=(x*y,filter_length,spike_idx.shape[0]))
        for spike in range(spike_vals.shape[0]):
          
          # Check to see if the spike occurs before the first 20 frames
          if spike_idx[spike] < filter_length:
            continue
          else:
            # Otherwise, just take the last 20 frames
            # and weight those frames by the number of spikes
            frames = spike_vals[spike]*stim[:,(spike_idx[spike]-filter_length):spike_idx[spike]]
            ST[:,:,spike] = frames

            # Keep track of all spikes
            total_spikes = total_spikes + spike_vals[spike]
        
        ST_sums[:,:,trial] = np.sum(ST,axis=2)

    STA = np.sum(ST_sums,axis=2)/total_spikes

    STA_reshaped = STA.reshape(x, y, -1)

    max_idx = np.abs(STA_reshaped).argmax()
    max_x, max_y, max_t = np.unravel_index(max_idx, STA_reshaped.shape)

    spatial_STA = STA_reshaped[:, :, max_t].copy()
    spatial_STA /= np.linalg.norm(spatial_STA)

    temp_STA = STA_reshaped[max_x, max_y].copy()
    temp_STA /= np.linalg.norm(temp_STA)

    if save_STA:
      os.chdir(save_STA_path)
      filename = 'Cell' + str(cell_num) + 'SpatialSTA' + '.h5'
      with h5py.File(filename, mode='w') as f:
          f.create_dataset('spatial_STA', data=spatial_STA, compression=3)

      filename = 'Cell' + str(cell_num) + 'TempSTA' + '.h5'
      with h5py.File(filename, mode='w') as f:
          f.create_dataset('temp_STA', data=temp_STA, compression=3)

  return STA, spatial_STA, temp_STA


# In[ ]:


def crop_STA(spatial_STA, sigma_coeff=1.5):

  """
  This function crops the spatial_STA to reduce the number of inputs to the neural network.
  To crop the spatial STA, this function first fits a 2D gaussian to the spatial STA and extracts an elliptical contour of the fitted Gaussian.

  Dependencies: Requires the fitellipse function from fitellipse.py.
  
  Inputs:
    spatial_STA: spatial STA normalized to the Euclidean norm
    sigma_coeff: sigma value for the elliptical contour of the fitted Gaussian

  Outputs:
    STA_cropped_full:
    crop:
    ell_pixels: 
  """

  ell = fitellipse(spatial_STA, sigma=sigma_coeff)
  y_max = round(max(ell[1]))
  y_min = round(min(ell[1]))
  x_max = round(max(ell[0]))
  x_min = round(min(ell[0]))
  crop = (slice(x_min, x_max, None), slice(y_min, y_max, None))

  STA_cropped = spatial_STA[crop]
  crop_x = STA_cropped.shape[0]
  crop_y = STA_cropped.shape[1]
  
  ell_ref = np.zeros(shape=(ell[0].shape[0],2))
  ell_ref[:,0] = np.round(ell[0]-x_min)
  ell_ref[:,1] = np.round(ell[1]-y_min)
  ell_ref = ell_ref.astype(int)

  ell_pixels = np.zeros(shape=(crop_x,crop_y))
  y_range = np.unique(ell_ref[:,1])
  for i in range(max(ell_ref[:,1])):
    exists = i in y_range
    if exists:
      x_coords = np.where(ell_ref[:,1]==i)[0]
      curr_min_x = min(ell_ref[x_coords,0])
      curr_max_x = max(ell_ref[x_coords,0])
      ell_pixels[curr_min_x:curr_max_x,i] = 1
    else:
      continue
  for i in range(max(ell_ref[:,1])):
    if i == 0 or i == max(ell_ref[:,1])-1:
      continue
    if np.all(ell_pixels[:,i] == 0):
      ell_pixels[:,i] = ell_pixels[:,i-1]

  STA_cropped_full = np.zeros(shape=(crop_x*crop_y))
  STA_cropped_full[ell_pixels.astype(bool).flatten()] = STA_cropped.flatten()[ell_pixels.astype(bool).flatten()]
  STA_cropped_full = STA_cropped_full.reshape(crop_x,crop_y)

  return STA_cropped_full, crop, ell_pixels


# # **Section 3: STE**

# In[ ]:


def gen_SEs(x, y, spikecounts, temp_STA, STA_cropped, crop, ell_pixels, filter_length, num_frames, num_trials, stim_gen=False):

  """
  This function generates the stimulus ensembles for all frames across all trials for a particular cell.

  Inputs:
    x: Stimulus x dimension.
    y: Stimulus y dimension.
    spikecounts: Spiking data for the cell of interest.
    temp_STA: Temporal STA normalized to the Euclidean norm.
    STA_cropped:
    crop:
    ell_pixels: 
    filter_length:
    num_frames:
    num_trials:
  
  Outputs:
    ell_SEs:
    labels:
  """

  crop_x = STA_cropped.shape[0]
  crop_y = STA_cropped.shape[1]
  cropped_SEs = np.zeros(shape=(crop_x*crop_y,(num_frames-filter_length)*num_trials))
  labels = []
  counter = 0

  # Iterate over the stimulus blocks
  for trial in range(num_trials):
      
      if stim_gen:
        # Recreate the stimulus
        stim, seed = ranb(seed, x * y * num_frames)
        stim = np.asarray(stim, dtype='int8')  # Turn boolean to integers
        stim = stim.reshape((y*x, num_frames), order='F')  # Re-shape array
        stim[stim == 0] = -1  # Change values to 1 and -1
        stim = stim[crop]
        stim = stim.reshape(crop_x*crop_y,num_frames)
      else:
        os.chdir(stim_path)
        # Recreate the cropped stimulus
        filename = 'stim' + str(trial).zfill(4) + '.h5'
        with h5py.File(filename, mode='r') as f:
            stim = f['stim'][:,:]
            stim = stim.reshape(x,y,num_frames)
            stim = stim[crop]
            stim = stim.reshape(crop_x*crop_y,num_frames)

      for frame in range(filter_length,num_frames):
        # Generate the SE for that frame
        frames = stim[:,(frame-filter_length):frame]
        frames = temp_STA*frames
        curr_SE = np.mean(frames,axis=1)

        # Store the SE
        cropped_SEs[:,counter] = curr_SE.flatten()
        if spikecounts[frame,trial] == 0:
          labels.append(0)
        else:
          labels.append(1)
        counter = counter + 1
  
  ell_SEs = []
  for i in range(cropped_SEs.shape[1]):
    ell_SEs.append(cropped_SEs[ell_pixels.astype(bool).flatten(),i])
  ell_SEs = np.asarray(ell_SEs)

  labels = np.asarray(labels)

  return ell_SEs, labels


# # **Section 4: Neural Network**

# In[ ]:


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return (x, y)

    def __len__(self):
        return self.x.shape[0]


# In[ ]:


class Spike_Classifier(nn.Module):
    
  """
  This is the class that creates a neural network for determining presence/absence of a spike from a "spike"-triggered ensemble

  Network architecture:
  - Input layer
  - First hidden layer: fully connected layer of size node_num nodes
  - Output layer: a linear layer with one node, representing the spike probability

  Activation functions: rectified linear activation function for the hidden layer and sigmoidal activation function for the output layer.
  Reference for choosing appropriate activation function: https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
  """
  
  def __init__(self, pixel_num=0, node_num=0):
      super(Spike_Classifier, self).__init__()
      # For the first layer, the input is the flattened ESE and the output is the number of hidden nodes
      self.layer1 = nn.Linear(in_features=pixel_num,out_features=node_num)
      # For the output layer, the input is the number of hidden nodes from the first layer and the output is the spiking probability
      self.layer2 = nn.Linear(in_features=node_num,out_features=1)

  def forward(self, input):
      # Rectified linear activation function for the hidden layer
      layer1_output = F.relu(self.layer1(input))
      # Sigmoidal activation function for the output layer
      spike_prob = torch.sigmoid(self.layer2(layer1_output))
      return spike_prob


# In[ ]:


def _train(model, data_loader, optimizer, L1_coeff):

  """
  This function implements one epoch of training a neural network.

  Inputs:
    model: the neural network to be trained
    data_loader: for loading the network input and labels from the training dataset
    optimizer: the optimiztion method, e.g., SGD
    L1_coeff: L1 weight decay factor 

  Outputs:
    model: the trained model
    train_loss: average loss value on the entire training dataset
    train_accuracy: average accuracy on the entire training dataset

  Reference for loss function for binary classification: https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
  """
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  train_loss = 0
  total = 0
  correct = 0

  for i, data in enumerate(data_loader, 0):
    # Data is a batch of featuresets and labels
    SEs, labels = data
    SEs, labels = SEs.to(device),labels.to(device)

    # Zero the gradient at the beginning of a new batch
    optimizer.zero_grad()

    # Pass the data to the model
    spike_probs = model(SEs.float())

    # Calculate the loss
    regularization_loss = 0
    for name, param in model.named_parameters():
        if name == 'layer1.weight':
          regularization_loss = regularization_loss + torch.sum(abs(param))
    classify_loss = nn.BCELoss()(spike_probs.flatten(),labels.flatten().float())
    loss = classify_loss + L1_coeff * regularization_loss

    # Backpropagate the loss
    loss.backward()

    # Adjust the weights
    optimizer.step()
    
    # If the spike probability is greater than 0.5, assign it a 1 for "spike"
    # If the spike probability is less than 0.5, assign it a 0 for "no spike"
    predicted = torch.round(spike_probs.data.flatten())
    correct += predicted.eq(labels.data).sum().item()
    total += labels.nelement()
    train_loss += loss.item()
   
  average_loss = train_loss/len(data_loader)
  average_accuracy = correct/total

  return model, average_loss, average_accuracy


# In[ ]:


def _test(model, data_loader, L1_coeff):
	
  """
  This function evaluates a trained neural network on a validation set or a testing set. 

  Inputs:
    model: trained neural network
    data_loader: for loading the network input and targets from the validation or testing dataset

  Output:
    test_loss: average loss value on the entire validation or testing dataset 
    test_accuracy: percentage of correctly classified samples in the validation or testing dataset
  """
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  test_loss = 0
  total = 0
  correct = 0

  for i, data in enumerate(data_loader, 0):
    # Data is a batch of featuresets and labels
    SEs, labels = data
    SEs, labels = SEs.to(device), labels.to(device)

    # Pass the data to the model
    spike_probs = model(SEs.float())

    # Calculate the loss
    regularization_loss = 0
    for name, param in model.named_parameters():
        if name == 'layer1.weight':
          regularization_loss = regularization_loss + torch.sum(abs(param))
    classify_loss = nn.BCELoss()(spike_probs.flatten(),labels.flatten().float())
    loss = classify_loss + L1_coeff * regularization_loss

    # Backpropagate the loss
    loss.backward()

    # If the spike probability is greater than 0.5, assign it a 1 for "spike"
    # If the spike probability is less than 0.5, assign it a 0 for "no spike"
    predicted = torch.round(spike_probs.data.flatten())
    correct += predicted.eq(labels.data).sum().item()
    total += labels.nelement()
    test_loss += loss.item()

  average_test_loss = test_loss/len(data_loader)
  average_test_accuracy = correct/total

  return average_test_loss, average_test_accuracy


# In[ ]:


def run_model(model, running_mode='train', train_set=None, valid_set=None, test_set=None, sampler=None, 
              batch_size=1, learning_rate=0.5, n_epochs=1, stop_thr=1e-5, L1_coeff=0, L2_coeff=0,
              dynamic_lr=False, scheduler_step=0, scheduler_gamma=0,
              shuffle=True, print_output=False):
	
  """
  This function either trains or evaluates a model. 

  training mode: the model is trained and evaluated on a validation set, if provided. 
            If no validation set is provided, the training is performed for a fixed number of epochs. 
            Otherwise, the model should be evaluted on the validation set at the end of each epoch and the training is be stopped based on one of these two conditions (whichever happens first): 
            1. The validation loss stops improving. 
            2. The maximum number of epochs is reached.

  testing mode: the trained model is evaluated on the testing set.

  Inputs: 

    model: the neural network to be trained or evaluated
    running_mode: string, 'train' or 'test'
    train_set: the training dataset 
    valid_set: the validation dataset
    test_set: the testing dataset
    batch_size: number of training samples fed to the model at each training step
    learning_rate: determines the step size in moving towards a local minimum
    n_epochs: maximum number of epoch for training the model 
    stop_thr: if the validation loss from one epoch to the next is less than this value, stop training
    L1_coeff: degree of L1 weight decay
    shuffle: determines if the shuffle property of the DataLoader is on/off
    print_output: whether the function should print training and validation losses and accuracies and the learning rate after each epoch

  Outputs when running_mode == 'train':

    model: the trained model 
    loss: dictionary with keys 'train' and 'valid'
        The value of each key is a list of loss values. Each loss value is the average of training/validation loss over one epoch.
        If the validation set is not provided just return an empty list.
    accuracies: dictionary with keys 'train' and 'valid'
        The value of each key is a list of accuracies (percentage of correctly classified samples in the dataset). Each accuracy value is the average of training/validation accuracies over one epoch. 
        If the validation set is not provided just return an empty list.

  Outputs when running_mode == 'test':

    loss: the average loss value over the testing set. 
    accuracy: percentage of correctly classified samples in the testing set. 
  """
  
  losses = {'train':[],'valid':[]}
  accuracies = {'train':[],'valid':[]}
  previous_loss = float('inf')
  average_loss = 0
  count = 0

  if running_mode == "train":
    training = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
    
    # Create an optimizer for the parameters with the assigned learning rate
    optimizer = optim.SGD(model.parameters(), learning_rate)                  

    # Want to stop training after either:
    # 1) A certain number of epochs has passed
    # 2) The model stops improving, or the loss decreases by less than a stop threshold
    # Whichever comes first
    while count < n_epochs and abs(previous_loss-average_loss) > stop_thr:
      
      # Set model to "train" model
      model.train(True)

      # Update previous loss to the loss from the last epoch
      previous_loss = average_loss

      # Train the model
      model, average_loss, accuracy = _train(model, training, optimizer, L1_coeff)
    
      # Keep track of losses and accuracies
      losses['train'].append(average_loss)
      accuracies['train'].append(accuracy)
      
      # Validate the model if a validation set is provided
      if valid_set is not None:
        
        # Set model to "eval" mode
        # Certain functions like Dropout and Batchnorm behave differently based on the model setting
        model.train(False)
        
        # Load validation data
        validation = DataLoader(valid_set, batch_size=batch_size, shuffle=shuffle)

        # Test model on validation set
        valid_loss, valid_accuracy = _test(model, validation, L1_coeff)

        losses['valid'].append(valid_loss)
        accuracies['valid'].append(valid_accuracy)

      # Print losses, accuracies, and learning rate after each epoch
      if print_output:
        print('Epoch {} completed'.format(count))
        print('Training Loss: {}. Training Accuracy: {}'.format(average_loss, accuracy))
        print('Validation Loss: {}. Validation Accuracy: {}'.format(valid_loss, valid_accuracy))
        print('Learning rate: {}'.format(optimizer.param_groups[0]['lr']))
        print('-'*20)
      
      count += 1

    return model, losses, accuracies

  elif running_mode == "test":

    # Set model to "eval" mode
    model.train(False)

    # Load the testing set
    testing = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)

    # Apply the model to the testing set
    return _test(model, testing, L1_coeff)


# 
# # **Section 5: Spatial Autocorrelation**

# In[ ]:


def get_MoransI(subunit, crop_x, crop_y):
  subunit_norm = subunit - np.mean(np.mean(subunit, axis=1))
  sum_neighbor_products = 0
  sum_weights = 0
  sum_squared_deviations = 0

  for x in range(crop_x):
    for y in range(crop_y):
      
      if x > 0:
        sum_neighbor_products = sum_neighbor_products + subunit_norm[x,y]*subunit_norm[x-1,y]
        sum_weights = sum_weights + 1
      if x < crop_x - 1:
        sum_neighbor_products = sum_neighbor_products + subunit_norm[x,y]*subunit_norm[x+1,y]
        sum_weights = sum_weights + 1
      if y > 0:
        sum_neighbor_products = sum_neighbor_products + subunit_norm[x,y]*subunit_norm[x,y-1]
        sum_weights = sum_weights + 1
      if y < crop_y - 1:
        sum_neighbor_products = sum_neighbor_products + subunit_norm[x,y]*subunit_norm[x,y+1]
        sum_weights = sum_weights + 1
      sum_squared_deviations = sum_squared_deviations + subunit_norm[x,y]**2

  MoransI = crop_x*crop_y*sum_neighbor_products/sum_weights/sum_squared_deviations
  return MoransI


# In[ ]:


def get_subunits(weights, crop_x, crop_y, threshold_MoransI=0.25):
  morans_vals = np.zeros((weights.shape[0]))
  for i in range(weights.shape[0]):
    curr_moran = get_MoransI(weights[i,:].reshape(crop_x,crop_y), crop_x, crop_y)
    morans_vals[i] = curr_moran
  subunit_ind = np.where(morans_vals > threshold_MoransI)[0]
  subunits = weights[subunit_ind,:]
  return subunits


# # **Section 6: Prediction**

# In[ ]:


def gen_STA_model(x, y, spikecounts, crop, crop_x, crop_y, filter_length, num_frames, num_trials, STAc, 
                  stim_gen=False, stim_path=None, running_seed=None, save_stim=False, save_stim_path=None):

  def nonlin(x, a1, a2, a3):
    return a1 * np.log(1 + np.exp(a2 * (x + a3)))

  if stim_gen:
    seed = running_seed
  
    # Iterate over the stimulus blocks
    for trial in range(num_trials):
        
        if stim_gen:
          # Recreate the stimulus
          stim, seed = ranb(seed, x * y * num_frames)
          stim = np.asarray(stim, dtype='int8')  # Turn boolean to integers
          stim = stim.reshape((y*x, num_frames), order='F')  # Re-shape array
          stim[stim == 0] = -1  # Change values to 1 and -1
        else:
          os.chdir(stim_path)
          filename = 'stim' + str(trial).zfill(4) + '.h5'
          with h5py.File(filename, mode='r') as f:
              stim = f['stim'][:,:]

        if stim_gen and save_stim:
          # Save the stimulus
          filename = 'stim' + str(trial).zfill(4) + '.h5'
          os.chdir(save_stim_path)
          with h5py.File(filename, mode='w') as f:
              f.create_dataset('stim', data=stim, compression=3)
  else:
    os.chdir(stim_path)
    for trial in range(num_trials):
      filename = 'stim' + str(trial).zfill(4) + '.h5'
      with h5py.File(filename, mode='r') as f:
          stim = f['stim'][:,:]
          stim = stim.reshape(x,y,num_frames)
          stim = stim[crop]
          stim = stim.reshape(crop_x*crop_y,num_frames)

  filtered_stims = []
  for trial in range(num_trials):
      for frame in range(filter_length,num_frames):
        frames = stim[:,(frame-filter_length):frame]
        filtered_stim = STAc.flatten('F').dot(frames.flatten('F'))
        filtered_stims.append(filtered_stim)

  filtered_stims = np.asarray(filtered_stims)

  num_bins = 40
  bins = mquantiles(filtered_stims, np.linspace(0, 1, num_bins+1, endpoint=False)[1:])

  spike_counts = np.zeros(shape=(num_bins))
  all_spikecounts = spikecounts[filter_length:,].T.flatten()

  bin_ind = np.digitize(filtered_stims, bins)

  for bin in range(num_bins):
      indices = np.where(bin_ind == bin)[0]
      spike_counts[bin] = all_spikecounts[indices].mean()

  spike_counts /= dt

  indices = ~np.isnan(spike_counts)
  spike_counts = spike_counts[indices]
  bins = bins[indices]

  params_STA = optimize.leastsq(lambda p, x, y: y - nonlin(x, *p), [1, 1, 1], args=(bins[1:], spike_counts[1:]))[0]
  return params_STA


# In[ ]:


def get_STA_predictions(cell_num, params_STA, STAc, crop, crop_x, crop_y, 
                        stim_frz_gen=False, stim_frz_path=None, save_stim_frz=False, save_stim_frz_path=None):

  def nonlin(x, a1, a2, a3):
    return a1 * np.log(1 + np.exp(a2 * (x + a3)))
  
  # Screen resolution
  screen_x = 800
  screen_y = 600

  # Stimulus pixel ("stixel") size, i.e. how many screen pixels are one stixel
  stixel_w = 4
  stixel_h = 4

  # Actual stimulus dimensions
  x = int(np.ceil(screen_x / stixel_w))
  y = int(np.ceil(screen_y / stixel_h))

  # Stimulus update rate
  refresh_rate = 30  # Hz
  dt = 1 / refresh_rate  # Seconds

  # The length of the STA filter
  filter_length_sec =  0.67  # 600 ms
  filter_length = int(filter_length_sec / dt)  # Number of frames
  
  # Load spikes of the frozen stimulus
  spikecounts_frz = spio.loadmat(data_path)['spk2']
  num_cells, num_frames_frz, num_trials = spikecounts_frz.shape

  # Again, only interested in the cell we used above
  spikecounts_frz = spikecounts_frz[cell_num]

  if stim_frz_gen:
    # Random seed to generate the stimulus sequence
    frozen_seed = -20000

    # Create the frozen stimulus (always the same and repeated for all trials)
    stim_frz = ranb(frozen_seed, x * y * num_frames_frz)[0]
    stim_frz = np.asarray(stim_frz, dtype='int8')  # Turn boolean to integers
    stim_frz = stim_frz.reshape((y,x, num_frames_frz), order='F')  # Re-shape array
    stim_frz = np.moveaxis(stim_frz, 0, 1) # Switch x and y
    stim_frz[stim_frz == 0] = -1  # Change values to 1 and -1

    if stim_frz_gen and save_stim_frz:
      os.chdir(save_stim_frz_path)
      filename = 'stim_frz.h5'
      with h5py.File(filename, mode='w') as f:
          f.create_dataset('stim_frz', data=stim_frz, compression=3)
  else:
    os.chdir(stim_frz_path)
    filename = 'stim_frz.h5'
    with h5py.File(filename, mode='r') as f:
      stim_frz = f['stim_frz'][:,:]
  
  preds_STA = []

  # Recreate the cropped stimulus
  stim_frz = stim_frz[crop]
  stim_frz = np.moveaxis(stim_frz, 0, 1) # Switch x and y
  stim_frz = stim_frz.reshape((crop_y*crop_x,num_frames_frz), order='F')

  for frame in range(filter_length,num_frames_frz):
    # Generate the SE for that frame
    frames = stim_frz[:,(frame-filter_length):frame]
    filtered_stim_frz = STAc.flatten('F').dot(frames.flatten('F'))
    rate = nonlin(filtered_stim_frz,*params_STA)
    preds_STA.append(rate)

  preds_STA = np.asarray(preds_STA)
  time = np.arange(num_frames_frz)*dt
  actual_fr = np.mean(spikecounts_frz,axis=1)/dt

  corr_coef = np.corrcoef(actual_fr[filter_length:],preds_STA)[0,1]
  return preds_STA, corr_coef, actual_fr, time


# In[ ]:


def gen_subunit_model(subunits, x, y, spikecounts, filter_length, num_frames, num_trials, 
                      spatial_STA, temp_STA, crop, crop_x, crop_y, stim_gen=False, stim_path=None):

  def nonlin(x, a1, a2, a3):
    return a1 * np.log(1 + np.exp(a2 * (x + a3)))
  
  # Normalize the subunits to the Euclidian norm
  num_subunits = subunits.shape[0]
  for i in range(num_subunits):
    subunits[i,:,:] /= np.linalg.norm(subunits[i,:,:].flatten())

  init_values = np.zeros(num_subunits) / num_subunits
  subunits = subunits.reshape(num_subunits, -1)

  # Check if the cell is an ON-cell or OFF-cell. If the cell is an OFF-cell, flip the sign of the cropped spatial STA.
  if spatial_STA.flatten()[np.abs(spatial_STA.flatten()).argmax()] > 0:
    weights = optimize.leastsq(lambda w: subunits.T.dot(w) - spatial_STA.flatten(), init_values)[0]
  else:
    weights = optimize.leastsq(lambda w: subunits.T.dot(w) - -1*spatial_STA.flatten(), init_values)[0]

  os.chdir(stim_path)

  filtered_stims = np.zeros(shape=((num_frames-filter_length)*num_trials,num_subunits))
  counter = 0

  for trial in trange(num_trials):
      
      filename = 'stim' + str(trial).zfill(4) + '.h5'
      with h5py.File(filename, mode='r') as f:
          stim = f['stim'][:,:]
          stim = stim.reshape(x,y,num_frames)
          stim = stim[crop]
          stim = stim.reshape(crop_x*crop_y,num_frames)

      for frame in range(filter_length,num_frames):
        frames = stim[:,(frame-filter_length):frame]
        frames = temp_STA*frames
        curr_SE = np.mean(frames,axis=1)
        filtered_stims[counter,:] = subunits.dot(curr_SE)
        counter = counter + 1

  filtered_stims[filtered_stims < 0] = 0
  filtered_stims = filtered_stims.dot(weights)

  num_bins = 40

  bins = mquantiles(filtered_stims, np.linspace(0, 1, num_bins+1, endpoint=False)[1:])

  spike_counts = np.zeros(shape=(num_bins))
  all_spikecounts = spikecounts[filter_length:,].T.flatten()

  bin_ind = np.digitize(filtered_stims, bins)

  for bin in range(num_bins):
      indices = np.where(bin_ind == bin)[0]
      spike_counts[bin] = all_spikecounts[indices].mean() / dt

  spike_counts[np.isnan(spike_counts)] = 0

  params_subunits = optimize.leastsq(lambda p, x, y: y - nonlin(x, *p), [0, 0, 0], args=(bins, spike_counts))[0]
  return params_subunits, weights


# In[ ]:


def get_subunit_predictions(cell_num, params_subunits, weights, num_frames, temp_STA, crop, crop_x, crop_y,
                            stim_frz_gen=False, stim_frz_path=None, save_stim_frz=False, save_stim_frz_path=None):

  def nonlin(x, a1, a2, a3):
    return a1 * np.log(1 + np.exp(a2 * (x + a3)))
  
  # Screen resolution
  screen_x = 800
  screen_y = 600

  # Stimulus pixel ("stixel") size, i.e. how many screen pixels are one stixel
  stixel_w = 4
  stixel_h = 4

  # Actual stimulus dimensions
  x = int(np.ceil(screen_x / stixel_w))
  y = int(np.ceil(screen_y / stixel_h))

  # Stimulus update rate
  refresh_rate = 30  # Hz
  dt = 1 / refresh_rate  # Seconds

  # The length of the STA filter
  filter_length_sec =  0.67  # 600 ms
  filter_length = int(filter_length_sec / dt)  # Number of frames
  
  # Load spikes of the frozen stimulus
  spikecounts_frz = spio.loadmat(data_path)['spk2']
  num_cells, num_frames_frz, num_trials = spikecounts_frz.shape

  # Again, only interested in the cell we used above
  spikecounts_frz = spikecounts_frz[cell_num]

  if stim_frz_gen:
    # Random seed to generate the stimulus sequence
    frozen_seed = -20000

    # Create the frozen stimulus (always the same and repeated for all trials)
    stim_frz = ranb(frozen_seed, x * y * num_frames_frz)[0]
    stim_frz = np.asarray(stim_frz, dtype='int8')  # Turn boolean to integers
    stim_frz = stim_frz.reshape((y,x, num_frames_frz), order='F')  # Re-shape array
    stim_frz = np.moveaxis(stim_frz, 0, 1) # Switch x and y
    stim_frz[stim_frz == 0] = -1  # Change values to 1 and -1

    if stim_frz_gen and save_stim_frz:
      os.chdir(save_stim_frz_path)
      filename = 'stim_frz.h5'
      with h5py.File(filename, mode='w') as f:
          f.create_dataset('stim_frz', data=stim_frz, compression=3)
  else:
    os.chdir(stim_frz_path)
    filename = 'stim_frz.h5'
    with h5py.File(filename, mode='r') as f:
      stim_frz = f['stim_frz'][:,:]
  
  # Recreate the cropped stimulus
  stim_frz = stim_frz[crop]
  stim_frz = np.moveaxis(stim_frz, 0, 1) # Switch x and y
  stim_frz = stim_frz.reshape((crop_y*crop_x,num_frames_frz), order='F')

  filtered_stim_frz = np.zeros(shape=(1,num_subunits))
  preds_sub = []

  for frame in range(filter_length,num_frames_frz):
    frames = stim_frz[:,(frame-filter_length):frame]
    frames = temp_STA*frames
    curr_SE = np.mean(frames,axis=1)
    filtered_stim_frz = subunits.dot(curr_SE)
    filtered_stim_frz = filtered_stim_frz.dot(weights)
    rate = nonlin(filtered_stim_frz,*params_subunits)
    preds_sub.append(rate)

  preds_sub = np.asarray(preds_sub)
  time = np.arange(num_frames_frz)*dt
  actual_fr = np.mean(spikecounts_frz,axis=1)/dt

  corr_coef = np.corrcoef(actual_fr[filter_length:],preds_sub)[0,1]
  return preds_sub, corr_coef, actual_fr, time


# # **Section 7: Plotting**

# In[ ]:


def plot_subunits(subunits, crop_x, crop_y):

  if save_fig:
    os.chdir(fig_path)
    plt.savefig(fig_name)
    files.download(fig_name)


# In[ ]:


def plot_predictions(preds, actual, time, filter_length, preds_label=None, save_fig=False, fig_path=None, fig_name=None):
  
  fig = plt.figure()
  plt.plot(time[filter_length:], actual[filter_length:], label="Actual")
  plt.plot(time[filter_length:], preds, label=preds_label)
  plt.legend()
  ax = plt.gca()
  x_lim = ax.set_xlim([0,10])

  if save_fig:
    os.chdir(fig_path)
    plt.savefig(fig_name)
    files.download(fig_name)


# # **Section 8: Subunit Analyses**

# In[ ]:


def jaccard(a, b):
    intersection = a.intersection(b)
    union = a.union(b)
    index = intersection.area / union.area
    return index


# In[ ]:


def get_stable_solution(cell_num):

  ellipse_sigma = 1.5  # 2d Gaussian elliptical fit

  data_path = '/content/drive/MyDrive/Colab Notebooks/cell_data_01_NC.mat'
  stim_path = '/content/drive/MyDrive/Colab Notebooks/Stim Data'
  STA_path = '/content/drive/MyDrive/Colab Notebooks/STA & Crop'

  os.chdir('/content/drive/MyDrive/Colab Notebooks/STA & Crop')
  filename = 'Cell ' + str(cell_num) + ' Uncropped Spatial STA' + '.h5'
  with h5py.File(filename, mode='r') as f:
      spatial_STA = f['spatial_STA'][:,:]

  os.chdir('/content/drive/MyDrive/Colab Notebooks/')
  from fitellipse import fitellipse
  ell = fitellipse(spatial_STA, sigma=3)
  y_max = round(max(ell[1]))
  y_min = round(min(ell[1]))
  x_max = round(max(ell[0]))
  x_min = round(min(ell[0]))
  crop = (slice(x_min, x_max, None), slice(y_min, y_max, None))

  spatial_STAc = spatial_STA[crop]
  crop_x = spatial_STAc.shape[0]
  crop_y = spatial_STAc.shape[1]

  # Get the distribution of the number of subunits across all 20 runs
  num_subunits = []
  for ver in range(0,20): 
    
    os.chdir('/content/drive/MyDrive/Colab Notebooks/Subunits/All Cells')
    if ver == 0:
      filename = "Cell " + str(cell_num) + ", 60 Nodes, 0.5 LR, 0.0001 L1 Coeff, 100 Epochs Subunits.h5"
    else:
      if cell_num == 3 and (ver == 1 or ver == 2):
        filename = "Cell " + str(cell_num) + ", 60 Nodes, 0.5 LR, 0.0001 L1 Coeff, 200 Epochs Subunits (" + str(ver) + ").h5"
      else:
        filename = "Cell " + str(cell_num) + ", 60 Nodes, 0.5 LR, 0.0001 L1 Coeff, 100 Epochs Subunits (" + str(ver) + ").h5"
    
    with h5py.File(filename, mode='r') as f:
      subunits = f['subunits'][:,:]

    num_subunits.append(subunits.shape[0])

  # Find the mode of that distribution and the number of times that mode appears
  # If there are multiple modes, select the largest one
  mode = stats.mode(num_subunits)[0][-1]
  n_layouts = stats.mode(num_subunits)[1][0]

  # Gather all the subunits from all the layouts as polygon objects
  polygons = np.zeros((n_layouts), 'O')
  vers = []
  counter = 0

  for ver in range(0,20):

    os.chdir('/content/drive/MyDrive/Colab Notebooks/Subunits/All Cells')
    if ver == 0:
      filename = "Cell " + str(cell_num) + ", 60 Nodes, 0.5 LR, 0.0001 L1 Coeff, 100 Epochs Subunits.h5"
    else:
      if cell_num == 3 and (ver == 1 or ver == 2):
        filename = "Cell " + str(cell_num) + ", 60 Nodes, 0.5 LR, 0.0001 L1 Coeff, 200 Epochs Subunits (" + str(ver) + ").h5"
      else:
        filename = "Cell " + str(cell_num) + ", 60 Nodes, 0.5 LR, 0.0001 L1 Coeff, 100 Epochs Subunits (" + str(ver) + ").h5"
    
    with h5py.File(filename, mode='r') as f:
      subunits = f['subunits'][:,:]

    if subunits.shape[0] == mode:
      ellipses = np.array([fitellipse(subunits[i,:].reshape(crop_x,crop_y), sigma=ellipse_sigma) for i in range(subunits.shape[0])])
      polygons[counter] = np.array([shpgeom.Polygon(el.T) for el in ellipses])
      counter = counter + 1
      vers.append(ver)

  """
  For every subunit in a layout:
  1. Calculate the Jacard index between that subunit and all the other subunits in all other layouts

  curr_distances: Number of Subunits x Number of Subunits x Number of Runs matrix. For a given Run X,
  this matrix represents the distances from the subunits in Run X (rows) to all subunits (columns) 
  in all other runs (3rd dimension). 
  """

  distances_for_all_layouts = []

  for curr_layout in range(n_layouts):
    curr_distances = np.zeros(shape=(mode,mode,n_layouts))
    # Distances from subunits in the same layout are excluded
    curr_distances[:,:,curr_layout] = float("nan")
    # Take each subunit
    curr_subunits = polygons[curr_layout]
    # And calculate the Jaccard index between that subunit and all the other subunits in all other layouts
    for curr_subunit_ind in range(mode):
      curr_subunit = curr_subunits[curr_subunit_ind]
      for other_layout in range(n_layouts):
        if other_layout == curr_layout:
          continue
        else:
          other_subunits = polygons[other_layout]
          for other_subunit_ind in range(mode):
            other_subunit = other_subunits[other_subunit_ind]
            curr_distances[curr_subunit_ind,other_subunit_ind,other_layout] = jaccard(curr_subunit, other_subunit)
    distances_for_all_layouts.append(curr_distances)

  """
  2. Assess similarity between layouts

  """

  run_similarity = np.zeros(shape=(n_layouts,n_layouts))
  np.fill_diagonal(run_similarity,float('nan'))

  for curr_layout in range(n_layouts):
    distances_for_curr_layout = distances_for_all_layouts[curr_layout]
    for other_layout in range(n_layouts):
      if curr_layout == other_layout:
        continue
      else:
        curr_distances = distances_for_curr_layout[:,:,other_layout]
        smallest_distances = []
        for subunit in range(mode):
          curr_smallest_dist_ind = np.unravel_index(np.argmax(curr_distances),shape=curr_distances.shape)
          row_ind = curr_smallest_dist_ind[0]
          col_ind = curr_smallest_dist_ind[1]

          curr_smallest_dist = curr_distances[curr_smallest_dist_ind]
          smallest_distances.append(curr_smallest_dist)

          curr_distances[row_ind,:].fill(-1)
          curr_distances[:,col_ind].fill(-1)
        run_similarity[curr_layout,other_layout] = np.mean(np.asarray(smallest_distances))

  avg_similarity = np.nanmean(run_similarity,axis=0)
  stable_ver = vers[np.argmax(avg_similarity)]

  return stable_ver


# In[ ]:


def subunit_sizes(subunits, crop_x, crop_y):
  
  diameters = []
  num_subunits = subunits.shape[0]

  for subunit_ind in range(num_subunits):
    curr_subunit = subunits[subunit_ind,:].reshape(crop_x,crop_y)
    if np.mean(curr_subunit) < 0:
      print("skip")
      continue
    ell_properties = fitellipse(curr_subunit, sigma=1.5, raw=True)
    r_major = ell_properties[2]/2
    r_minor = ell_properties[3]/2
    area = math.pi*r_major*r_minor
    effective_diameter = 30*2*math.sqrt(area/math.pi)
    diameters.append(effective_diameter)
  
  return diameters


# In[ ]:


def rf_size(rf):

  ell_properties = fitellipse(rf, sigma=1.5, raw=True)
  r_major = ell_properties[2]/2
  r_minor = ell_properties[3]/2
  area = math.pi*r_major*r_minor
  effective_diameter = 30*2*math.sqrt(area/math.pi)

  return effective_diameter


# # **Playing with Cophenetic Correlation**

# In[ ]:


cell_num = 0
node_num = 60
L1_reg = 1e-4
learn_rate = 0.5
num_epochs = 100

os.chdir('/content/drive/MyDrive/Colab Notebooks/Lab Rotation/STA & Crop')
filename = 'Cell ' + str(cell_num) + ' Uncropped Spatial STA' + '.h5'
with h5py.File(filename, mode='r') as f:
    spatial_STA = f['spatial_STA'][:,:]

os.chdir('/content/drive/MyDrive/Colab Notebooks/Lab Rotation/Subunits')
filename = 'Cell ' + str(cell_num) + ', ' + str(node_num) + ' Nodes, ' + str(learn_rate) + ' LR, ' + str(L1_reg) + ' L1 Coeff, ' + str(num_epochs) + ' Epochs Subunits (2).h5'
with h5py.File(filename, mode='r') as f:
  subunits = f['subunits'][:,:]
num_subunits = subunits.shape[0]


# In[ ]:


from scipy.cluster.hierarchy import linkage, cophenet

cophenet(linkage(subunitsall))


# In[ ]:


subunits1 = subunits


# In[ ]:


subunits2 = subunits


# In[ ]:


subunits1.shape


# In[ ]:


subunits2.shape


# In[ ]:


subunitsall = np.zeros(shape=(17,550))
subunitsall[:9,:] = subunits1
subunitsall[9:,:] = subunits2


# In[ ]:




