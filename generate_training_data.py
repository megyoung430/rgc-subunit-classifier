import h5py
import numpy as np
from fitellipse import fitellipse

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