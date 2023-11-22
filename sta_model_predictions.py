def gen_STA_model(x, y, spikecounts, crop, crop_x, crop_y, filter_length, num_frames, num_trials, STAc, stim_gen=False, stim_path=None, running_seed=None, save_stim=False, save_stim_path=None):
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