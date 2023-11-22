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