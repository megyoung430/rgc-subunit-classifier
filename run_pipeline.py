import numpy as np

def subunit_model(cell_num, ver, data_path, stim_path, stim_save_path=None, STA_path=None, subunits_save_path=None, sigma=3, node_num=60, batch_size=25, learn_rate=0.5, num_epochs=100, stop_thr=0, L1_reg=0):
	
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
	traindataset = Spike_Dataset(x_train,y_train)
	testdataset = Spike_Dataset(x_test,y_test)
	validdataset = Spike_Dataset(x_valid,y_valid)

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
																				L1_coeff=L1_reg,
																				shuffle=False, print_output=True)
	
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