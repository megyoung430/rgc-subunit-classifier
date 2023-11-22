import numpy as np
import math
from fitellipse import fitellipse

def get_MoransI(subunit, crop_x, crop_y):

	"""
	This function calculates the spatial autocorrelation as the Moran's I for any putative subunit, i.e., set of weights 
	to a particular hidden node in the network.

	Inputs: 
		- subunit ():
		- crop_x (int): the size of the cropped STA / the input image in the x-dimension, in number of pixels
		- crop_y (int): the size of the cropped STA / the input image in the y-dimension, in number of pixels
	
	Output:
		-MoransI (float): the MoransI/spatial autocorrelation value for that particular putative subunit
	"""
	
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

def get_subunits(weights, crop_x, crop_y, threshold_MoransI=0.25):

	"""
	This function retrieves the identified subunits from the weights of the nodes in the hidden layer of the network.

	Inputs:
		- weights (num_hidden nodes x num_pixels array): the weights from the input layer to the hidden layer in the neural network
		- crop_x (int): the size of the cropped STA / the input image in the x-dimension, in number of pixels
		- crop_y (int): the size of the cropped STA / the input image in the y-dimension, in number of pixels
		- threshold_MoransI (float): the threshold for the Moran's I value, above which the weights of that hidden node will be
		considered a subunit
	
	Outputs:
		- subunits (num_subunits x num_pixels array): the subunits identified from the weights of the network
	"""

	morans_vals = np.zeros((weights.shape[0]))
	for i in range(weights.shape[0]):
		curr_moran = get_MoransI(weights[i,:].reshape(crop_x,crop_y), crop_x, crop_y)
		morans_vals[i] = curr_moran
	subunit_ind = np.where(morans_vals > threshold_MoransI)[0]
	subunits = weights[subunit_ind,:]
	return subunits

def jaccard(a, b):

	"""
	This function calculates the Jaccard index of two shapes.

	Inputs:
		- a (shpgeom Polygon object): the first shape of interest
		- b (shpgoem Polygon object): the second shape of interest

	Outputs:
		- index (float): the Jaccard index of the two shapes.
	"""
	
	intersection = a.intersection(b)
	union = a.union(b)
	index = intersection.area / union.area
	return index

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
	total_vers = 20
	for ver in range(0,total_vers): 
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

	for ver in range(0,total_vers):
		
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

def rf_size(rf):

	"""
	This function calculates the size of the receptive field of the cell in terms of its effective diameter.

	Inputs:
		- rf
	
	Outputs:
		- effective_diameter (float): the effective diameter of the receptive field for the given cell.
	"""
	
	ell_properties = fitellipse(rf, sigma=1.5, raw=True)
	r_major = ell_properties[2]/2
	r_minor = ell_properties[3]/2
	area = math.pi*r_major*r_minor
	effective_diameter = 30*2*math.sqrt(area/math.pi)
	return effective_diameter