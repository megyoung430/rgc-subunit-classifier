import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.dataset import Dataset

class Spike_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return (x, y)

    def __len__(self):
        return self.x.shape[0]

class Spike_Classifier(nn.Module):

    """
    This is the class that creates a neural network for determining presence/absence of a spike from a "spike"-triggered ensemble

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size node_num nodes
    - Output layer: a linear layer with one node, representing the spike probability

    Activation functions: rectified linear activation function for the hidden layer and sigmoidal activation function for the output layer.
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

def _train(model, data_loader, optimizer, L1_coeff):

    """
    This function implements one epoch of training a neural network.

    Inputs:
        - model: the neural network to be trained
        - data_loader: for loading the network input and labels from the training dataset
        - optimizer: the optimiztion method, e.g., SGD
        - L1_coeff: L1 weight decay factor 

    Outputs:
        - model: the trained model
        - train_loss: average loss value on the entire training dataset
        - train_accuracy: average accuracy on the entire training dataset
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

def _test(model, data_loader, L1_coeff):
    
    """
    This function evaluates the trained network on a validation set or a testing set. 
    
    Inputs:
        - model: trained neural network
        - data_loader: for loading the network input and targets from the validation or testing dataset
    
    Output:
        - test_loss: average loss value on the entire validation or testing dataset 
        - test_accuracy: percentage of correctly classified samples in the validation or testing dataset
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
        
        # If the spike probability is greater than 0.5, assign it a 1 for "spike"
        # If the spike probability is less than 0.5, assign it a 0 for "no spike"
        predicted = torch.round(spike_probs.data.flatten())
        correct += predicted.eq(labels.data).sum().item()
        total += labels.nelement()
        test_loss += loss.item()
    
    average_test_loss = test_loss/len(data_loader)
    average_test_accuracy = correct/total
    return average_test_loss, average_test_accuracy

def run_model(model, running_mode='train', train_set=None, valid_set=None, test_set=None, sampler=None, batch_size=1, learning_rate=0.5, n_epochs=1, stop_thr=1e-5, L1_coeff=0, L2_coeff=0, shuffle=True, print_output=False):
    
    """
    This function either trains or evaluates a model. 
    
    training mode: the model is trained and evaluated on a validation set, if provided.
        If no validation set is provided, the training is performed for a fixed number of epochs. 
        Otherwise, the model should be evaluted on the validation set at the end of each epoch and the training is be stopped based on one of these two conditions (whichever happens first): 
        1. The validation loss stops improving. 
        2. The maximum number of epochs is reached.
    testing mode: the trained model is evaluated on the testing set.
    
    Inputs:
        - model: the neural network to be trained or evaluated
        - running_mode: string, 'train' or 'test'
        - train_set: the training dataset 
        - valid_set: the validation dataset
        - test_set: the testing dataset
        - batch_size: number of training samples fed to the model at each training step
        - learning_rate: determines the step size in moving towards a local minimum
        - n_epochs: maximum number of epoch for training the model 
        - stop_thr: if the validation loss from one epoch to the next is less than this value, stop training
        - L1_coeff: degree of L1 weight decay
        - shuffle: determines if the shuffle property of the DataLoader is on/off
        - print_output: whether the function should print training and validation losses and accuracies and the learning rate after each epoch
    
    Outputs when running_mode == 'train':
        - model: the trained model 
        - loss: dictionary with keys 'train' and 'valid'
            The value of each key is a list of loss values. Each loss value is the average of training/validation loss over one epoch.
            If the validation set is not provided just return an empty list.
        - accuracies: dictionary with keys 'train' and 'valid'
            The value of each key is a list of accuracies (percentage of correctly classified samples in the dataset). Each accuracy value is the average of training/validation accuracies over one epoch. 
            If the validation set is not provided just return an empty list.
    
    Outputs when running_mode == 'test':
        - loss: the average loss value over the testing set. 
        - accuracy: percentage of correctly classified samples in the testing set. 
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