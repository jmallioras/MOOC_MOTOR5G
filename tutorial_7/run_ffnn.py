import torch
import torch.nn as nn
import os

# Get current working directory
cwd = os.getcwd()

class FeedForwardNet(nn.Module):
    def __init__(self, layers):
        super(FeedForwardNet, self).__init__()
        # Initialize the model
        self.hidden = nn.ModuleList() # Hidden layer list

        # Fill the list depending on the layer size dictated by "layers"
        for input_size, output_size in zip(layers,layers[1:]):
          linear = nn.Linear(input_size, output_size)
          self.hidden.append(linear)

    def forward(self,activation):

        # Get the number of hidden layers
        L=len(self.hidden)

        # Use the tanh activation function after each hidden layer
        # Except the final layer, where we use the sigmoid
        for (l,linear_transform) in zip(range(L),self.hidden):
          if l<L-1:
            activation = (torch.tanh(linear_transform(activation))) # tanh
          else:
            activation = (torch.sigmoid(linear_transform(activation))) # sigmoid

        return activation

# Define and initialize the RNN
layers = [3,256,512,32]
ffnn_model = FeedForwardNet(layers)


## REPLACE PATH ACCORDING TO THE LOCATION OF YOUR SAVED MODEL ##
MODEL_PATH = cwd + "/ffnn_model.pt"
# Import the pre-trained model and set it to evaluation mode
ffnn_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
ffnn_model.eval()

# Convert MATLAB input x to tensor
input = torch.tensor(x)
# Normalize input from [30,150] to [0,1]
input = (input-30)/120
# Get FFNN output
out_ffnn = ffnn_model(input).flatten()
# De-normalize output from [0, 1] to [-1, 1]
out_ffnn = (2*out_ffnn)-1
# Send it back to MATLAB as a list
output = out_ffnn.tolist()
