import torch
import torch.nn as nn
import os

# Get current working directory
cwd = os.getcwd()

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, recType):
        super(RNN, self).__init__()

        # Store the design-related hyperparameters
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.recType = recType

        # Depending on the type of RNN, create layers of GRU or LSTM units
        if self.recType=='GRU':
          self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif self.recType=='LSTM':
          self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Initialize the final linear transformation layer to "translate" the
        # hidden size of the RNN to the desired weight vector
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden states (h0), and cell states (c0) if LSTM is selected.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        if self.recType =='LSTM':
          c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Depending on the RNN type, use the relative processing units
        if self.recType=='GRU':
          out,_ = self.gru(x,h0)
        elif self.recType=='LSTM':
          out,_ = self.lstm(x,(h0,c0))

        # Keep the output of the final timestep only
        out = out[:, -1, :]

        # Pass the RNN output through the linear layer
        out = self.fc(out)

        return out
        
# Define and initialize the RNN
input_size = 1  # Number of input features
hidden_size = 256  # Number of features in the hidden state
num_layers = 2  # Number of stacked RNN layers
output_size = 32  # Number of output features
recType = 'LSTM'  # Choose between 'GRU' and 'LSTM'
lstm_model = RNN(input_size, hidden_size, num_layers, output_size, recType)

## REPLACE PATH ACCORDING TO THE LOCATION OF YOUR SAVED MODEL ##
MODEL_PATH = cwd + "/lstm_model.pt"
# Import the pre-trained model and set it to evaluation mode
lstm_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
lstm_model.eval()

# Convert MATLAB input x to tensor
input = torch.tensor(x)
# Normalize input from [30,150] to [0,1]
input = (input-30)/120
# Reshape input to have the shape
# BATCH_SIZE x SEQUENCE LENGTH x INPUT SIZE
input = input.reshape(1,3,1)
# Get LSTM output
out_lstm = lstm_model(input).flatten()
# De-normalize output from [0, 1] to [-1, 1]
out_lstm = (2*out_lstm)-1
# Send it back to MATLAB as a list
output = out_lstm.tolist()
