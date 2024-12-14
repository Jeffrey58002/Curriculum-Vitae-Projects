import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import random
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CSV data
csv_file_path = 'C:/Users/home/Desktop/Camera_recognition/Shortened_prediction.csv'
data = pd.read_csv(csv_file_path)
data = data.drop(['Image Name', 'Prediction', 'Confidence'], axis=1)

SOS_token = 0  # Start of sequence. Assuming 0 is not a YUV value in your normalized data
EOS_token = 1  # End of sequence

currentTime = time.strftime("%Y%m%d-%H%M%S")
output_steps =1

# Initialize a scaler
scaler = MinMaxScaler()
data[['Y Value', 'U Value', 'V Value']] = scaler.fit_transform(data[['Y Value', 'U Value', 'V Value']])
data = data[['Y Value', 'U Value', 'V Value']]  # Keep only YUV columns
#Make sure you adjust this function if you've changed the number of features or how they're arranged in your data.
def create_sequences(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i:(i + input_steps)])
        y.append(data[(i + input_steps):(i + input_steps + output_steps)])
    return np.array(X), np.array(y)

# Assuming `data` is a numpy array with the YUV values
X, y = create_sequences(data.values, input_steps=5, output_steps=1)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert your X_train, y_train, X_test, and y_test to tensors and create DataLoader for them
train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

batch_size = 64
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

MAX_LENGTH = max(len(seq) for seq in X_train)  # Replace with actual calculation based on your data
target_length = output_steps

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, fc_size=128, num_layers=2, dropout=0.5):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(input_size, fc_size)
        #LSTM with biderictional and dropout
        self.lstm = nn.LSTM(fc_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, input_seq):
        fc_out = F.relu(self.fc(input_seq))
        outputs, (hidden, cell) = self.lstm(fc_out)
        return outputs, (hidden, cell)
    def initHidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size, device=device),
                torch.zeros(1, batch_size, self.hidden_size, device=device))
    
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, fc_size=128):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(input_size, fc_size)
        self.lstm = nn.LSTM(fc_size, hidden_size, batch_first=True)  # Change to batch_first
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden):
        fc_out = F.relu(self.fc(input_seq))
        output, (hidden, cell) = self.lstm(fc_out, hidden)
        prediction = self.fc_out(output)
        return prediction, (hidden, cell)
    def initHidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size, device=device),
                torch.zeros(1, batch_size, self.hidden_size, device=device))

#Train
def train(encoder, decoder, data_loader, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio):
    encoder.train()
    decoder.train()

    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Training')
    for input_tensor, target_tensor in progress_bar:
        for input_tensor, target_tensor in data_loader:
            batch_size = input_tensor.size(0)
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            
            # Reset gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            # Initialize encoder state
            encoder_hidden = encoder.initHidden(batch_size)
            
            # Encoder step
            encoder_output, encoder_hidden = encoder(input_tensor)
            
            # Decoder step
            decoder_input = input_tensor[:, -1, :]  # Start with the last point of the input sequence
            decoder_input = decoder_input.unsqueeze(1)  # To go from [batch_size, features] to [batch_size, 1, features]
            decoder_hidden = encoder_hidden
            
            loss = 0
            
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                
                # Squeeze the output to remove the sequence length dimension
                decoder_output = decoder_output.squeeze(1)
                
                loss += criterion(decoder_output, target_tensor[:, di])
                
                # Determine whether to use teacher forcing
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                
                if use_teacher_forcing:
                    # Teacher forcing: Next input comes from ground truth
                    decoder_input = target_tensor[:, di].unsqueeze(1)
                else:
                    # No teacher forcing: Next input comes from predictions
                    decoder_input = decoder_output.unsqueeze(1)  # Adjust the shape of decoder_output if necessary

            loss.backward()  # Backpropagation
            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()
            # Update the progress bar with the current loss
            #progress_bar.set_postfix({'Loss': loss.item()})

    return total_loss / len(data_loader)


def validate(encoder, decoder, data_loader, criterion, max_length):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        total_loss = 0
        for input_tensor, target_tensor in data_loader:
            batch_size = input_tensor.size(0)
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            # Initialize encoder state
            encoder_hidden = encoder.initHidden(batch_size)
            
            # Encoding
            encoder_output, encoder_hidden = encoder(input_tensor)

            # Decoding
            decoder_input = input_tensor[:, -1, :].unsqueeze(1)  # Reshape for LSTM
            decoder_hidden = encoder_hidden

            loss = 0
            
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                 # Squeeze the output to remove the sequence length dimension
                decoder_output = decoder_output.squeeze(1)
                loss += criterion(decoder_output, target_tensor[:, di])
                decoder_input = decoder_output.unsqueeze(1)  # Use model's own prediction for next input

            total_loss += loss.item()

        return total_loss / len(data_loader)



def train_and_validate(encoder, decoder, train_dataloader, test_dataloader, encoder_optimizer, decoder_optimizer, criterion, epochs, print_every, teacher_forcing_ratio):
    """
    Handles the training and validation of the model.
    Returns the training and validation loss history.
    """
    train_losses = []       # Document the results
    validation_losses = []

    for epoch in range(1, epochs + 1):
        train_loss = train(encoder, decoder, train_dataloader, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio)
        train_losses.append(train_loss)

        validation_loss = validate(encoder, decoder, test_dataloader, criterion, MAX_LENGTH)
        validation_losses.append(validation_loss)

        if epoch % print_every == 0:
            print(f'Epoch {epoch}, Training loss: {train_loss:.4f}, Validation loss: {validation_loss:.4f}')
    
    return train_losses, validation_losses
def plot_losses(train_losses, validation_losses, folder_name="training_results {}".format(currentTime)):
    """
    Plots the training and validation losses and saves the plot in a specified folder.
    """
    
    # Check if folder exists, and create it if it doesn't
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Plot the data
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    
    # Save the plot in the specified folder
    plot_filename = os.path.join(folder_name, 'loss_plot.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free memory


def main():
    input_size = 3  # YUV channels
    hidden_size = 128
    output_size = 3  # Predicting YUV values
    epochs = 4
    print_every = 100
    teacher_ratio = 0.5  # Adjust it for your need

# Initialize your model, optimizers, and other components here...
    encoder = Encoder(input_size, hidden_size).to(device)
    decoder = Decoder(input_size , hidden_size, output_size).to(device)

    criterion = nn.MSELoss()
    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())
    train_losses, validation_losses = train_and_validate(
                encoder, decoder, train_dataloader, test_dataloader, 
                encoder_optimizer, decoder_optimizer, criterion, 
                epochs, print_every, teacher_ratio
    )
    # Plot the losses
    plot_losses(train_losses, validation_losses)

if __name__ == '__main__':
    main()