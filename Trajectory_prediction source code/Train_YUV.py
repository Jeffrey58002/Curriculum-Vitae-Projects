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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CSV data
csv_file_path = 'C:/Users/home/Desktop/Camera_recognition/predictions.csv'
data = pd.read_csv(csv_file_path)
data = data.drop(['Image Name', 'Prediction', 'Confidence'], axis=1)

SOS_token = 0  # Start of sequence. Assuming 0 is not a YUV value in your normalized data
EOS_token = 1  # End of sequence

currentTime = time.strftime("%Y%m%d-%H%M%S")
#data.head() # Display the first few rows of the dataframe


# Convert timestamps to datetime objects
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%Y%m%d-%H%M%S')
# Convert to seconds since the first timestamp
data['Timestamp'] = (data['Timestamp'] - data['Timestamp'].iloc[0]).dt.total_seconds()


# Initialize a scaler
scaler = MinMaxScaler()
data[['Y Value', 'U Value', 'V Value']] = scaler.fit_transform(data[['Y Value', 'U Value', 'V Value']])
data = data['Y Value', 'U Value', 'V Value']
def create_sequences(data, input_steps=5, output_steps=1):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i:(i + input_steps), :])
        y.append(data[(i + input_steps):(i + input_steps + output_steps), :])
    return np.array(X), np.array(y)

# numpy array with the YUV values
X, y = create_sequences(data.values, input_steps=5, output_steps=1)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert  X_train, y_train, X_test, and y_test to tensors and create DataLoader for them
train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

batch_size = 64
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

MAX_LENGTH = max(len(seq) for seq in X_train)  

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size , fc_size=128):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(input_size, fc_size)
        self.lstm = nn.LSTM(input_size, hidden_size)  # Using LSTM instead of GRU

    def forward(self, input_seq, hidden):
        output, (hidden, cell) = self.lstm(input_seq.view(len(input_seq), 1, -1), hidden)
        return output, (hidden, cell)

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device),
                torch.zeros(1, 1, self.hidden_size, device=device))  # LSTM requires a tuple (hidden state, cell state)
    
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, fc_size=128):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(input_size, fc_size)
        self.lstm = nn.LSTM(fc_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden):
        output = F.relu(input_seq)
        output, (hidden, cell) = self.lstm(output.view(1, 1, -1), hidden)
        output = self.out(output.squeeze(0))
        return output, (hidden, cell)
#Train
def train(encoder, decoder, data_loader, encoder_optimizer, decoder_optimizer, criterion, max_length, teacher_forcing_ratio):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    # Encoding
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)  # Adjust for LSTM

    # Decoding
    decoder_input = torch.tensor([[SOS_token]], device=device)  # Adjust for batch dimension if necessary

    # Adjust decoder_hidden to handle LSTM tuple
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))  # Ensure dimensions match
        if decoder_input.item() == EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def validate(input_tensor, target_tensor, encoder, decoder, criterion, max_length=MAX_LENGTH):
    with torch.no_grad():
        encoder_hidden = encoder.initHidden().to(device)

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        loss = 0

        # Encoding
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)  # Adjust for LSTM

        # Decoding
        decoder_input = torch.tensor([[SOS_token]], device=device)  # Start with the start-of-sequence token

        decoder_hidden = encoder_hidden

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))

            if decoder_input.item() == EOS_token:
                break

        return loss.item() / target_length


def train_and_validate(encoder, decoder, train_dataloader, test_dataloader, encoder_optimizer, decoder_optimizer, criterion, epochs, print_every):
    """
    Handles the training and validation of the model.
    Returns the training and validation loss history.
    """
    train_losses = []       # Document the results
    validation_losses = []

    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        train_loss = 0

        for input_tensor, target_tensor in train_dataloader:
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            train_loss += loss

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        
        # Validation phase
        validation_loss = 0
        encoder.eval()
        decoder.eval()
        for input_tensor, target_tensor in test_dataloader:
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
            loss = validate(input_tensor, target_tensor, encoder, decoder, criterion)
            validation_loss += loss

        validation_loss /= len(test_dataloader)
        validation_losses.append(validation_loss)

        if epoch % print_every == 0:
            print(f'Epoch {epoch}, Training loss: {train_loss}, Validation loss: {validation_loss}')
    
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

    epochs = 100
    print_every = 100

# Initialize  model, optimizers, and other components.
    encoder = Encoder(input_size, hidden_size).to(device)
    decoder = Decoder(input_size, hidden_size, output_size).to(device)  # Ensure the input_size matches the number of features


    criterion = nn.MSELoss()
    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())
    train_losses, validation_losses = train_and_validate(encoder, decoder, train_dataloader, test_dataloader, encoder_optimizer, decoder_optimizer, criterion, epochs, print_every)
    
    # Plot the losses
    plot_losses(train_losses, validation_losses)

if __name__ == '__main__':
    main()
