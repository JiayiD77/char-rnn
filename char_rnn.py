import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# Models
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.RNN(input_size=input_size, 
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout)
        self.decoder = nn.Linear(in_features=hidden_size,
                                 out_features=output_size)
        
    def forward(self, x, hidden_state):
        embeddings = self.embedding(x)
        output, hidden_state = self.rnn(embeddings, hidden_state)
        output = self.decoder(output)
        return output, hidden_state

class RNNC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.RNNCell(input_size=input_size, 
                          hidden_size=hidden_size)
        self.decoder = nn.Linear(in_features=hidden_size,
                                 out_features=output_size)
        
    def forward(self, x, hidden_state):
        embeddings = self.embedding(x)
        hidden_state = self.rnn(embeddings, hidden_state)
        output = self.decoder(hidden_state)
        return output, hidden_state
    
# Data preparation
text = open("input.txt", 'r').read()
chars = sorted(list(set(text)))
data_size = len(text)
vocab_size = len(chars)
print("-------------------------------------------------------------------")
print("Preparing training data......")
print(f"Total number of characters in the text is: {data_size}")
print(f"Total number of unique characters is: {vocab_size}")
print(f"The unique characters are: {chars}")
print("-------------------------------------------------------------------")

chars_to_ids = {c:i for i, c in enumerate(chars)}
ids_to_chars = {i:c for i, c in enumerate(chars)}

encode = lambda s:[chars_to_ids[c] for c in s]
decode = lambda l:[ids_to_chars[i] for i in l]

data = torch.tensor(encode(text), dtype=torch.long)
split = int(0.9*len(data))
train_data = data[:split]
test_data = data[split:]
# train_data = torch.unsqueeze(data[:split], dim=1)
# test_data = torch.unsqueeze(data[split:], dim=1)

# Parameters
hidden_size = 256
seq_len = 128
num_layers = 3
dropout = 0
lr = 1e-4
epochs = 150
test_seq_len = 5000

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Available device: {device}")
print("-------------------------------------------------------------------")

# training
rnn = RNN(input_size=vocab_size,
          hidden_size=hidden_size,
          output_size=vocab_size,
          num_layers=num_layers,
          dropout=dropout).to(device)

rnnc = RNNC(input_size=vocab_size,
          hidden_size=hidden_size,
          output_size=vocab_size).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=rnn.parameters(), lr=lr)

train_data = train_data.to(device)
test_data = test_data.to(device)

print("Training.....")
for epoch in range(epochs):
    start_idx = torch.randint(0, 200, (1,))
    n = 0    
    total_loss = 0
    hidden_state = None
    
    while True:
        input_seq = train_data[start_idx : start_idx+seq_len]
        target_seq = train_data[start_idx+1 : start_idx+seq_len+1]
        
        output_seq, hidden_state = rnn(input_seq, hidden_state)
        hidden_state = hidden_state.data
        
        loss = loss_fn(torch.squeeze(output_seq), torch.squeeze(target_seq))
        total_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        start_idx += seq_len
        n += 1
        
        if start_idx + seq_len + 1 > len(train_data) - 1:
            break
        
    if epoch % 5 == 0:    
        print(f"Epoch: {epoch}")
        print(f"Loss: {total_loss/n:.5f}") 

print("-------------------------------------------------------------------")
print("Generating Texts.....")

char_num = 0
hidden_state_gen = None
start_idx_gen = torch.randint(0, 200, (1,))
input_seq_gen = test_data[start_idx_gen : start_idx_gen+seq_len]
while True:
    
    input = input_seq_gen[-seq_len:]
    output, hidden_state_gen = rnn(input, hidden_state_gen)
    output = F.softmax(torch.squeeze(output[-1]), dim=0)
    dist = Categorical(output)
    index = dist.sample()
    print(ids_to_chars[index.item()], end='')
    
    input_seq_gen = torch.cat((input_seq_gen, index.unsqueeze(dim=0)), dim=0)
    char_num += 1
    
    if char_num > test_seq_len:
        break
print('\n')
print("-------------------------------------------------------------------")