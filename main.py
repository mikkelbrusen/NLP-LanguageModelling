#%%
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from collections import Counter

import data
import model
import utils

args_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
args_train_batch_size = 20 # batch size
args_bptt = 35 # sequence length
args_embed_size = 650 # emsize
args_hidden_size = 650 # nhid
args_num_layers = 2 # nlayers
args_num_epochs = 40
args_learning_rate = 20
args_clip = 0.25
args_log_interval = 2
args_save = "model.pt" # "/content/gdrive/My Drive/NLP/save/model.pt"
args_data = "./data/penn" # /content/gdrive/My Drive/NLP/data/penn/
args_rnn_type = "LSTM" # LSTM or RNN
args_seed = 1111

torch.manual_seed(args_seed)

# Load "Penn Treebank" dataset
corpus = data.Corpus(args_data)

eval_batch_size = 10
test_batch_size = 10
train_data = utils.batchify(corpus.train, args_train_batch_size,args_cuda)
val_data = utils.batchify(corpus.valid, eval_batch_size,args_cuda)
test_data = utils.batchify(corpus.test, test_batch_size,args_cuda)

ntokens = len(corpus.dictionary)

# RNNModel(10000, 400, 1150, 3)
model = model.RNNModel(ntokens, args_embed_size, args_hidden_size, args_num_layers,mode=args_rnn_type).to(device)
criterion = nn.CrossEntropyLoss()

#%%
# Define the trainging and evaluation functions

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(test_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args_bptt):
            data, targets = utils.get_batch(data_source, i,args_bptt)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = utils.repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args_train_batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args_bptt)):
        data, targets = utils.get_batch(train_data, i,args_bptt)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = utils.repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args_clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args_log_interval == 0 and batch > 0:
            cur_loss = total_loss / args_log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args_bptt, lr,
                elapsed * 1000 / args_log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


#%%
# Then do the actual training
# Loop over epochs.
lr = args_learning_rate
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args_num_epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args_save, 'wb') as f:
                torch.save(model, f)
            
            # save model to variable
            #best_model = model
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


#%%
# Finally, open the best saved model run it on the test data
# Load the best saved model.
with open(args_save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Take model from memory
#model = best_model

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)