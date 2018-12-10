import torch.nn as nn

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    # RNNModel(10000, 400, 1150, 3)
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5,mode="LSTM"):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        
        self.rnn_type = mode
        if(mode == "LSTM"):
          self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        else:
          self.rnn = nn.RNN(ninp, nhid, nlayers, dropout=dropout)  
        
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        
        # output.size() = 70, 20, 1150 = seq_len, batch_size, hidden_size
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        
        # decoded.size() = 1400, 10000 = seq_len*batch_size, ntokens
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        
        # decoded.view(...).size() = 70, 20,10000 = seq_len, batch_size, ntokens
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)