# This is the RNN class
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.config = config
        if config['use_lstm']:
            self.lstm = True
            self.rnn = nn.LSTM(64, 
                          config['hidden_size'], 
                          batch_first=True,
                         )
        else:
            self.lstm = False
            self.rnn = nn.RNN(64, 
                          config['hidden_size'], 
                          batch_first=True,
                         )

        self.conv1 = nn.Conv2d(22, 64, (1, 5), stride=(1, 3))
        self.conv2 = nn.Conv2d(64, 64, (1, 5), stride=(1, 3))

        # Use attention
        self.mha = MultiheadAttention(config['hidden_size'], config['nhead'], batch_first=True)
        self.dropout = torch.nn.Dropout(config['dropout'])
        self.output_layer = nn.Linear(config['hidden_size'], 4)
        
    def forward(self, inp):  
        # Convolution  
        batch_size = inp.shape[0]
        seq_len = inp.shape[2]
        inp = inp.unsqueeze(2)
        
        # Performing convolution
        inp = self.conv1(inp)
        inp = self.dropout(inp)
        inp = self.conv2(inp)
        inp = self.dropout(inp)
        inp = inp.reshape(batch_size, 64, -1)
        inp = inp.transpose(1, 2)
        # Inp shape now is now (batch size, seq_len, 64)
        
        # RNN
        if self.lstm:
            out, (_, c_n) = self.rnn(inp)
        else:
            out, c_n = self.rnn(inp)
        c_n = c_n.reshape(batch_size, 1, -1)
        
        # Use attention here
        if self.config["use_attention"]:
            output = self.output_layer(self.mha(c_n, out, out)[0].reshape(batch_size, self.config['hidden_size']))
        else:
            c_n = c_n.squeeze(1)
            output = self.output_layer(c_n)
         
        return output
    
