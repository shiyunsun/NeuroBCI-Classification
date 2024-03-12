import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

# Acknowledgement: Part of the code is modified 
# from my past assignments from CS247 Advanced Data Mining

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        # The default positional encoding method
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        pow_term = torch.pow(10000, (torch.arange(0, d_model, 2)/d_model))
        div_term = position/pow_term
        pe[:, 0::2] = torch.sin(div_term)
        pe[:, 1::2] = torch.cos(div_term)
        pe = pe.view(1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len, :]
        return x
    
class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        # Attention
        self.mha = MultiheadAttention(d_model, num_heads, batch_first=True) 
        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        out = self.layernorm1(self.dropout(self.mha(x, enc_output, enc_output)[0]))
        
        return out

class Transformer(torch.nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.inp_encoding = nn.Linear(64, config['d_model'])
        self.pos_encoding = PositionalEncoding(config['d_model'], 2500)
        self.class_encoding = nn.Embedding(4, config['d_model'])

        self.conv1 = nn.Conv2d(22, 64, (1, 5), stride=(1, 3))
        self.conv2 = nn.Conv2d(64, 64, (1, 5), stride=(1, 3))
        
        self.decoder = DecoderLayer(config['d_model'], 
                                    config['nhead'], 
                                    config['dim_feedforward'], 
                                    config['dropout'])
        
        self.dropout = torch.nn.Dropout(config['dropout'])
        
        self.final_layer = torch.nn.Linear(4 *config['d_model'], 4)

    def forward(self, inp):
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
        
        # Project the input to a higher dim, then performing positional encoding
        inp = self.inp_encoding(inp)
        enc_output = self.pos_encoding(inp)
        enc_output = self.dropout(enc_output)
        
        # Class encoding, make sure the q, k, v are have the same dim
        classes = torch.arange(4).expand(enc_output.shape[0], -1)
        classes = self.class_encoding(classes)
        
        # Performing decoding, produce the raw score for each class, perform softmax when trainning
        dec_output = self.decoder(classes, enc_output)
        dec_output = dec_output.view(dec_output.shape[0], -1)
        dec_output = self.dropout(dec_output)
        output = self.final_layer(dec_output)
        return output