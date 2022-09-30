import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(160000, 1000)
        self.rnn = nn.LSTM(1000, 1000, 4)
    
    def forward(self, x):
        x = self.embed(x) 
        x, (h, c) = self.rnn(x)
        return (h, c)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(80000, 1000)
        self.rnn = nn.LSTM(1000, 1000, 4)
        self.predict = nn.Linear(1000, 80000)
    
    def forward(self, x, h, c):
        x = self.embed(x)
        x, (h, c) = self.rnn(x, (h, c))
        x = self.predict(x)
        return x, (h, c)

class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, target=None):
        pass