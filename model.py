#TORCH
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, LeakyReLU, Sigmoid
from torch import nn
import torch

#UTILS
from torchsummary import summary
'''
class Encoder(Module):
    
    def __init__(self, vocab_size, z_size):
        
        super(Encoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.z_size = z_size
        
        self.encode = Sequential(
            Conv1d(vocab_size, 16 , 1, 1),
            LeakyReLU(0.1, inplace=True),
            Conv1d(16, 32, 1, 2),
            LeakyReLU(0.1, inplace=True),
            Conv1d(32, 64, 1, 2),
            LeakyReLU(0.1, inplace=True),
            Conv1d(64, z_size*2, 1, 1)
        )
        
        
    def forward(self, x):
        
        x = self.encode(x)
        x =  x.view(-1, 2, self.z_size )
        return x
    
class Decoder(Module):
    
    def __init__(self, vocab_size, z_size):
        
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.z_size = z_size
        
        self.decode = nn.Sequential(
            Conv1d(1, 64, 1, 1),
            LeakyReLU(0.1, inplace=True),
            Conv1d(64, 32, 1, 2),
            LeakyReLU(0.1, inplace=True),
            Conv1d(32, 64, 1, 2),
            LeakyReLU(0.1, inplace=True),
            Conv1d(64, vocab_size, 1, 128),
            Sigmoid()
        )
        
    def forward(self, x):
        
        x = self.decode(x)
        x = x.view(-1, self.vocab_size,1)
        return x
    
'''

class Encoder(Module):
    
    def __init__(self, vocab_size, z_size):
        
        super(Encoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.z_size = z_size
        
        self.encode = Sequential(
            Conv1d(vocab_size, 64 , 1, 1),
            LeakyReLU(0.1, inplace=True),
            Conv1d(64, 128, 1, 2),
            LeakyReLU(0.1, inplace=True),
            Conv1d(128, 256, 1, 2),
            LeakyReLU(0.1, inplace=True),
            Conv1d(256, z_size*2, 1, 1)
        )
        
        
        
    def forward(self, x):
        
        x = self.encode(x)
        x =  x.view(-1, 2, self.z_size )
        return x


class Decoder(Module):
    
    def __init__(self, vocab_size, z_size):
        
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.z_size = z_size
        
        self.decode = nn.Sequential(
            Conv1d(1, 256, 1, 1),
            LeakyReLU(0.1, inplace=True),
            Conv1d(256, 128, 1, 2),
            LeakyReLU(0.1, inplace=True),
            Conv1d(128, 64, 1, 2),
            LeakyReLU(0.1, inplace=True),
            Conv1d(64, vocab_size, 1, 1),
            Sigmoid()
        )
        
    def forward(self, x):
        print(x.shape)
        x = self.decode(x)
        print(x.shape)
        x = x.view(-1, self.vocab_size,1)
        return x

 
d = Decoder(28782,512).to("cpu")

print(summary(d,(1,512)))
