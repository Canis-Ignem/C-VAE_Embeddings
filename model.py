#TORCH
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, LeakyReLU, Sigmoid
from torch import nn
import torch

#UTILS
from torchsummary import summary

class Encoder(Module):
    
    def __init__(self, vocab_size, z_size):
        
        super(Encoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.z_size = z_size
        
        self.encode = Sequential(
            Conv1d(vocab_size, vocab_size//2 , 1, 1),
            LeakyReLU(0.1, inplace=True),
            Conv1d(vocab_size//2, vocab_size//4, 1, 2),
            LeakyReLU(0.1, inplace=True),
            Conv1d(vocab_size//4, vocab_size//8, 1, 2),
            LeakyReLU(0.1, inplace=True),
            Conv1d(vocab_size//8, z_size*2, 1, 1)
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
            Conv1d(z_size, vocab_size//8, 1, 1),
            LeakyReLU(0.1, inplace=True),
            Conv1d(vocab_size//8, vocab_size//4, 1, 2),
            LeakyReLU(0.1, inplace=True),
            Conv1d(vocab_size//4, vocab_size//2, 1, 2),
            LeakyReLU(0.1, inplace=True),
            Conv1d(vocab_size//2, vocab_size, 1, 4),
            Sigmoid()
        )
        
    def forward(self, x):
        
        x = self.decode(x)
        x = x.view(-1, 1, self.vocab_size)
        return x