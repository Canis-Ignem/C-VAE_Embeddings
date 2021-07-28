from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool2d, Module, Softmax, BatchNorm1d, Dropout, LeakyReLU, Sigmoid
from torch import nn
import torch

#UTILS
from torchsummary import summary


ENCODER = torch.load("models/encoder.pth")
VOCAB = torch.load("models/vocab.pth")

def encode_word(word):
    
    x = torch.tensor( (1, len(VOCAB),1 ) ).cuda()
    x[0][VOCAB[word]][0] = 1
    out = ENCODER(x)
    return out


print(encode_word("cat"))
    
    