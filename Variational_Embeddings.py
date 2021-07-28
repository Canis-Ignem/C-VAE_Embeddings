from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool2d, Module, Softmax, BatchNorm1d, Dropout, LeakyReLU, Sigmoid
import torch.nn.functional as F
import torch.distributions as D
from torch import nn
import torch

#UTILS
from torchsummary import summary


ENCODER = torch.load("models/encoder.pth")
VOCAB = torch.load("models/vocab.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def encode_word(word):
    
    
    prior = D.Normal(torch.zeros(512,).to(device), torch.ones(512,).to(device))
    x = torch.zeros( (1,len(VOCAB), 1) )
    
    x[0][VOCAB[word]][0] = 1
    out = ENCODER(x)
    
    z_mu = out[:, 0, :]
    z_logvar = out[:, 1, :]         
    epsilon = prior.sample()
    
    z = z_mu.to(device) + epsilon.to(device) * (z_logvar.to(device) / 2).exp()
    return z


print(encode_word("cat"))
    
    