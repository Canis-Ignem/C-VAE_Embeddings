#TORCH
from torchtext.data.utils import get_tokenizer
import torch.nn.functional as F
import torch.distributions as D
from torch import nn
import torch

#UTILS
from torchsummary import summary




tokenizer = get_tokenizer("basic_english")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER = torch.load("models/encoder.pth").to(device)
VOCAB = torch.load("models/vocab.pth").to(device)


def encode_word(word):
    
    prior = D.Normal(torch.zeros(512,).to(device), torch.ones(512,).to(device))
    x = torch.zeros( (2,len(VOCAB), 1) ).to(device)

    x[0][VOCAB[word]][0] = 1
    out = ENCODER(x)
    
    z_mu = out[0, 0, :]
    z_logvar = out[0, 1, :]         
    epsilon = prior.sample()
    
    z = z_mu.to(device) + epsilon.to(device) * (z_logvar.to(device) / 2).exp()
    
    return z


sentences = ["I like to eat", "I like to drink", "I like to sleep", "I like to play"]




def encode_sentence(sentence):
    
    prior = D.Normal(torch.zeros(512,).to(device), torch.ones(512,).to(device))
    words = tokenizer(sentence)   
    x = torch.zeros( (len(words),len(VOCAB), 1) ).to(device)
    
    for i in range(len(words)):
        
        x[i][VOCAB[words[i]]][0] = 1
        
    out = ENCODER(x)
    z_mu = out[:, 0, :]
    z_logvar = out[:, 1, :]         
    epsilon = prior.sample()
    
    z = z_mu.to(device) + epsilon.to(device) * (z_logvar.to(device) / 2).exp()
    z = z.mean(dim=0)
    
    return z


    