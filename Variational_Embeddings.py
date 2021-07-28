from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool2d, Module, Softmax, BatchNorm1d, Dropout, LeakyReLU, Sigmoid
from torchtext.data.functional import sentencepiece_tokenizer, generate_sp_model
import torch.nn.functional as F
import torch.distributions as D
from torch import nn
import torch

#UTILS
from torchsummary import summary





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER = torch.load("models/encoder.pth").to(device)
VOCAB = torch.load("models/vocab.pth").to(device)

spm = generate_sp_model(".data/WikiText2/wikitext-2-v1.zip", vocab_size= len(VOCAB), model_type="word")

sp_tokens_generator = sentencepiece_tokenizer(spm)


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



'''
def encode_word(sentence):
    
    
    prior = D.Normal(torch.zeros(512,).to(device), torch.ones(512,).to(device))
    x = torch.zeros( (2,len(VOCAB), 1) ).to(device)
    
    x[0][VOCAB[word]][0] = 1
    out = ENCODER(x)
    
    z_mu = out[0, 0, :]
    z_logvar = out[0, 1, :]         
    epsilon = prior.sample()
    
    z = z_mu.to(device) + epsilon.to(device) * (z_logvar.to(device) / 2).exp()
    return z


print(encode_word("cat").size())
''' 
    