#TORCH
from torchtext import data, datasets

#SPACY
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

#UTILS
import pandas as pd
import numpy as np
import torchtext


import io
import torch
from torchtext.datasets import WikiText103, WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter

bptt = 35
batch_size = 1
val_batch_size = 10
tokenizer = get_tokenizer('basic_english')

def preprocess(raw_text_iter, vocab):
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def to_batches(data, batch_size):

    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    return data.to(device)
  
def get_batch(source, i):
    seq_len = min(1, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data.squeeze(0), target.squeeze(0)

def get_data():

    train_iter = WikiText2(split='train')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    train_iter, val_iter, test_iter = WikiText2()
    
    train_data = preprocess(train_iter, vocab)
    val_data = preprocess(val_iter, vocab)
    test_data = preprocess(test_iter, vocab)
    
    train_data = to_batches(train_data, batch_size)
    val_data = to_batches(val_data, val_batch_size)
    test_data = to_batches(test_data, val_batch_size)
    
    return train_data, val_data, test_data, vocab


train, val, _, l = get_data()
print(train.shape)
print(train.max())
print(len(l))
print(l["blue"])
'''
x , y = get_batch(train, 0)

input = torch.zeros((batch_size,len(l),1))
out = torch.zeros((batch_size,len(l),1))

for j in range(batch_size):
    input[j][x[j]][0] = 1
    input[j][y[j]][0] = 1
    print(input[j].shape)
    print(input[j].shape)
'''