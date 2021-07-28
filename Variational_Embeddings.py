from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool2d, Module, Softmax, BatchNorm1d, Dropout, LeakyReLU, Sigmoid
from torch import nn
import torch

#UTILS
from torchsummary import summary


ENCODER = torch.load("models/encoder.py")


def encode_word(word):
    return
    
    