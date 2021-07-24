from model import Encoder, Decoder
from torchvision import transforms
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
import data_handler as dh
import torch.nn as nn
import torchvision
import torch





lr = 0.00002
epochs = 5

train, val, _, vocab = dh.get_data()
encoder = Encoder(len(vocab), 512)
decoder = Decoder(len(vocab), 512)


train, val, _, l = dh.get_data()


optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr = lr, betas=(0.5, 0.999))

for epoch in range(epochs):
    reconstruct_loss = 0    #total reconstruction loss
    kl_loss = 0             #total kl divergence loss
    train_loss = 0          #total train loss(reconstruction + 2*kl loss)
    encoder.train()
    decoder.train()

    for i in range(0, train.size(0) - 1):
        prior = D.Normal(torch.zeros(512, ), torch.ones(512, 512))
        x , y = dh.get_batch(train, i)
        
        input = torch.zeros( (1,len(vocab), 1) )
        output = torch.zeros( (1,len(vocab), 1) )
        
        for j in range(dh.batch_size):
            input[0][x][0] = 1
            output[0][y][0] = 1
            
        optimizer.zero_grad()
        
        encoded_op = encoder(input) #output statistics for latent space
        print(encoded_op.shape)
        
        z_mu = encoded_op[0, :, 0]
        z_logvar = encoded_op[0, :, 1]
        
        reconstruction_loss = 0            #loss for a batch
        epsilon = prior.sample()
        print(epsilon.shape)
        print(z_mu.shape)
        print(z_logvar.shape)
        z = z_mu + epsilon * (z_logvar / 2).exp()
        print(z.shape)
        output_data = decoder(z.unsqueeze(2))
        
        reconstruction_loss += F.binary_cross_entropy(output_data, output.detach(), size_average=False)
        
        q = D.Normal(z_mu, (z_logvar / 2).exp())
        kld_loss = D.kl_divergence(q, prior).sum()
        reconstruct_loss += reconstruction_loss.item()
        kl_loss += kld_loss.item()
        loss = (reconstruction_loss + 2 * kld_loss)        #total loss for the processed batch
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
    if epoch % 1:
        print("Epoch: {} \t Loss: {} \t reconstruction_loss: {} \t KL Loss: \t:  {} ".format(epoch, train_loss, reconstruct_loss, kl_loss))