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



optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr = lr, betas=(0.5, 0.999))

for epoch in range(epochs):
    reconstruct_loss = 0    #total reconstruction loss
    kl_loss = 0             #total kl divergence loss
    train_loss = 0          #total train loss(reconstruction + 2*kl loss)
    encoder.train()
    decoder.train()

    for i in range(0, train.size(0) - 1):
        prior = D.Normal(torch.zeros(512, ), torch.ones(512,))
        x , y = dh.get_batch(train, i)
        
        input = torch.zeros( (1,len(vocab), 1) )
        output = torch.zeros( (1,len(vocab), 1) )
        
        for j in range(dh.batch_size):
            input[0][x][0] = 1
            output[0][y][0] = 1
            
        optimizer.zero_grad()
        
        encoded_op = encoder(input) 
        print(encoded_op.shape)
        
        z_mu = encoded_op[0, 0, :]
        z_logvar = encoded_op[0, 1, :]
        
        reconstruction_loss = 0            
        epsilon = prior.sample()
        
        print(epsilon.shape)
        print(z_mu.shape)
        print(z_logvar.shape)
        
        z = z_mu + epsilon * (z_logvar / 2).exp()
        print(z.shape)
        output_data = decoder(z.unsqueeze(0).unsqueeze(0)).squeeze(0)
        print(output_data.shape)
        print(output.shape)
        reconstruction_loss += F.binary_cross_entropy(output_data.squeeze(0), output.detach(), size_average=False)
        
        q = D.Normal(z_mu, (z_logvar / 2).exp())
        kld_loss = D.kl_divergence(q, prior).sum()
        reconstruct_loss += reconstruction_loss.item()
        kl_loss += kld_loss.item()
        loss = (reconstruction_loss + 2 * kld_loss)        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
    if epoch % 1:
        print("Epoch: {} \t Loss: {} \t reconstruction_loss: {} \t KL Loss: \t:  {} ".format(epoch, train_loss, reconstruct_loss, kl_loss))
        
        # VALIDATION
        with torch.no_grad():
            
            val_reconstruct_loss = 0    
            val_kl_loss = 0             
            val_loss = 0 
            
            for i in range(0, val.size(0) - 1):
                
                prior = D.Normal(torch.zeros(512, ), torch.ones(512,))
                x , y = dh.get_batch(val, i)
                
                input = torch.zeros( (1,len(vocab), 1) )
                output = torch.zeros( (1,len(vocab), 1) )
                
                for j in range(dh.batch_size):
                    input[0][x][0] = 1
                    output[0][y][0] = 1
                    
                optimizer.zero_grad()
                
                encoded_op = encoder(input)
                print(encoded_op.shape)
                
                z_mu = encoded_op[0, 0, :]
                z_logvar = encoded_op[0, 1, :]
                
                reconstruction_loss = 0            
                epsilon = prior.sample()
                
                print(epsilon.shape)
                print(z_mu.shape)
                print(z_logvar.shape)
                
                z = z_mu + epsilon * (z_logvar / 2).exp()
                print(z.shape)
                output_data = decoder(z.unsqueeze(0).unsqueeze(0)).squeeze(0)
                print(output_data.shape)
                print(output.shape)
                reconstruction_loss += F.binary_cross_entropy(output_data.squeeze(0), output.detach(), size_average=False)
                
                q = D.Normal(z_mu, (z_logvar / 2).exp())
                kld_loss = D.kl_divergence(q, prior).sum()
                val_reconstruct_loss += reconstruction_loss.item()
                val_kl_loss += kld_loss.item()
                loss = (reconstruction_loss + 2 * kld_loss)        
                val_loss += loss.item()
                optimizer.step()
            print("Epoch: {} \t Loss: {} \t reconstruction_loss: {} \t KL Loss: \t:  {} ".format(epoch, val_loss, val_reconstruct_loss, val_kl_loss))
    
    high = vocab("high")
    tall = vocab("tall")
    
    high = torch.zeros( (1,len(vocab), 1) )[0][high][0] = 1
    tall = torch.zeros( (1,len(vocab), 1) )[0][tall][0] = 1
    
    high_emb = encoder(high) 
    tall_emb = encoder(tall)
    
    print( F.cosine_similarity(high_emb, tall_emb) )