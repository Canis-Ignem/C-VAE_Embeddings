from torch._C import device
from model import Encoder, Decoder
from torchvision import transforms
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim
import data_handler as dh
from tqdm import tqdm
import torch.nn as nn
import torchvision
import argparse
import torch


parser = argparse.ArgumentParser(description="C-VAE")
parser.add_argument('--data', metavar = 'd', type = str, required = False)

args = vars(parser.parse_args())


def validate(epoch, encoder, decoder, device, val_set, vocab):
    
    best_val_loss = 10000
    val_reconstruct_loss = 0    
    val_kl_loss = 0             
    val_loss = 0 
    
    
    
    with torch.no_grad():
        
        print("VALIDATING: \n")
        for i in tqdm(range(0, val_set.size(0)-1)):
            
            prior = D.Normal(torch.zeros(512, ).to(device), torch.ones(512,).to(device))
            x , y = dh.get_batch(val_set, i)
            
            input = torch.zeros( (dh.batch_size,len(vocab), 1) )
            output = torch.zeros( (dh.batch_size,len(vocab), 1) )
            
            for j in range(dh.val_batch_size):
                input[j][x[j]][0] = 1
                output[j][y[j]][0] = 1
            
            encoded_op = encoder(input.to(device)) 
            #print(encoded_op.shape)
            
            z_mu = encoded_op[:, 0, :]
            z_logvar = encoded_op[:, 1, :]      
            epsilon = prior.sample()
            
            #print(epsilon.shape)
            #print(z_mu.shape)
            #print(z_logvar.shape)
            
            z = z_mu.to(device) + epsilon.to(device) * (z_logvar.to(device) / 2).exp()
            #print(z.shape)
            
            output_data = decoder( z.unsqueeze(1).to(device)).squeeze(0) 
            #print(output_data.shape)
            #print(output.shape)
            
            reconstruction_loss = F.binary_cross_entropy(output_data.to(device), output.detach().to(device), size_average=False)
            val_reconstruct_loss += reconstruction_loss.item()
            
            q = D.Normal(z_mu.to(device), (z_logvar.to(device) / 2).exp())
            kld_loss = D.kl_divergence(q, prior).sum()
            val_kl_loss += kld_loss.item()
            
            loss = (reconstruction_loss + 2 * kld_loss)        
            val_loss += loss.item()
            
        print("Epoch: {} \t val_Loss: {} \t val_reconstruction_loss: {} \t val_KL Loss: \t:  {} \n".format(epoch, val_loss, val_reconstruct_loss, val_kl_loss))
        
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            
            torch.save(encoder, "./models/encoder.pth")
            torch.save(decoder, "./models/decoder.pth")
    
    high = vocab["high"]
    tall = vocab["tall"]
    
    high_tensor = torch.zeros( (1,len(vocab), 1) )
    high_tensor[0][high][0] = 1
    
    tall_tensor = torch.zeros( (1,len(vocab), 1) )
    tall_tensor[0][tall][0] = 1
    
    high_op = encoder(high_tensor.to(device))
    tall_op = encoder(tall_tensor.to(device))
    
    high_emb = get_embedding(high_op, prior, device)
    tall_emb = get_embedding(tall_op, prior, device)
    print( F.cosine_similarity(high_emb, tall_emb) )


def train(optimizer, device, encoder, decoder, train_set, val_set, vocab, epochs = 5):
    
    
    
    for epoch in range(epochs):
        reconstruct_loss = 0    #total reconstruction loss
        kl_loss = 0             #total kl divergence loss
        train_loss = 0          #total train loss(reconstruction + 2*kl loss)
        encoder.train()
        decoder.train()
        print("EPOCH: {}\n".format(epoch))
        for i in tqdm( range(0, train_set.size(0) -1 ) ):
            
            prior = D.Normal(torch.zeros(512,).to(device), torch.ones(512,).to(device))
            x , y = dh.get_batch(train_set, i)
            
            input = torch.zeros( (dh.batch_size,len(vocab), 1) )
            output = torch.zeros( (dh.batch_size,len(vocab), 1) )
            
            
            for j in range(dh.batch_size):
                input[j][x[j]][0] = 1
                output[j][y[j]][0] = 1
            
            optimizer.zero_grad()
            
            encoded_op = encoder(input.to(device)) 
            #print(encoded_op.shape)
            
            z_mu = encoded_op[:, 0, :]
            z_logvar = encoded_op[:, 1, :]
            epsilon = prior.sample()
            
            #print(epsilon.shape)
            #print(z_mu.shape)
            #print(z_logvar.shape)
            
            z = z_mu.to(device) + epsilon.to(device) * (z_logvar.to(device) / 2).exp()
            #print(z.shape)
            
            output_data = decoder(z.unsqueeze(1).to(device)).squeeze(0)
            #print(output_data.shape)
            #print(output.shape)
             
            reconstruction_loss = F.binary_cross_entropy(output_data.to(device), output.to(device), size_average=False)
            reconstruct_loss += reconstruction_loss.item()
            
            q = D.Normal(z_mu.to(device), (z_logvar.to(device) / 2).exp())
            kld_loss = D.kl_divergence(q, prior).sum()
            kl_loss += kld_loss.item()
            
            loss = (reconstruction_loss + 2 * kld_loss)        
            loss.backward()
            train_loss += loss.item()
            
            optimizer.step()
            
        if epoch % 2 == 0:
            print("Epoch: {} \t Loss: {} \t reconstruction_loss: {} \t KL Loss: \t:  {}  \n".format(epoch, train_loss, reconstruct_loss, kl_loss))
            
            validate(epoch, encoder, decoder, device, val_set, vocab)
                

def get_embedding(encoded_op, prior, device):
    
    z_mu = encoded_op[:, 0, :]
    z_logvar = encoded_op[:, 1, :]         
    epsilon = prior.sample()
    z = z_mu.to(device) + epsilon.to(device) * (z_logvar.to(device) / 2).exp()
    return z

def main():
    
    lr = 0.00002
    epochs = 50

    train_set, val_set, _, vocab = dh.get_data(args['data'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training using: ", device)

    encoder = Encoder(len(vocab), 512)
    decoder = Decoder(len(vocab), 512)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    best_val_loss = 100

    optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr = lr, betas=(0.5, 0.999))
    
    train(optimizer, device, encoder, decoder, train_set, val_set, vocab, epochs = epochs)

if __name__ == '__main__':
    main()
