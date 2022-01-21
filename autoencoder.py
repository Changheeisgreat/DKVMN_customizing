import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class AUTOENCODER(nn.Module):
    """Makes the main denoising auto
    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self,mode):
        super(AUTOENCODER, self).__init__()
        self.mode=mode
        if self.mode==0:
           #train
            self.in_shape=18530
            self.out_shape=1090*10
            self.x=7486
            
        elif self.mode==1:
            self.in_shape=153
            self.out_shape=9*10
            self.x=8093
            
        self.encode = nn.Sequential(
            nn.Linear(self.in_shape, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, self.out_shape),
        )
        
        self.decode = nn.Sequential(
            nn.BatchNorm1d(self.out_shape),
            nn.Linear(self.out_shape, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, self.in_shape)
        )
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    
def di_reduction(model,x):
    error=nn.MSELoss()
    optimizer=optim.Adam(model.parameters())
    n_epochs=5
    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        output = model(x)
        loss = error(output, x)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f'epoch {epoch} \t Loss: {loss.item():.4g}')

    with torch.no_grad():
        encoded = model.encode(x)
        enc = encoded.cpu().detach().numpy()
    return enc
