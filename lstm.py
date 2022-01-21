import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class LSTM(nn.Module):

    def __init__(self,batch_size,input_size,output_size):
        super(LSTM, self).__init__()
        self.batch_size=batch_size
        self.input_size=input_size
        self.output_size=output_size
        self.hidden_size =10

        self.fc = nn.Linear(self.hidden_size,self.output_size,bias=True)

    def __call__(self, x):
        x=x.reshape(self.batch_size,-1,self.input_size)
        self.num_layers=x.shape[1]
        self.lstm = nn.LSTM(input_size=self.input_size, 
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers, 
                            batch_first=True)
                
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        
        return out

