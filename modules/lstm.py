import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size,hidden_size, num_layers, device):
        super(LSTM,self)._init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.fc = nn.Sequential(
                nn.Linear(hidden_size,64),
                nn.ReLU(inplace=True),
                nn.Linear(64,32),
                nn.ReLU(inplace=True),
                nn.Linear(32,1)  )
        self.dropout = nn.Dropout(p=0.2)
        
        
    def forward(self,x):
        out,_ = self.lstm(x)
        out = self.fc(out[:,-1])
        
        return out 
                
                
                
                
                
                
                
                

