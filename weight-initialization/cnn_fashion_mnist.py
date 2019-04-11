from torch import nn
from torch.nn.functional as F

class Cnn_fashion_mnist(nn.Module):
    
    def __init__(self, hidden_i = 256, hidden_ii = 128, constant_weight = None):
        super(Cnn_fashion_mnist, self).__init__()

        self.hidden_layer_i = nn.Linear(784, hidden_i)

        self.hidden_layer_ii = nn.Linear(hidden_i, hidden_ii)

        self.output = nn.Linear(hidden_ii, 10)
        
        self.dropout = nn.Dropout(p = 0.3)
        
        if constant_weight is not None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, constant_weight)
                    nn.init.constant_(m.bias, 0)
        

    def forward(self, x):

        x = F.relu(self.hidden_layer_i(x))
        
        x = self.dropout(x)
        
        x = F.relu(self.hidden_layer_ii(x))
        
        x = self.dropout(x)
        
        x = self.output(x)
        
        return x