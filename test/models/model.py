import torch
from torch import nn
import torch.nn.functional as F

class MyNeuralNet(torch.nn.Module):
    """ Basic neural network class. 
    
    Args:
        in_features: number of input features
        out_features: number of output features
    
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x