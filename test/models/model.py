import torch
from torch import nn
import torch.nn.functional as F


class MyNeuralNet(torch.nn.Module):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self, in_features: int, out_features: int, hidden_1:int, hidden_2:int, hidden_3:int, dropout:float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_1)
        self.bn1 = nn.BatchNorm1d(hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.bn2 = nn.BatchNorm1d(hidden_3)
        self.fc4 = nn.Linear(hidden_3, out_features)

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
