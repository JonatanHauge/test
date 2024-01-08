from test.models.model import MyNeuralNet
import torch
import pytest

@pytest.mark.parametrize("test_input,expected", [((10,28,28), torch.Size([10, 10])), ((20,28,28), torch.Size([20, 10]))])
def test_model(test_input, expected):
    input = torch.randn(test_input)
    model = MyNeuralNet(784, 10, 256, 128, 64, 0.5)
    output = model(input)
    assert output.shape == expected