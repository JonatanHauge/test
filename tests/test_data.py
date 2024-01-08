import torch
from tests import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT
import os
import pytest

#@pytest.mark.skipif(not os.path.exists(os.path.join(_PATH_DATA, "processed", "corruptmnist", "train_images.pt")),
#                    reason="Data files not found")
#@pytest.mark.skipif(not os.path.exists(os.path.join(_PATH_DATA, "processed", "corruptmnist", "test_images.pt")),
#                    reason="Data files not found")
#@pytest.mark.skipif(not os.path.exists(os.path.join(_PATH_DATA, "processed", "corruptmnist", "train_target.pt")),
#                    reason="Data files not found")
#@pytest.mark.skipif(not os.path.exists(os.path.join(_PATH_DATA, "processed", "corruptmnist", "test_target.pt")),
#                    reason="Data files not found")
def test_data_lenght():

    #exit_code = os.system('make data')
    #assert exit_code == 0, 'Data generation failed'
    #train = torch.load(os.path.join(_PATH_DATA, "processed", "corruptmnist", "train_images.pt"))
    #test = torch.load(os.path.join(_PATH_DATA, "processed", "corruptmnist", "test_images.pt"))
    #train_target = torch.load(os.path.join(_PATH_DATA, "processed", "corruptmnist", "train_target.pt"))
    #test_target = torch.load(os.path.join(_PATH_DATA, "processed", "corruptmnist", "test_target.pt"))

    #assert len(train) == len(train_target)
    #assert len(test) == len(test_target)
    #assert train[0].shape == torch.Size([28, 28])
    #assert torch.unique(train_target).shape == torch.Size([10])
    #assert torch.unique(test_target).shape == torch.Size([10])

    assert os.path.exists(os.path.join(_PATH_DATA, "raw", "corruptmnist", "test_images.pt"))