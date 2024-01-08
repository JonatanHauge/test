import unittest
import torch
import os


def test_training():
    #train_pl(1e-3, 1, "models/", False)
    assert os.path.exists('models/trained_model.pt')

    exit_code = os.system('make train train_params=test_params')

    assert exit_code == 0, 'Training failed'



