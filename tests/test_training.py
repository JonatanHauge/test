import os

import torch


def test_training():
    # train_pl(1e-3, 1, "models/", False)
    assert os.path.exists("models/trained_model.pt")

    os.environ["WANDB_MODE"] = "offline"
    exit_code = os.system("make train train_params=test_params")

    assert exit_code == 0, "Training failed"
