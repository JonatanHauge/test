import torch
from test.models.model import MyNeuralNet
import argparse


def predict(
    model_checkpoint: str,
    data_path: str,
) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    model = MyNeuralNet(784, 10, 256, 128, 64, 0.5)
    model_dict = torch.load(model_checkpoint)
    model.load_state_dict(model_dict)
    model.eval()

    new_data = torch.load(data_path)

    # Normalize (Skal ændres til at normalisere med mean og std fra train set)
    mean = new_data.mean()
    std = new_data.std()
    new_data = (new_data - mean) / (std + 1e-8)  # Normalize new data

    dataloader = torch.utils.data.DataLoader(new_data, batch_size=64, shuffle=False)
    preds = torch.Tensor([])
    for images in dataloader:
        images = images.view(images.shape[0], -1)
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        preds = torch.cat((preds, predicted), dim=0)

    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction for a given model and dataloader.")
    parser.add_argument("model_checkpoint", type=str, help="Path to the model checkpoint file")
    parser.add_argument("data_path", type=str, help="Path to the data file")

    args = parser.parse_args()

    torch.save(predict(args.model_checkpoint, args.data_path), "test/predictions.pt")
    print("Model predictions saved to test/predictions.pt")
