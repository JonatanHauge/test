import click
import torch
from models.model import MyNeuralNet
import matplotlib.pyplot as plt
import os
import hydra


@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=10, help="Number of training epochs")
@click.option("--output-dir", default="model_checkpoints/", help="Directory to save model checkpoints")
@click.option("--plot", default=False, help="Plot loss over epochs if True")
def train(lr, epochs, output_dir, plot):
    """Train a model on MNIST."""
    print("Training day and night")
    print("Learning_rate", lr)
    print("epochs:", epochs)

    # TODO: Implement training loop here
    model = MyNeuralNet(784, 10, 256, 128, 64, 0.5)
    train_images = torch.load("data/processed/corruptmnist/train_images.pt")
    train_targets = torch.load("data/processed/corruptmnist/train_target.pt")
    train_set = torch.utils.data.TensorDataset(train_images, train_targets)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.NLLLoss()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    loss_epochs = []
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss/len(train_loader)}")
            loss_epochs.append(running_loss / len(train_loader))
    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.plot(loss_epochs)
        plt.savefig("reports/figures/loss.png")
        plt.show()

    checkpoint_path = os.path.join(output_dir, f"trained_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print("Model checkpoint: ", model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyNeuralNet(784, 10, 256, 128, 64, 0.5)
    model_dict = torch.load(os.path.join(f"model_checkpoints/", model_checkpoint))
    model.load_state_dict(model_dict)
    test_images = torch.load("data/processed/corruptmnist/test_images.pt")
    test_targets = torch.load("data/processed/corruptmnist/test_target.pt")
    test_set = torch.utils.data.TensorDataset(test_images, test_targets)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, drop_last=False)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.view(images.shape[0], -1)
            output = model(images)
            _, predicted = torch.max(output.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total

    print(f"Accuracy of the model on the test images: {accuracy:.2f}%")


cli.add_command(train)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
