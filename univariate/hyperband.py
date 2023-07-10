import time
from typing import Tuple

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

EPOCHS = 5
TRIALS = 2
LR = [1e-5, 1e-1]
BATCH_SIZE = [64, 128, 256, 512]
HIDDEN_sIZE = [64, 128, 256, 512]
NUM_LAYERS = [1, 3]
OPT_RANDOM_RESIZE_CROP = 1
OPT_RANDOM_FLIP = 1
OPT_RANDOM_ROTATION = 1


# Multi-layer perceptron model
class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.view(x.shape[0], -1)
        for layer in self.layers:
            out = torch.relu(layer(out))
        out = self.output_layer(out)
        return out


# Function to get data loaders
def get_data_loaders(train_transforms: transforms.Compose, test_transforms: transforms.Compose, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    # Load MNIST training data
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=train_transforms)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Define test loader
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=test_transforms)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

    return trainloader, testloader


# Function to create the model
def create_model(lr: float, hidden_size: int, num_layers: int, device: torch.device) -> Tuple[nn.Module, nn.CrossEntropyLoss, optim.Adam]:
    model = MLP(784, hidden_size, num_layers, 10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer


# Function to get transforms
def get_transforms(trial: optuna.trial.Trial) -> Tuple[transforms.Compose, transforms.Compose]:
    transform_list = [transforms.ToTensor()]
    if OPT_RANDOM_RESIZE_CROP:
        use_random_resized_crop = trial.suggest_categorical('use_random_resized_crop', [0, 1])
        if use_random_resized_crop:
            transform_list.append(transforms.RandomResizedCrop(size=28, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC))

    if OPT_RANDOM_FLIP:
        use_random_flip = trial.suggest_categorical('use_random_flip', [0, 1])
        if use_random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

    if OPT_RANDOM_ROTATION:
        use_random_rotation = trial.suggest_categorical('use_random_rotation', [0, 1])
        if use_random_rotation:
            transform_list.append(transforms.RandomRotation(degrees=(0, 180)))

    transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    train_transforms = transforms.Compose(transform_list)
    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    return train_transforms, test_transforms


# Function to evaluate the model
def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


# Define objective function for Optuna study
def objective(trial: optuna.trial.Trial) -> float:
    lr = trial.suggest_float('lr', LR[0], LR[1], log=True)
    batch_size = trial.suggest_categorical('batch_size', BATCH_SIZE)
    hidden_size = trial.suggest_categorical('hidden_size', HIDDEN_sIZE)
    num_layers = trial.suggest_int('num_layers', NUM_LAYERS[0], NUM_LAYERS[1])

    train_transforms, test_transforms = get_transforms(trial)

    trainloader, testloader = get_data_loaders(train_transforms, test_transforms, batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, criterion, optimizer = create_model(lr, hidden_size, num_layers, device)

    tqdm.write(f"Current trial: learning rate {lr}, batch size {batch_size}, hidden size {hidden_size}, num_layers {num_layers}")
    tqdm.write(f"Training transforms: {train_transforms}")
    time.sleep(0.1)  # forcing time between log and tqdm bar because they're still overlapping even with tqdm.write - prob. due to tqdm version/bug

    val_accuracy = 0
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(trainloader, total=len(trainloader), leave=False)
        for i, data in enumerate(loop):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}] - Validation accuracy: {val_accuracy}")

        val_accuracy = evaluate_model(model, testloader, device)
        trial.report(val_accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_accuracy


def main():
    optuna.logging.enable_default_handler()
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction='maximize', study_name="MNIST_MLP_optimization")

    study.optimize(objective, n_trials=TRIALS)

    best_params = study.best_params
    best_accuracy = study.best_value

    tqdm.write(f"\nBest accuracy: {best_accuracy}\n")
    tqdm.write(f"Best hyperparameters: {best_params}\n")

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("plots/hyperband_optimization_history.pdf")

    fig = optuna.visualization.plot_intermediate_values(study)
    fig.write_image("plots/hyperband_intermediate_values.pdf")


if __name__ == '__main__':
    main()
