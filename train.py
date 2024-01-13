import argparse
from model import FlowerClassifier
from utils import load_data, save_checkpoint
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

def train(model, criterion, optimizer, dataloaders, device, epochs=10):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Epoch {}/{} - Training Loss: {:.4f}".format(epoch + 1, epochs, running_loss / len(dataloaders['train'])))

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        validation_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                validation_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print("Epoch {}/{} - Validation Loss: {:.4f}, Accuracy: {:.4f}".format(epoch + 1, epochs, validation_loss / len(dataloaders['valid']), accuracy))


def main():
    parser = argparse.ArgumentParser(description='Train a flower classifier')
    parser.add_argument('data_dir', help='Directory of the dataset')
    parser.add_argument('--save_dir', dest='save_dir', default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', dest='arch', default='vgg16', help='Architecture (default: vgg16)')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.0009, help='Learning rate')
    parser.add_argument('--hidden_units', dest='hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

    image_datasets, dataloaders = load_data(args.data_dir)

    model = FlowerClassifier(num_classes=102, hidden_units=args.hidden_units)  # Assuming 102 classes for flowers
    model = model.to(device)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train(model, criterion, optimizer, dataloaders, device, epochs=args.epochs)

    save_checkpoint(model, image_datasets, optimizer, args.epochs, checkpoint_path=os.path.join(args.save_dir, 'flower_checkpoint.pth'))

if __name__ == "__main__":
    main()
