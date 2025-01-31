import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Train a flower classifier")
    parser.add_argument("data_dir", type=str, help="Directory of dataset")
    parser.add_argument("--save_dir", type=str, default="./", help="Directory to save the checkpoint")
    parser.add_argument("--arch", type=str, default="vgg16", help="Model architecture (default: vgg16)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    
    return parser.parse_args()

def load_data(data_dir):
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)

    return train_loader, valid_loader, train_data.class_to_idx

def build_model(arch, hidden_units):
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze feature extractor

    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    return model

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        valid_loss = 0
        accuracy = 0
        model.eval()

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(train_loader):.3f}.. "
              f"Valid loss: {valid_loss/len(valid_loader):.3f}.. "
              f"Accuracy: {accuracy/len(valid_loader):.3f}")

def save_checkpoint(model, save_dir, arch, hidden_units, class_to_idx):
    checkpoint = {
        "architecture": arch,
        "hidden_units": hidden_units,
        "class_to_idx": class_to_idx,
        "state_dict": model.state_dict()
    }
    torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pth"))

def main():
    args = get_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, class_to_idx = load_data(args.data_dir)
    model = build_model(args.arch, args.hidden_units)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    train_model(model, train_loader, valid_loader, criterion, optimizer, device, args.epochs)
    save_checkpoint(model, args.save_dir, args.arch, args.hidden_units, class_to_idx)

if __name__ == "__main__":
    main()
