import torch
from torchvision import models, transforms
from PIL import Image
import argparse
import json

def get_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    parser.add_argument("image_path", type=str, help="Path to the image")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("--top_k", type=int, default=3, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to category-to-name JSON file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint["architecture"])(pretrained=True)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[0].in_features, checkpoint["hidden_units"]),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(checkpoint["hidden_units"], 102),
        torch.nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    
    return model

def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(image_path, model, top_k, device):
    model.to(device)
    model.eval()
    
    image = process_image(image_path).to(device)
    
    with torch.no_grad():
        output = model(image)
    
    probs = torch.exp(output).topk(top_k)
    indices = probs.indices.cpu().numpy().squeeze()
    probabilities = probs.values.cpu().numpy().squeeze()
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]
    
    return probabilities, classes

def main():
    args = get_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    model = load_checkpoint(args.checkpoint)
    
    probs, classes = predict(args.image_path, model, args.top_k, device)

    if args.category_names:
        with open(args.category_names, "r") as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name.get(cls, "Unknown") for cls in classes]
    else:
        class_names = classes

    print("\nPredictions:")
    for i in range(args.top_k):
        print(f"{class_names[i]}: {probs[i]*100:.2f}%")

if __name__ == "__main__":
    main()
