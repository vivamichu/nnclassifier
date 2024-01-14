import argparse
import json
import torch
from model import FlowerClassifier
from utils import process_image

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    
    model = FlowerClassifier(num_classes=len(checkpoint['class_to_idx']))
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, checkpoint['class_to_idx']

def predict(image_path, model, topk=5, device='cuda'):
    model.eval()
    
    image = process_image(image_path)
    image = image.unsqueeze(0) 

    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        probabilities, classes = torch.exp(output).topk(topk)

    return probabilities, classes

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('checkpoint', help='Path to the checkpoint file')
    parser.add_argument('--top_k', dest='top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json', help='Mapping of categories to real names')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

    model, class_to_idx = load_checkpoint(args.checkpoint)
    model = model.to(device)

    idx_to_class = {idx: class_ for class_, idx in class_to_idx.items()}

    probabilities, classes = predict(args.image_path, model, topk=args.top_k, device=device)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    class_names = [cat_to_name[idx_to_class[idx]] for idx in classes.cpu().numpy()[0]]

    for i in range(args.top_k):
        print(f"Top {i + 1}: {class_names[i]} with probability {probabilities.cpu().numpy()[0][i]:.4f}")

if __name__ == "__main__":
    main()
