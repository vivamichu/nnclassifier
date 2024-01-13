import os
import torch
from torchvision import transforms, datasets
from PIL import Image
import numpy as np

def load_data(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'valid', 'test']}
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True, pin_memory=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=False, pin_memory=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=False, pin_memory=True)
    }

    return image_datasets, dataloaders

def save_checkpoint(model, image_datasets, optimizer, epochs, checkpoint_path='flower_checkpoint2.pth'):
    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx,
        'epochs': epochs,
        'optimizer_state': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def process_image(image_path):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a PyTorch Tensor
    '''
    image = Image.open(image_path)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor_image = preprocess(image)

    return tensor_image