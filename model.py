# model.py

import torch
import torch.nn as nn
import torchvision.models as models

class FlowerClassifier(nn.Module):
    def __init__(self, num_classes, hidden_units=4096):
        super(FlowerClassifier, self).__init__()

        # Load the pre-trained VGG16 model
        vgg16 = models.vgg16(pretrained=True)

        # Freeze the pre-trained layers
        for param in vgg16.features.parameters():
            param.requires_grad = False

        # Extract the features (excluding the classifier)
        self.features = vgg16.features

        # Calculate the number of features after the convolutional layers
        num_conv_features = self._get_num_conv_features()

        # Add a new classifier with customizable hidden units
        self.classifier = nn.Sequential(
            nn.Linear(num_conv_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the convolutional features
        x = self.classifier(x)
        return x

    def _get_num_conv_features(self):
        # Test with a dummy input to calculate the number of features
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.features(dummy_input)
            num_conv_features = features.view(features.size(0), -1).size(1)
        return num_conv_features
