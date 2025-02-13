import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision import datasets
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

# Load a pre-trained ResNet model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Define loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Wrap the model with ART's PyTorchClassifier
classifier = PyTorchClassifier(
    model=model,
    clip_values=(0, 1),
    loss=loss_fn,
    optimizer=optimizer,
    input_shape=(3, 224, 224),
    nb_classes=1000
)

# Load a sample image from CIFAR-10 dataset
dataset = datasets.CIFAR10(root="./data", train=False, download=True)
image, label = dataset[0]  # Get first image

# Apply preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = transform(image).unsqueeze(0).numpy()  # Convert to numpy

# Apply FGSM attack
attack = FastGradientMethod(estimator=classifier, eps=0.03)
adv_image = attack.generate(image)

print("Evasion attack completed!")
