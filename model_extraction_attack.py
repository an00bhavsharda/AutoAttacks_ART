import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from art.estimators.classification import PyTorchClassifier
from art.attacks.extraction import KnockoffNets

# Step 1: Define the Victim Model (Target Model)
class VictimModel(nn.Module):
    def __init__(self):
        super(VictimModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x

# Step 2: Load Dataset (CIFAR-10)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Step 3: Train the Victim Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
victim_model = VictimModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(victim_model.parameters(), lr=0.001)

# Training Loop (1 epoch for quick execution)
for images, labels in trainloader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = victim_model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    break  # Train on a small batch for quick execution

# Step 4: Wrap the Victim Model with ART
victim_classifier = PyTorchClassifier(
    model=victim_model,
    clip_values=(-1, 1),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=10
)

# Step 5: Define the Attack Model (Attacker's Model)
attacker_model = VictimModel().to(device)  # Assume attacker knows model structure

attacker_classifier = PyTorchClassifier(
    model=attacker_model,
    clip_values=(-1, 1),
    loss=criterion,
    optimizer=optim.Adam(attacker_model.parameters(), lr=0.001),
    input_shape=(3, 32, 32),
    nb_classes=10
)

# Step 6: Convert DataLoader to NumPy Arrays for Attack
x_train = []
for images, _ in trainloader:
    x_train.append(images.numpy())
    if len(x_train) * 64 >= 1000:  # Stop after 1000 queries
        break

x_train = np.concatenate(x_train, axis=0)

# Step 7: Perform Model Extraction Attack
attack = KnockoffNets(classifier=victim_classifier, batch_size_fit=64, batch_size_query=64, nb_epochs=1)

# ✅ FIXED: Pass the attacker's classifier explicitly
attack.extract(thieved_classifier=attacker_classifier, x=x_train, num_queries=1000)

print("✅ Model extraction attack completed! The attacker has a stolen copy of the victim model.")
