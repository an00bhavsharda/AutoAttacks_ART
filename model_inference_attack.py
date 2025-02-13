import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from art.estimators.classification import PyTorchClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased

# Step 1: Load Dataset (CIFAR-10)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_size = int(0.5 * len(dataset))  # 50% of data for training
test_size = len(dataset) - train_size  # 50% for testing

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 2: Define the Victim Model
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
victim_model = VictimModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(victim_model.parameters(), lr=0.001)

# Step 3: Train the Victim Model
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = victim_model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    break  # Train on a small batch for quick execution

# Step 4: Wrap the Model with ART
victim_classifier = PyTorchClassifier(
    model=victim_model,
    clip_values=(-1, 1),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=10
)

# Step 5: Prepare Attack Data
x_train = []
y_train = []
for images, labels in train_loader:
    x_train.append(images.numpy())
    y_train.append(labels.numpy())
    break  # Use a small subset for testing

x_train = np.concatenate(x_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

x_test = []
y_test = []
for images, labels in test_loader:
    x_test.append(images.numpy())
    y_test.append(labels.numpy())
    break  # Use a small subset for testing

x_test = np.concatenate(x_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

# Step 6: Perform Membership Inference Attack
attack = MembershipInferenceBlackBoxRuleBased(victim_classifier)

# Attack directly without fitting:
train_predictions = attack.infer(x_train, y_train)  # Should return mostly 1s (in training set)
test_predictions = attack.infer(x_test, y_test)  # Should return mostly 0s (not in training set)

# Evaluate Attack Performance
train_success = np.mean(train_predictions)
test_success = 1 - np.mean(test_predictions)

print(f"✅ Attack Success Rate on Training Data: {train_success:.2f}")
print(f"✅ Attack Success Rate on Test Data: {test_success:.2f}")
