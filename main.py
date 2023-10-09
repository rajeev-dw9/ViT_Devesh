import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import VisionTransformer
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# Define the Vision Image Transformer (ViT) Ensemble Model
class ViTEnsemble(nn.Module):
    def __init__(self, num_models, num_classes):
        super(ViTEnsemble, self).__init__()
        self.models = nn.ModuleList([VisionTransformer.from_name('ViT-B_16_224', num_classes=num_classes) for _ in range(num_models)])

    def forward(self, x):
        # Get predictions from all models
        outputs = [model(x) for model in self.models]
        # Average the predictions
        avg_output = torch.stack(outputs).mean(0)
        return avg_output

# Adversarial Perturbation
def adversarial_perturbation(model, data, target, epsilon):
    data.requires_grad = True
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, target)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * data.grad.data.sign()
    perturbed_data = data + perturbation
    return perturbed_data

# Hyperparameters
num_models = 3
num_classes = 7  # For example, for 7 facial emotions
learning_rate = 0.001
epsilon = 0.03  # For adversarial perturbation
batch_size = 32
epochs = 10

# Load Dataset
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = ImageFolder(root='path_to_dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the Ensemble Model
model = ViTEnsemble(num_models=num_models, num_classes=num_classes).to('cuda')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training Loop
for epoch in range(epochs):
    for data, target in dataloader:
        data, target = data.to('cuda'), target.to('cuda')
        
        # Apply adversarial perturbation
        perturbed_data = adversarial_perturbation(model, data, target, epsilon)
        
        optimizer.zero_grad()
        outputs = model(perturbed_data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

print("Training Complete!")
