import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import os

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
data_dir = '/Users/nithinrajulapati/Downloads/PROJECT 1/output/images'

# Prepare a new directory for valid classes
valid_data_dir = '/Users/nithinrajulapati/Downloads/PROJECT 1/output/valid_images'
if not os.path.exists(valid_data_dir):
    os.makedirs(valid_data_dir)

# Check each class directory for valid image files
image_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']

for class_dir in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_dir)
    if os.path.isdir(class_path):
        image_files = [f for f in os.listdir(class_path) if os.path.splitext(f)[-1].lower() in image_extensions]
        if image_files:
            new_class_path = os.path.join(valid_data_dir, class_dir)
            if not os.path.exists(new_class_path):
                os.makedirs(new_class_path)
            for image_file in image_files:
                src_path = os.path.join(class_path, image_file)
                dst_path = os.path.join(new_class_path, image_file)
                if not os.path.exists(dst_path):
                    os.symlink(src_path, dst_path)
            print(f"Class: {class_dir}, Files: {image_files}")

# If there are no valid classes with images, raise an error
if not os.listdir(valid_data_dir):
    raise FileNotFoundError("No valid image files found in the specified directory.")

dataset = datasets.ImageFolder(valid_data_dir, transform=transform)
num_classes = len(dataset.classes)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define model
model = models.resnet50(pretrained=None)  # Change to pretrained=None to avoid the warning
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Set the number of output classes dynamically

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 2  # Reduce the number of epochs for faster output
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 0:  # Print progress every 10 batches
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item()}")
    print(f"Epoch {epoch+1} finished, Average Loss: {running_loss/len(train_loader)}")

# Save model
torch.save(model.state_dict(), '/Users/nithinrajulapati/Downloads/PROJECT 1/z_FINAL_RESULTS/sup_learning_resnet50.pth')
