import os
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Define data directory and transformations
data_dir = "/Users/nithinrajulapati/Downloads/PROJECT 1/output/images"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to filter out classes without valid image files
def get_valid_classes(data_dir):
    valid_classes = []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for root, _, files in os.walk(class_dir):
                if any(file.endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')) for file in files):
                    valid_classes.append(class_name)
                    break
    return valid_classes

# Filter classes and create a subset of the dataset
valid_classes = get_valid_classes(data_dir)
print(f"Valid classes found: {valid_classes}")

valid_data_dir = os.path.join(data_dir, "valid")
os.makedirs(valid_data_dir, exist_ok=True)
for class_name in valid_classes:
    src_class_dir = os.path.join(data_dir, class_name)
    dst_class_dir = os.path.join(valid_data_dir, class_name)
    if not os.path.exists(dst_class_dir):
        os.symlink(src_class_dir, dst_class_dir)

# Load dataset and create dataloader
try:
    dataset = datasets.ImageFolder(valid_data_dir, transform=transform)
    if len(dataset) == 0:
        raise ValueError("The dataset directory does not contain any valid image files.")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
except FileNotFoundError as e:
    print(e)
    exit()
except ValueError as e:
    print(e)
    exit()

print(f"Number of classes: {len(dataset.classes)}")
print(f"Classes: {dataset.classes}")

# Load the model
model = models.resnet50(weights=None)  # Adjust according to your specific model setup
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(dataset.classes))
model.load_state_dict(torch.load("/Users/nithinrajulapati/Downloads/PROJECT 1/z_FINAL_RESULTS/moco_v2_resnet50.pth"))
print("Model loaded successfully.")

# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels

# Evaluate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Evaluating model...")
all_preds, all_labels = evaluate_model(model, dataloader, device)

# Generate classification report
report = classification_report(all_labels, all_preds, target_names=dataset.classes, zero_division=1)
print("Classification Report:\n", report)

# Calculate Top-1 and Top-5 accuracy
def top_k_accuracy(output, target, k=1):
    with torch.no_grad():
        max_k_preds = torch.topk(output, k, dim=1).indices
        return (max_k_preds == target.view(-1, 1)).float().mean().item()

model.eval()
top1_acc = 0
top5_acc = 0
with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        top1_acc += top_k_accuracy(outputs, labels, k=1)
        top5_acc += top_k_accuracy(outputs, labels, k=5)

top1_acc /= len(dataloader)
top5_acc /= len(dataloader)

print(f"Top-1 Accuracy: {top1_acc * 100:.2f}%")
print(f"Top-5 Accuracy: {top5_acc * 100:.2f}%")
