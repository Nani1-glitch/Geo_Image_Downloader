import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from custom_dataset import CustomDataset
from custom_model import MoCo, load_model
import numpy as np

# File paths
geo_model_path = "/Users/nithinrajulapati/Downloads/PROJECT 1/output/geography_aware_model.pth"
moco_model_path = "/Users/nithinrajulapati/Downloads/PROJECT 1/output/moco_model.pth"
valid_csv_path = '/Users/nithinrajulapati/Downloads/PROJECT 1/output/valid_images.csv'
image_root_dir = '/Users/nithinrajulapati/Downloads/PROJECT 1/output/images'

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
geo_model = load_model(geo_model_path, model_type='custom')
moco_model = load_model(moco_model_path, model_type='custom')

# Load dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
valid_dataset = CustomDataset(csv_file=valid_csv_path, root_dir=image_root_dir, transform=transform)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

# Function to evaluate model
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.encoder_q(inputs)
            _, preds = torch.max(outputs, 1)  # Convert model output to class predictions
            
            # Debugging prints
            print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
            print(f"Outputs: {outputs}, Labels: {labels}")

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Ensure labels are in the correct format
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    # Debugging lengths
    print(f"Length of all_preds: {len(all_preds)}, Length of all_labels: {len(all_labels)}")

    if len(all_preds) != len(all_labels):
        min_length = min(len(all_preds), len(all_labels))
        all_preds = all_preds[:min_length]
        all_labels = all_labels[:min_length]

    # Ensure all_preds and all_labels are of type int
    all_preds = all_preds.astype(int)
    all_labels = all_labels.astype(int)

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, report, cm, all_labels, all_preds

# Evaluate models
geo_model_accuracy, geo_model_report, geo_model_cm, geo_all_labels, geo_all_preds = evaluate_model(geo_model, valid_dataloader)
moco_model_accuracy, moco_model_report, moco_model_cm, moco_all_labels, moco_all_preds = evaluate_model(moco_model, valid_dataloader)

# Print results
print(f"Geography Aware Model Accuracy: {geo_model_accuracy}")
print(f"Geography Aware Model Classification Report:\n{geo_model_report}")
print(f"MoCo Model Accuracy: {moco_model_accuracy}")
print(f"MoCo Model Classification Report:\n{moco_model_report}")

# Plot confusion matrices
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(set(geo_all_labels)))
    plt.xticks(tick_marks, tick_marks, rotation=45)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plot_confusion_matrix(geo_model_cm, title='Geography Aware Model Confusion Matrix')

plt.subplot(1, 2, 2)
plot_confusion_matrix(moco_model_cm, title='MoCo Model Confusion Matrix')

plt.show()
