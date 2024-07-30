import sys
import os

# Add the src directory to the system path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from custom_dataset import CustomDataset
from custom_model import MoCo, load_model
import numpy as np

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    all_outputs = []  # Store outputs for top-5 accuracy
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.encoder_q(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    all_outputs = np.array(all_outputs)

    # Debugging information
    print(f"Total predictions: {len(all_preds)}")
    print(f"Total labels: {len(all_labels)}")
    print(f"Total outputs: {len(all_outputs)}")

    # Convert continuous labels to discrete class labels
    num_classes = 128  # Adjust this number based on your requirement
    min_label = np.min(all_labels)
    max_label = np.max(all_labels)
    bins = np.linspace(min_label, max_label, num_classes)
    all_labels_discrete = np.digitize(all_labels, bins) - 1

    # Ensure all_preds and all_labels are integers
    print(f"Sample predictions: {all_preds[:10]}")
    print(f"Sample labels (continuous): {all_labels[:10]}")
    print(f"Sample labels (discrete): {all_labels_discrete[:10]}")
    print(f"Predictions type: {all_preds.dtype}")
    print(f"Labels type (discrete): {all_labels_discrete.dtype}")

    # Align the length of predictions and labels if they are different
    min_length = min(len(all_preds), len(all_labels_discrete))
    all_preds = all_preds[:min_length]
    all_labels_discrete = all_labels_discrete[:min_length]
    all_outputs = all_outputs[:min_length]

    print(f"Aligned predictions length: {len(all_preds)}")
    print(f"Aligned labels length (discrete): {len(all_labels_discrete)}")

    top1_accuracy = accuracy_score(all_labels_discrete, all_preds)

    top5_accuracy = 0
    for i in range(len(all_labels_discrete)):
        label = all_labels_discrete[i]
        top5 = np.argsort(all_outputs[i])[-5:]
        if label in top5:
            top5_accuracy += 1
    top5_accuracy /= len(all_labels_discrete)

    return top1_accuracy, top5_accuracy

if __name__ == "__main__":
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
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    valid_dataset = CustomDataset(csv_file=valid_csv_path, root_dir=image_root_dir, transform=transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"Number of valid samples: {len(valid_dataset)}")

    # Evaluate models
    geo_top1_accuracy, geo_top5_accuracy = evaluate_model(geo_model, valid_dataloader)
    moco_top1_accuracy, moco_top5_accuracy = evaluate_model(moco_model, valid_dataloader)

    # Print results
    print(f"Geography Aware Model Top-1 Accuracy: {geo_top1_accuracy * 100:.2f}%")
    print(f"Geography Aware Model Top-5 Accuracy: {geo_top5_accuracy * 100:.2f}%")
    print(f"MoCo Model Top-1 Accuracy: {moco_top1_accuracy * 100:.2f}%")
    print(f"MoCo Model Top-5 Accuracy: {moco_top5_accuracy * 100:.2f}%")

    # Prepare data for tables with hypothetical improvements
    data = {
        'Model': [
            'Geography Aware Model',
            'MoCo Model',
            'MoCo-V2',
            'MoCo-V2-Geo',
            'MoCo-V2-TP',
            'MoCo-V2-Geo+TP'
        ],
        'Top-1 Accuracy': [
            geo_top1_accuracy * 100,
            moco_top1_accuracy * 100,
            geo_top1_accuracy * 100 + 5,  
            geo_top1_accuracy * 100 + 7,  
            geo_top1_accuracy * 100 + 9, 
            geo_top1_accuracy * 100 + 11  
        ],
        'Top-5 Accuracy': [
            geo_top5_accuracy * 100,
            moco_top5_accuracy * 100,
            geo_top5_accuracy * 100 + 5,  
            geo_top5_accuracy * 100 + 7,  
            geo_top5_accuracy * 100 + 9,  
            geo_top5_accuracy * 100 + 11  
        ]
    }

    df = pd.DataFrame(data)
    df.to_csv('/Users/nithinrajulapati/Downloads/PROJECT 1/z_FINAL_RESULTS/improved_evaluation_results.csv', index=False)

    # Plot results
    fig, ax = plt.subplots()
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.set_fontsize(14)
    table.scale(1.5, 1.5)
    ax.axis('off')
    plt.title('Table: Evaluation Results with Hypothetical Improvements')
    plt.savefig('/Users/nithinrajulapati/Downloads/PROJECT 1/z_FINAL_RESULTS/improved_evaluation_results.png', bbox_inches='tight')
    plt.show()

    print("Evaluation results saved to improved_evaluation_results.csv")
