import os
import sys

# Add the project directory to the sys.path
sys.path.append('/Users/nithinrajulapati/Downloads/PROJECT 1')

import torch

# Path to your trained models
moco_model_path = "/Users/nithinrajulapati/Downloads/PROJECT 1/output/moco_model.pth"
geo_model_path = "/Users/nithinrajulapati/Downloads/PROJECT 1/output/geography_aware_model.pth"

if not os.path.exists(moco_model_path) or not os.path.exists(geo_model_path):
    # Train the model if it doesn't exist
    import src.train as train_script
    train_script.train_combined_model()
else:
    print("Trained models already exist. Skipping training step.")

# Now proceed with preprocessing and evaluation
import src.preprocess_images as preprocess_script
preprocess_script.preprocess_images("/Users/nithinrajulapati/Downloads/PROJECT 1/output/images", "/Users/nithinrajulapati/Downloads/PROJECT 1/output/processed_images.csv")

import src.evaluate as evaluate_script
evaluate_script.evaluate_combined_model()
