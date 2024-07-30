# src/generate_valid_csv.py

import os
import pandas as pd

def generate_valid_csv(input_csv, output_csv, root_dir):
    data_frame = pd.read_csv(input_csv)
    valid_rows = []
    
    for idx in range(len(data_frame)):
        img_name = os.path.join(root_dir, data_frame.iloc[idx, 0])
        if os.path.exists(img_name):
            valid_rows.append(data_frame.iloc[idx])
    
    valid_data_frame = pd.DataFrame(valid_rows)
    valid_data_frame.to_csv(output_csv, index=False)
    print(f"Valid CSV generated with {len(valid_data_frame)} entries.")

# Usage
input_csv = '/Users/nithinrajulapati/Downloads/PROJECT 1/output/processed_images.csv'
output_csv = '/Users/nithinrajulapati/Downloads/PROJECT 1/output/valid_images.csv'
root_dir = '/Users/nithinrajulapati/Downloads/PROJECT 1/output/images'

generate_valid_csv(input_csv, output_csv, root_dir)
