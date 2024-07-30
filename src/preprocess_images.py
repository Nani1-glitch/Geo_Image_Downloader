import pandas as pd
import os

def preprocess_images(image_dir, output_csv):
    data = []
    
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                meta_path = img_path.replace('.jpg', '.json')
                
                if not os.path.exists(meta_path):
                    print(f"Metadata file not found for {img_path}, skipping...")
                    continue
                
                with open(meta_path, 'r') as meta_file:
                    metadata = eval(meta_file.read())
                
                data.append([img_path, metadata['latitude'], metadata['longitude']])
    
    df = pd.DataFrame(data, columns=['file_path', 'latitude', 'longitude'])
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}")




if __name__ == "__main__":
    image_dir = "/Users/nithinrajulapati/Downloads/PROJECT 1/output/images"
    output_csv = "/Users/nithinrajulapati/Downloads/PROJECT 1/output/processed_images.csv"
    preprocess_images(image_dir, output_csv)
