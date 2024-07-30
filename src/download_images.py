import os
import pandas as pd
import requests
from flickrapi import FlickrAPI

FLICKR_PUBLIC = '06de6faa1b76d6e5e0a439cee8450773'
FLICKR_SECRET = '67e1b3767e7d099d'

def download_images(class_csv, output_dir, num_images=100):
    flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')
    class_data = pd.read_csv(class_csv)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for idx, row in class_data.iterrows():
        class_name = row['class_name']
        query = row['class_name']  # Use 'class_name' as the query term
        
        print(f"Searching for images of class: {class_name}")
        photos = flickr.photos.search(text=query, per_page=num_images, extras='geo')
        
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        for i, photo in enumerate(photos['photos']['photo']):
            photo_id = photo['id']
            title = photo['title']
            url = f"http://farm{photo['farm']}.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}.jpg"
            try:
                geo_info = flickr.photos.geo.getLocation(photo_id=photo_id)
                lat = geo_info['photo']['location']['latitude']
                lon = geo_info['photo']['location']['longitude']
                
                image_path = os.path.join(class_dir, f"{title}_{photo_id}.jpg")
                metadata_path = os.path.join(class_dir, f"{title}_{photo_id}.json")
                
                img_data = requests.get(url).content
                with open(image_path, 'wb') as handler:
                    handler.write(img_data)
                
                with open(metadata_path, 'w') as meta_file:
                    meta_file.write(f"{{'latitude': {lat}, 'longitude': {lon}, 'url': '{url}'}}")
                
                print(f"Downloaded {i+1}/{num_images} images for class {class_name}")
            except Exception as e:
                print(f"Failed to download image {i+1}/{num_images} for class {class_name}: {e}")

if __name__ == "__main__":
    class_csv = "/Users/nithinrajulapati/Downloads/PROJECT 1/data/classes_in_imagenet.csv"
    output_dir = "/Users/nithinrajulapati/Downloads/PROJECT 1/output/images"
    download_images(class_csv, output_dir)
