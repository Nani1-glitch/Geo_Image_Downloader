import sys
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from custom_model import MoCo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')  # Ensure image is in RGB format
        if self.transform:
            image = self.transform(image)
        return image, int(self.annotations.iloc[idx, 1])

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def train_contrastive_model():
    train_dataset = ImageDataset(csv_file='/Users/nithinrajulapati/Downloads/PROJECT 1/output/valid_images.csv', 
                                 root_dir='/Users/nithinrajulapati/Downloads/PROJECT 1/output/images',
                                 transform=transforms.Compose([
                                     transforms.Resize((224, 224)),  # Resize images to 224x224
                                     transforms.ToTensor()  # Convert images to tensor
                                 ]))

    print(f"Number of valid samples: {len(train_dataset)}")
    if len(train_dataset) == 0:
        print("No valid samples found. Exiting training.")
        return

    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)

    model = MoCo(base_encoder=models.resnet18, dim=128, K=65536, m=0.999, T=0.07).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)

    for epoch in range(10):  # Replace with the number of epochs you want
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, target = model(inputs, inputs)  # Pass inputs as positional arguments
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0

    torch.save(model.state_dict(), '/Users/nithinrajulapati/Downloads/PROJECT 1/output/trained_model.pth')
    print("Finished Training")

if __name__ == "__main__":
    train_contrastive_model()
