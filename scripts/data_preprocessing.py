import os
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

# Define dataset paths
DATA_DIR = "/your_path"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CSV_PATH = os.path.join(DATA_DIR, "labels.csv")

def load_data():
    """Load dataset from CSV and validate image paths."""
    df = pd.read_csv(CSV_PATH)
    df["image_path"] = df["image_name"].apply(lambda x: os.path.join(IMAGE_DIR, x))
    df = df.drop(columns=["image_name"])
    df = df[df["image_path"].apply(os.path.exists)]  # Filter out missing images
    df[df.columns.difference(["image_path"])] = df[df.columns.difference(["image_path"])].astype("float32")
    return df

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.RandomAffine(degrees=30, translate=(0.2, 0.2)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class PizzaDataset(Dataset):
    """PyTorch Dataset class for pizza images."""
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = dataframe.drop(columns=["image_path"]).values.astype("float32")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["image_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = self.labels[idx]
        return image, labels

