import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from PIL import Image
import os
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

class HouseDataset(Dataset):
    def __init__(self, tabular_data_path, image_base_dir, transform=None):
        self.tabular_data = pd.read_csv(tabular_data_path)
        self.image_base_dir = image_base_dir
        self.transform = transform
        self.image_paths_dict = self._create_image_paths_dict()

    def _create_image_paths_dict(self):
        image_files = [f for f in os.listdir(self.image_base_dir) if f.endswith(".jpg")]
        image_paths_dict = {}
        for f in image_files:
            match = re.match(r"(\d+)_(\w+).jpg", f)
            if match:
                house_id = int(match.group(1))
                room_type = match.group(2)
                if house_id not in image_paths_dict:
                    image_paths_dict[house_id] = {}
                image_paths_dict[house_id][room_type] = os.path.join(self.image_base_dir, f)
        return image_paths_dict

    def __len__(self):
        return len(self.tabular_data)

    def __getitem__(self, idx):
        house_id = self.tabular_data.iloc[idx]["house_id"]
        tabular_features = self.tabular_data.iloc[idx][["bedrooms", "bathrooms", "area", "zipcode"]].values.astype(float)
        price = self.tabular_data.iloc[idx]["price"]

        images = {}
        # Only use frontal image for now to speed up training
        expected_types = ["frontal"]
        house_images = self.image_paths_dict.get(house_id, {})

        for img_type in expected_types:
            img_path = house_images.get(img_type)
            if img_path and os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                images[img_type] = image
            else:
                images[img_type] = torch.zeros(3, 224, 224) 

        return {
            "tabular_features": torch.tensor(tabular_features, dtype=torch.float32),
            "images": images,
            "price": torch.tensor(price, dtype=torch.float32)
        }

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56) 
        x = F.relu(self.fc1(x))
        return x

class MultimodalModel(nn.Module):
    def __init__(self, cnn_feature_dim, tabular_feature_dim):
        super(MultimodalModel, self).__init__()
        self.cnn_extractor = CNNFeatureExtractor()
        
        # Only 1 image per house, so 1 * cnn_feature_dim from images
        self.combined_feature_dim = cnn_feature_dim + tabular_feature_dim
        
        self.fc_combined = nn.Sequential(
            nn.Linear(self.combined_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1) 
        )

    def forward(self, tabular_features, images):
        frontal_features = self.cnn_extractor(images["frontal"])

        image_features = frontal_features

        combined_features = torch.cat((image_features, tabular_features), dim=1)
        
        output = self.fc_combined(combined_features)
        return output

if __name__ == '__main__':
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tabular_data_path = 'housing_data.csv'
    image_base_dir = 'Houses-dataset-master/Houses Dataset'

    full_dataset = HouseDataset(tabular_data_path, image_base_dir, image_transform)
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    cnn_feature_dim = 128
    tabular_feature_dim = 4
    model = MultimodalModel(cnn_feature_dim, tabular_feature_dim)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1 

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            tabular_features = data["tabular_features"]
            images = data["images"]
            prices = data["price"]

            optimizer.zero_grad()
            predictions = model(tabular_features, images).squeeze()
            loss = criterion(predictions, prices)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_dataloader):.4f}")

    print("Training finished.")

    model.eval()
    all_predictions = []
    all_true_prices = []

    with torch.no_grad():
        for data in test_dataloader:
            tabular_features = data["tabular_features"]
            images = data["images"]
            prices = data["price"]

            predictions = model(tabular_features, images).squeeze()
            all_predictions.extend(predictions.cpu().numpy())
            all_true_prices.extend(prices.cpu().numpy())

    mae = mean_absolute_error(all_true_prices, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_true_prices, all_predictions))

    print(f"\nEvaluation Results:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")


