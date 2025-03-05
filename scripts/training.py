import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_preprocessing import load_data, PizzaDataset, transform
from model import build_vit, device

# Define Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        """
        alpha: Class weight dictionary {label_index: weight}
        gamma: Controls focus on hard-to-classify samples
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        """Compute Focal Loss."""
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)  # Convert to probability
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        if self.alpha:
            weight = torch.tensor([self.alpha[i] for i in range(targets.shape[1])]).to(inputs.device)
            focal_loss = weight * focal_loss
        return focal_loss.mean()

# Train function
def train_model(epochs=3, batch_size=16, lr=2e-5):
    df = load_data()
    dataset = PizzaDataset(df, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_labels = len(df.columns) - 1
    model, feature_extractor = build_vit(num_labels)
    model.to(device)

    criterion = FocalLoss(alpha=None, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

    return model

if __name__ == "__main__":
    train_model()
