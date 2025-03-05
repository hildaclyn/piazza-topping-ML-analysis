import torch
from PIL import Image
from model import build_vit, device
from data_preprocessing import transform, load_data

df = load_data()
model, feature_extractor = build_vit(num_labels=len(df.columns) - 1)
model.load_state_dict(torch.load("models/pizza_vit.pth"))
model.to(device)
model.eval()

def predict_pizza_toppings(image_path):
    """Predict pizza toppings from an image."""
    image = Image.open(image_path).convert("RGB")
    encoding = feature_extractor(images=image, return_tensors="pt")
    image_tensor = encoding["pixel_values"].to(device)

    with torch.no_grad():
        output = model(image_tensor).logits.sigmoid().cpu().numpy()

    predicted_toppings = [df.columns[i+1] for i, score in enumerate(output[0]) if score > 0.3]
    return predicted_toppings

if __name__ == "__main__":
    image_path = "test_pizza.jpg"
    print("Predicted Toppings:", predict_pizza_toppings(image_path))
