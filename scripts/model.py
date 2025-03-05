import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor

def build_vit(num_labels):
    """Load pre-trained ViT model for multi-label classification."""
    model_name = "google/vit-base-patch16-224-in21k"
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    return model, feature_extractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
