from sklearn.metrics import accuracy_score
from inference import predict_pizza_toppings, df

def evaluate_model():
    """Evaluate model performance on test images."""
    correct_predictions = 0
    total_samples = 10

    for i in range(total_samples):
        test_image = df.iloc[i]["image_path"]
        true_labels = df.iloc[i, 1:].to_numpy().astype(int)
        predicted_toppings = predict_pizza_toppings(test_image)

        predicted_labels = [1 if label in predicted_toppings else 0 for label in df.columns[1:]]
        correct_predictions += sum(predicted_labels == true_labels)

    accuracy = correct_predictions / (total_samples * len(df.columns[1:]))
    print(f"Model Accuracy on test images: {accuracy:.2%}")

if __name__ == "__main__":
    evaluate_model()
