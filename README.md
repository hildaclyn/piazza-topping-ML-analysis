# piazza-topping-ML-analysis

### **📌 Pizza Topping Classification - README**  
A **Vision Transformer (ViT) powered pizza topping classifier** designed to identify multiple toppings on a pizza from an image. This project is ideal for **food recognition, menu digitization, and visual AI in e-commerce applications**.

---

## **📝 Project Overview**  
This project uses **ViT (Vision Transformer)** to classify pizza toppings from images. The model is trained on a dataset of **9,213 pizza images** with labeled toppings. It can predict multiple toppings per image using a **multi-label classification** approach.

---

## 📂 Dataset
This project uses a dataset sourced from **Kaggle**:  
**[Pizza Images with Topping Labels](https://www.kaggle.com/datasets/michaelbryantds/pizza-images-with-topping-labels)**  
The dataset was originally collected as part of the **MIT PizzaGAN project** ([MIT PizzaGAN](http://pizzagan.csail.mit.edu/)).  

📌 **Dataset Overview:**
- 9,213 labeled pizza images
- Labels for common toppings: `pepperoni, mushrooms, onions, bacon, etc.`
- Data collected via **crowdsourcing on Amazon Mechanical Turk (AMT)**

⚠ **Note:**  
The `data/` folder in this repository contains **only a few sample images** for testing purposes.  
The full dataset is too large to be stored in this repository. You can download the full dataset from:  
🔗 **[Kaggle Dataset - Pizza Images with Topping Labels](https://www.kaggle.com/datasets/michaelbryantds/pizza-images-with-topping-labels)**  

---

### **📊 Data Distribution**  
| Topping        | Count  |
|---------------|--------|
| Pepperoni     | 2044   |
| Onion         | 933    |
| Black Olives  | 851    |
| Mushrooms     | 875    |
| Pineapple     | 83     |
| Bacon         | 187    |

⚠ **Imbalanced Data**: Some toppings (e.g., pineapple, bacon) appear much less frequently than others (e.g., pepperoni).

---

## **🛠️ Tech Stack**
- **Python**
- **PyTorch**
- **Transformers (Hugging Face)**
- **OpenCV (Image Processing)**
- **Google Colab (Training)**
- **Matplotlib & Seaborn (Visualization)**

---

## **📌 Model Training Process**
### **1️⃣ Data Preprocessing**
- Images are **resized** to 224x224.
- **One-hot encoding** is applied to toppings.
- **Data augmentation** (rotation, color jitter, flipping) improves generalization.

### **2️⃣ Model Architecture**
- **ViT (Vision Transformer)** is used as a feature extractor.
- **Multi-label classification** with `sigmoid()` activation.
- **Focal Loss** is applied to handle **class imbalance**.

### **3️⃣ Training Details**
- **Optimizer**: AdamW (`lr=2e-5`)
- **Loss Function**: Custom **Focal Loss** with dynamic class weights.
- **Batch Size**: 8
- **Epochs**: 5 (can be extended)

---

## **📊 Evaluation & Results**
### **✅ Accuracy**
| Model | Test Accuracy |
|-------|--------------|
| ViT   | **83%**      |

### **🔍 Common Misclassifications**
- `onion` → mistaken as `black_olive`
- `pepperoni` → confused with `bacon`
- `pineapple` → often ignored (due to class imbalance)

✅ **To improve accuracy**, we used:
- **Focal Loss** to give more weight to underrepresented classes.
- **Data augmentation** to improve generalization.

---

## **🚀 How to Run**

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### **1️⃣ Install Dependencies**
```bash
pip install torch torchvision transformers opencv-python matplotlib seaborn
```

### **2️⃣ Load & Preprocess Data**
```python
from data_preprocessing import preprocess_data
df = preprocess_data("path_to_dataset")
```

### **3️⃣ Train the Model**
```python
from training import train_model
model = train_model(df)
```

### **4️⃣ Test with an Image**
```python
from inference import predict_pizza_toppings
toppings = predict_pizza_toppings("pizza.jpg", model)
print("Predicted Toppings:", toppings)
```

---

## **🔮 Future Improvements**
✅ **Switch to ResNet50** for more stable results on small datasets.  
✅ **Introduce Attention Mechanisms** to improve feature focus.  
✅ **Expand Dataset** with more diverse pizza images.  
✅ **Deploy as an API** for real-world applications.  

---

## **📢 Contributors**
- **Xindan Zhang** - ML Engineer  
- **Xindan Zhang** - AI Research  

🚀 **Feel free to contribute!** Open a pull request or report issues.  

---

## 📜 License

This project is open-source under the **MIT License**.

⚠ **Note**: The dataset used in this project was obtained from **Kaggle**:  
**[Pizza Images with Topping Labels](https://www.kaggle.com/datasets/michaelbryantds/pizza-images-with-topping-labels)**  
The dataset was originally sourced from **MIT's PizzaGAN project** ([MIT PizzaGAN](http://pizzagan.csail.mit.edu/)).  

The dataset license is **determined by the original authors**. If you plan to use this dataset for commercial purposes, please check with the dataset provider for permission. 

---

### **🎉 Ready to build your own pizza classifier? Let's get started! 🍕🔥**
