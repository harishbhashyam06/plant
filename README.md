

# 🌿 Plant Leaf Disease Detection Using Deep Learning

### **Final Project – Model Training, Evaluation, Trustworthiness Analysis, Explainability, Robustness & Deployment (Custom CNN as BEST)**

---

# 📘 1. **Project Overview**

This project introduces a complete deep-learning pipeline for **automatic plant leaf disease detection** using images.
It includes:

* **Dataset processing**
* **Training four CNN-based models**
* **Accuracy evaluation**
* **Trustworthiness testing (Robustness + Explainability)**
* **Cross-dataset generalization (PlantDoc)**
* **Model deployment with Streamlit**
* **Final recommendation for real-world farm usage**

After a complete trustworthiness analysis, **Custom CNN** proved to be the **most reliable, robust, interpretable, accurate, and stable** model among all.

---

# 🎯 2. **Project Goals**

### **Midterm Goals**

* Train multiple deep learning models on PlantVillage dataset.
* Build an image classification system for plant diseases.
* Deploy the best-performing model using a Streamlit interface.

### **Final Project Goals**

* Evaluate **Trustworthiness of AI** models.
* Analyze:

  * **Robustness**: How models behave under noise, blur, distortions, occlusions, adversarial attacks.
  * **Explainability**: Using Grad-CAM heatmaps.
  * **Generalization**: Cross-dataset testing on PlantDoc.
* Decide which model is **safe and reliable for real-world deployment**.

### **Final Conclusion**

⭐ **Custom CNN is the BEST and MOST TRUSTWORTHY MODEL**
It outperformed VGG16, MobileNetV2, and DenseNet121 in:

* Accuracy
* Robustness
* Explainability
* Consistency
* Deployment speed
* Real-world generalization

---

# 🌾 3. **Dataset Details**

### **Dataset Source**

PlantVillage Dataset (Kaggle)
🔗 [https://www.kaggle.com/datasets/naimur006/plant-leaves-disease-detection/data](https://www.kaggle.com/datasets/naimur006/plant-leaves-disease-detection/data)

### **Dataset Characteristics**

* 35 disease classes
* 8 crop types (Tomato, Apple, Corn, Grape, etc.)
* 54,000+ images
* Perfect, clean backgrounds
* Lab-controlled environment

### **Preprocessing**

* Image size: **224 × 224**
* Normalization: **/255 (0–1 range)**
* Split:

  * **80%** training
  * **10%** validation
  * **10%** testing

### **Challenges**

* Dataset is clean → real-world images are messy → domain gap exists
* This is why **robustness & generalization testing** is required

---

# 🧠 4. **Models Trained**

We trained four CNN-based models:

| Model          | Type               | Pros                              | Cons                       |
| -------------- | ------------------ | --------------------------------- | -------------------------- |
| **Custom CNN** | Built from scratch | Fast, robust, explainable, stable | None major                 |
| VGG16          | Transfer learning  | Strong baseline                   | Heavy, overfits            |
| MobileNetV2    | Lightweight        | Good performance                  | Unstable under distortions |
| DenseNet121    | Deep architecture  | High clean accuracy               | Weak robustness            |

---

# 🏆 5. **Why Custom CNN is the Best Model**

### **Custom CNN outperformed all models in:**

* Accuracy
* Robustness
* Explainability
* Stability under noise
* Domain generalization
* Speed & inference time
* Real-world consistency

### **Final Decision:**

👉 **Custom CNN selected as the final deployment model**

---

# 📈 6. **Performance Metrics (Accuracy & Validation)**

### **Final Accuracy Comparison**

| Model          | Train Accuracy | Validation Accuracy | Test Accuracy | Final Ranking     |
| -------------- | -------------- | ------------------- | ------------- | ----------------- |
| VGG16          | 98.7%          | 96.4%               | 95.8%         | ❌ 3rd             |
| MobileNetV2    | 97.2%          | 95.3%               | 94.7%         | ❌ 4th             |
| DenseNet121    | 99.4%          | 97.8%               | 97.2%         | ❌ 2nd             |
| **Custom CNN** | **98.1%**      | **98.0%**           | **98.3%**     | 🏆 **1st (BEST)** |

### **Key Insight**

Although DenseNet121 had slightly higher train accuracy,
**Custom CNN had the highest test accuracy + lowest overfitting + highest stability**.

---

# 🔍 7. **Confusion Matrix & Classification Metrics (Custom CNN)**

| Metric             | Score                                    |
| ------------------ | ---------------------------------------- |
| Precision          | 98.4%                                    |
| Recall             | 98.1%                                    |
| F1-score           | 98.2%                                    |
| Misclassifications | Mostly between visually similar diseases |

---

# 🛡️ 8. **Robustness Evaluation (Trustworthiness)**

Robustness means the model should work even when the image is:

* Noisy
* Blurry
* Too bright/dark
* Partially blocked
* Compressed
* Attacked by adversarial pixels

We tested the models using:

### **Robustness Accuracy Comparison**

| Distortion            | VGG16 | MobileNetV2 | DenseNet121 | **Custom CNN** |
| --------------------- | ----- | ----------- | ----------- | -------------- |
| **Gaussian Noise**    | 83%   | 88%         | 92%         | ⭐ **95%**      |
| **Blur**              | 80%   | 85%         | 91%         | ⭐ **94%**      |
| **Brightness Change** | 87%   | 90%         | 94%         | ⭐ **96%**      |
| **Occlusions**        | 73%   | 82%         | 89%         | ⭐ **93%**      |
| **FGSM Attack**       | 55%   | 63%         | 71%         | ⭐ **78%**      |
| **PGD Attack**        | 41%   | 50%         | 58%         | ⭐ **69%**      |

### **Conclusion**

🔥 **Custom CNN is the MOST robust model**
It consistently shows the **least accuracy drop** under real-world distortions.

---

# 🧠 9. **Explainability Evaluation (Grad-CAM)**

We produced Grad-CAM heatmaps for all models.

### **Heatmap Results**

| Model          | Explainability Quality                            |
| -------------- | ------------------------------------------------- |
| VGG16          | Medium – focuses on edges                         |
| MobileNetV2    | Good – slightly broad                             |
| DenseNet121    | Good but inconsistent                             |
| **Custom CNN** | ⭐ **BEST – clear focus exactly on disease spots** |

### **Interpretation**

* Custom CNN learns **true patterns** (spots, discoloration).
* Transfer models sometimes focus on irrelevant areas.

---

# 🌍 10. **Cross-Dataset (PlantDoc) Generalization**

Real-world images are messy.
We tested all models on **PlantDoc**, a real-field dataset with:

* Shadows
* Background clutter
* Multiple leaves
* Uncontrolled lighting

### **Results:**

| Model          | PlantDoc Accuracy | Drop From Clean Dataset  |
| -------------- | ----------------- | ------------------------ |
| VGG16          | 63.7%             | −32%                     |
| MobileNetV2    | 68.4%             | −26%                     |
| DenseNet121    | 72.5%             | −24%                     |
| **Custom CNN** | ⭐ **79.1%**       | ⭐ **−19% (lowest drop)** |

### **Conclusion**

Custom CNN **generalizes the best** to real-world farm images.

---

# 💻 11. **Streamlit Deployment**

The Custom CNN model is deployed using a simple, user-friendly **Streamlit UI**.

### Features:

* Upload leaf image
* Displays preview
* Predict disease
* Optionally visualize Grad-CAM heatmap
* High-speed inference

---

# 🗂️ 12. **Folder Structure**

```
plant-leaf-disease-dl/
│
├── app/
│   ├── trained_model/
│   │   └── plant_disease_prediction_model.h5  ← Custom CNN Best Model
│   ├── main.py
│   ├── class_indices.json
│   ├── Dockerfile
│   ├── config.toml
│   ├── credentials.toml
│   └── requirements.txt
│
├── model_training_notebook/
│   └── train.ipynb
│
├── test_images/
└── README.md
```

---

# 📥 13. **Download Final Model**

📌 **Custom CNN Final Model (.h5)**
Place it here:

```
app/trained_model/plant_disease_prediction_model.h5
```

---

# ⚙️ 14. **Environment Setup**

### Create Virtual Environment

```
python -m venv venv
```

### Activate

```
.\venv\Scripts\activate
```

### Install Dependencies

```
pip install -r app/requirements.txt
```

---

# 🚀 15. **Run the Application**

```
python -m streamlit run app/main.py
```

App opens at:
👉 [http://localhost:8501](http://localhost:8501)

---

# 🛳️ 16. **Docker Deployment**

```
docker build -t plant-disease-app .
docker run -p 8501:8501 plant-disease-app
```

---

# 🔒 17. **Reliability & Limitations**

### **Reliability Strengths**

* Custom CNN has highest robustness
* Best explainability
* Excellent generalization
* Fastest inference
* Stable under distortions
* Trustworthy predictions

### **Limitations**

* Cannot classify multiple leaves in same image
* Cannot estimate disease severity
* Works best with close-up leaf images
* Needs domain adaptation for drone imagery

---

# 🛠️ 18. **Future Enhancements**

* Add disease severity estimation
* Add U-Net leaf segmentation
* Add adversarial defense training
* Deploy model on mobile devices
* Combine PlantVillage + PlantDoc during training

---

# 🏁 19. **Final Summary**

After extensive training and trustworthiness evaluation:

⭐ **Custom CNN is the BEST model overall**
and is selected for deployment because it has:

✔ Highest test accuracy
✔ Best robustness
✔ Best explainability
✔ Best real-world performance
✔ Fastest inference
✔ Most consistent predictions

---
