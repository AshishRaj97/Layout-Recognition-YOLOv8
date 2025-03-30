## 📝 YOLOv8 Layout Recognition for Historical Documents  

This project is part of the RenAIssance Test I (Layout Organization Recognition). The goal is to develop a model that detects and segments the main text regions in scanned early modern printed sources, while ignoring embellishments like marginalia and decorations.
This project utilizes **YOLOv8** for **Layout Organization Recognition** in **historical scanned documents**.  
The model detects **text regions**, including **headings, main text, authors, and drop caps**, while ignoring embellishments. 

---

### 🚀 Features  

✔ **Dataset:** Custom dataset of historical document images, manually annotated for text regions.  
✔ **Model:** **YOLOv8m**, fine-tuned with custom **augmentation** & **hyperparameters**.  
✔ **Training:** **180 epochs**, optimized learning rate, batch size, and augmentation techniques.  
✔ **Evaluation:** Precision, Recall, **mAP@0.5**, **mAP@0.5:0.95**, and **training loss analysis**, IoU, and F1-score.  
✔ **Inference:** Predicts layouts on **new document images** and **saves** structured outputs.  
✔ **Visualization:** Training loss, **mAP trends, Precision-Recall curves, and Confusion Matrix**.  

---

### 📜 Dataset & Preprocessing

🔹 Dataset: The dataset consists of 6 scanned early modern printed sources.
🔹 Preprocessing: The PDFs were converted to images and annotated in Roboflow to label only the main text regions. The dataset was then exported in YOLOv8 format for training.
🔹 Challenges: Variability in text layout, marginalia, and embellishments required precise annotation.

📥 **Download Dataset & YOLO Results:**  
- 📁 [Dataset (Train, Test, Valid)](https://drive.google.com/drive/folders/1tvZZfsfFHPlLx26hQEDGcEnAJ6h9g0wm?usp=drive_link)  
- 📁 [YOLO Results (Trained Weights & Predictions)](https://drive.google.com/drive/folders/1hjbZ72TodFKLEgVIRXIIpPbUUwb7eqom?usp=drive_link)

---

### 🎯 Model Selection & Training Details

🔹Why YOLOv8?
  - YOLOv8 offers high-speed object detection and performs well on small datasets.
  - The model is lightweight yet powerful for text region detection in historical documents.
🔹 Training Setup:
  - Model: YOLOv8m
  - Optimizer: AdamW
  - Batch Size: 16
  - Epochs: 180
  - Learning Rate: 0.001
  - Augmentations: Rotation, Contrast Adjustments, Shearing, Perspective Transformations
  - Confidence Threshold & IoU: Tuned for better text region segmentation

---

### 🎯 Model Performance  

📌 **Final Evaluation Metrics:**  

| **Metric**    | **Value**  |
|--------------|-----------|
| Precision    | 85.4%     |
| Recall       | 86.3%     |
| mAP@0.5      | 91.2%     |
| mAP@0.5:0.95 | 75.9%     |
| IoU Score    | 82.5%     |
| F1-Score     | 85.8%     |

📊 **Training Loss & mAP Trends Over Epochs:**  

![download](https://github.com/user-attachments/assets/82e2b6ef-e2c8-4684-9735-b0061bd7f2b2)

📈 **Precision-Recall Curve:**  

![download](https://github.com/user-attachments/assets/e3e92478-d896-4f30-a716-0806564cf981)

✅ Successful Layout Recognition:

![d2869d59-2563-4794-80c4-a8d21063e2c1](https://github.com/user-attachments/assets/7aa4ef01-18a2-46fc-96cb-8b542b2b54f2)

📥 To view all graphs and predicted images, check the **Predicted Images and Curves** folder in the repository.

---

### 🚀 Observations  
✔ **Good model convergence** with minimal overfitting.  
✔ **Improved recall & precision**, enhancing layout recognition accuracy.  
✔ **Strong detection** of main text and drop caps, with minor misclassifications in headings and author labels.  
✔ **mAP@0.5 = 91.2%** and **mAP@0.5:0.95 = 75.9%**, indicating reliable detection but scope for layout refinement.  

---

### 🔄 Next Steps for Improvement  
✔ **Expand dataset** (current: 52 images) and add more text variations.  
✔ **Apply stronger augmentations** (rotation, shearing, perspective) to improve generalization.  
✔ **Fine-tune confidence threshold & IoU ** for better text region segmentation.  
✔ **Experiment with YOLOv8L or YOLOv8X** for improved feature extraction.  
✔ **Implement OCR post-processing** to assess text extraction accuracy.  
✔ **Optimize heading recognition & threshold settings** for enhanced classification. 

---

### ⚙️ Requirements  

- **Python 3.7+**
- **GPU (Recommended): A Tesla T4 (used in Colab) or any NVIDIA GPU with at least 8GB VRAM for faster training.**  
- **Ultralytics YOLOv8**  
- **PyTorch, Pandas, Matplotlib, Pillow (PIL)**

---

### 🚀 Installation  

#### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/AshishRaj97/Layout-Recognition-YOLOv8.git
```
#### 2️⃣ Install Required Packages
```bash
pip install ultralytics
pip install torch
pip install pandas matplotlib pillow
```
---

### 📜 License  
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.  

### 👨‍💻 Author  
📌 **Ashish Raj**  
📧 [ar469492@gmail.com] | 🖥️ [Your GitHub](https://github.com/your-username)  

🔹 If you find this project useful, give it a ⭐ on GitHub! 🚀  
