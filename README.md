# 📝 YOLOv8 Layout Recognition for Historical Documents  

This project utilizes **YOLOv8** for **Layout Organization Recognition** in **historical scanned documents**.  
The model detects **text regions**, including **headings, main text, authors, and drop caps**, while ignoring embellishments. 

---

## 🚀 Features  

✔ **Dataset:** Custom dataset of **historical document images** with detailed annotations.  
✔ **Model:** **YOLOv8m**, fine-tuned with custom **augmentation** & **hyperparameters**.  
✔ **Training:** **180 epochs**, optimized learning rate, batch size, and augmentation techniques.  
✔ **Evaluation:** Precision, Recall, **mAP@0.5**, **mAP@0.5:0.95**, and **training loss analysis**.  
✔ **Inference:** Predicts layouts on **new document images** and **saves** structured outputs.  
✔ **Visualization:** Training loss, **mAP trends, Precision-Recall curves, and Confusion Matrix**.  

---

## 📂 Project Structure  

📂 Layout_Recognition-YOLOv8/
┣ 📂 data/ # Dataset (Train, Test, Validation, data.yaml)
┣ 📂 YOLO_results/ # Trained models & predictions
┣ 📜 Layout_Recognition.ipynb # Jupyter Notebook (Training & Inference)
┣ 📜 README.md # Project Documentation
┣ 📜 LICENSE # MIT License

## ⚙️ Requirements  

- **Python 3.7+**  
- **Ultralytics YOLOv8**  
- **PyTorch, Pandas, Matplotlib, Pillow (PIL)**

## 🚀 Installation  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/YOLOv8-Layout-Recognition.git
```
### 2️⃣ Install Required Packages
```bash
pip install ultralytics
pip install torch
pip install pandas matplotlib pillow
```

### 📜 Dataset & Model  
🔹 **Dataset**: Scanned historical documents with varying layouts.  
🔹 **Model**: Trained using **YOLOv8m**, **AdamW optimizer**, and **custom augmentations**.  

📥 **Download Dataset & YOLO Results:**  
- 📁 [Dataset (Train, Test, Valid)](https://drive.google.com/drive/folders/1tvZZfsfFHPlLx26hQEDGcEnAJ6h9g0wm?usp=drive_link)  
- 📁 [YOLO Results (Trained Weights & Predictions)](https://drive.google.com/drive/folders/1hjbZ72TodFKLEgVIRXIIpPbUUwb7eqom?usp=drive_link)  

## 🎯 Model Performance  

📌 **Final Evaluation Metrics:**  

| **Metric**    | **Value**  |
|--------------|-----------|
| Precision    | 85.4%     |
| Recall       | 86.3%     |
| mAP@0.5      | 91.2%     |
| mAP@0.5:0.95 | 75.9%     |

📊 **Training Loss & mAP Trends Over Epochs:**  

![Training Loss & mAP Trends](path/to/training_loss_map.png) ![download](https://github.com/user-attachments/assets/82e2b6ef-e2c8-4684-9735-b0061bd7f2b2)

📈 **Precision-Recall Curve:**  

![Precision-Recall Curve](path/to/precision_recall_curve.png)

## 🚀 Observations  
✔ **Good model convergence** with minimal overfitting.  
✔ **Improved recall & precision**, enhancing layout recognition accuracy.  
✔ **Strong detection** of main text and drop caps, with minor misclassifications in headings and author labels.  
✔ **mAP@0.5 = 91.2%** and **mAP@0.5:0.95 = 75.9%**, indicating reliable detection but scope for layout refinement.  

---

## 🔄 Next Steps for Improvement  
✔ **Expand dataset** (current: 52 images) and add more text variations.  
✔ **Apply stronger augmentations** (rotation, shearing, perspective) to improve generalization.  
✔ **Fine-tune confidence threshold & IoU ** for better text region segmentation.  
✔ **Experiment with YOLOv8L or YOLOv8X** for improved feature extraction.  
✔ **Implement OCR post-processing** to assess text extraction accuracy.  
✔ **Optimize heading recognition & threshold settings** for enhanced classification.  

## 📜 License  
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.  

## 👨‍💻 Author  
📌 **Ashish Raj**  
📧 [ar469492@gmail.com] | 🖥️ [Your GitHub](https://github.com/your-username)  

🔹 If you find this project useful, give it a ⭐ on GitHub! 🚀  
