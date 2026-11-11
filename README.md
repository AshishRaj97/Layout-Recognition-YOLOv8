# Historical Document Layout Recognition with YOLOv8

A computer-vision project exploring the detection of text regions in scanned early-modern printed documents. The work was completed as part of the RenAIssance Test I (Layout Organization Recognition).

The goal was to identify meaningful page regions—such as main text, headings, author text, and drop caps—while avoiding decorative elements and marginalia.

## Project at a glance

- **Model:** Ultralytics YOLOv8m
- **Dataset:** 6 scanned printed sources, converted to images and manually annotated
- **Annotated image count:** 52
- **Training:** 180 epochs, batch size 16, learning rate 0.001, AdamW
- **Annotation workflow:** Roboflow
- **Evaluation:** precision, recall, F1-score, mAP@0.5, and mAP@0.5:0.95

This is an experimental project on a small, domain-specific dataset. The reported metrics should be interpreted in that context; they do not establish generalization to all historical documents.

## Problem and approach

Historical printed pages can contain multiple text regions alongside ornaments, marginal notes, and inconsistent layouts. A layout detector needs to localize the target regions without treating every visual element as text.

The workflow was:

1. Convert six scanned source documents from PDF to page images.
2. Annotate the target text regions in Roboflow.
3. Export the annotations in YOLO format.
4. Fine-tune YOLOv8m with image augmentation.
5. Evaluate the trained detector and inspect prediction visualizations.

## Dataset

The dataset was created from six scanned early-modern printed sources. The images were annotated to identify the main text regions and selected structural elements.

- [Training, validation, and test dataset](https://drive.google.com/drive/folders/1tvZZfsfFHPlLx26hQEDGcEnAJ6h9g0wm?usp=drive_link)
- [Training results, weights, and predictions](https://drive.google.com/drive/folders/1hjbZ72TodFKLEgVIRXIIpPbUUwb7eqom?usp=drive_link)

The dataset is small, so broader evaluation on additional documents is an important next step.

## Training configuration

| Setting | Value |
|---|---|
| Model | YOLOv8m |
| Epochs | 180 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Augmentation | Rotation, contrast adjustment, shear, perspective transforms |
| Training hardware | Google Colab Tesla T4 |

YOLOv8 was selected as a practical baseline because it provides an established object-detection workflow and can be fine-tuned on a relatively small custom dataset. Transformer-based document models remain interesting alternatives, but a fair comparison would require consistent data splits, preprocessing, and evaluation.

## Reported evaluation

The following values are from the experiment's recorded evaluation:

| Metric | Result |
|---|---:|
| Precision | 85.4% |
| Recall | 86.3% |
| F1-score | 85.8% |
| mAP@0.5 | 91.2% |
| mAP@0.5:0.95 | 75.9% |

Because the dataset contains only 52 annotated images, these results should be treated as preliminary rather than as evidence of broad out-of-domain performance.

## Visual results

**Training loss and mAP trends**

![Training loss and mAP trends](https://github.com/user-attachments/assets/82e2b6ef-e2c8-4684-9735-b0061bd7f2b2)

**Precision–recall curve**

![Precision recall curve](https://github.com/user-attachments/assets/e3e92478-d896-4f30-a716-0806564cf981)

**Example layout prediction**

![Example layout prediction](https://github.com/user-attachments/assets/7aa4ef01-18a2-46fc-96cb-8b542b2b54f2)

Additional plots and predictions are available in the linked results folder.

## Run the project

~~~bash
git clone https://github.com/AshishRaj97/Layout-Recognition-YOLOv8.git
cd Layout-Recognition-YOLOv8
python -m venv .venv
pip install ultralytics torch pandas matplotlib pillow
~~~

The repository's notebooks and training outputs contain the experiment workflow. To reproduce training, download the dataset, make sure the YOLO dataset configuration points to the correct train/validation/test directories, and use the recorded training configuration above.

For inference, use the trained weights from the results folder with the Ultralytics prediction interface. The exact weight path depends on where you place the downloaded files.

## Limitations and next steps

- The annotated dataset is small and covers only six source documents.
- Performance on unseen document collections has not yet been established.
- Some headings and author labels were misclassified in qualitative inspection.
- OCR integration and downstream text extraction have not been evaluated as part of this detector.
- A controlled comparison with a transformer-based or segmentation-based approach would make the model-selection conclusions stronger.

Planned work includes expanding the annotation set, evaluating on held-out source documents, documenting the exact inference command and environment versions, and exploring OCR as a separate downstream stage.

## License

See [LICENSE](LICENSE).

## Author

**Ashish Raj** · [GitHub](https://github.com/AshishRaj97) · [LinkedIn](https://www.linkedin.com/in/ashish-raj-327596306/)
