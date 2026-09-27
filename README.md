# Tuberculosis Detection Using Deep Learning

A deep learning-based chest X-ray classification system that detects whether an X-ray is **Normal** or shows **signs of Tuberculosis (TB)**.

The project uses transfer learning with multiple pretrained CNN architectures and selects the final model based primarily on **TB sensitivity (recall)**, since missing a TB-positive case is the most important error to monitor.

## Project Overview

Tuberculosis is a serious infectious disease that can affect the lungs. Chest X-rays can provide useful visual information for TB screening.

This project builds a machine learning pipeline that:

* Preprocesses chest X-ray images
* Splits the dataset into training, validation, and test sets
* Trains multiple transfer-learning models
* Evaluates models using clinically relevant classification metrics
* Selects a final model based on TB sensitivity and supporting metrics
* Provides a Streamlit interface for image-based prediction

> **Important:** This project is an educational/research screening system and is not a replacement for professional medical diagnosis.

## Dataset

The project uses a chest X-ray dataset containing two classes:

| Class     |    Images |
| --------- | --------: |
| Normal    |       514 |
| TB        |     2,494 |
| **Total** | **3,008** |

The original dataset is divided into:

* **Training:** 70%
* **Validation:** 15%
* **Testing:** 15%

### Final Split

| Dataset    | Normal |    TB | Total |
| ---------- | -----: | ----: | ----: |
| Train      |    359 | 1,745 | 2,104 |
| Validation |     77 |   374 |   451 |
| Test       |     78 |   375 |   453 |

The original dataset and generated `data/` directory are excluded from the GitHub repository using `.gitignore`.

## Models

Three pretrained CNN architectures were evaluated using transfer learning:

* ResNet50
* VGG16
* EfficientNetB0

The pretrained ImageNet feature-extraction layers were frozen, followed by:

* Global Average Pooling
* Dropout
* Binary classification layer with sigmoid activation

Class weights were used during training because the dataset contains substantially more TB images than Normal images.

## Model Evaluation

The primary evaluation metric is **TB Sensitivity (Recall)**.

This is important because a false negative means an X-ray containing TB signs was classified as Normal.

Secondary evaluation metrics include:

* Normal Specificity
* ROC-AUC
* Precision
* F1-score
* Confusion Matrix
* False Negatives
* False Positives

### Test Set Results

| Model          | TB Sensitivity | Normal Specificity |    ROC-AUC | False Negatives | False Positives |
| -------------- | -------------: | -----------------: | ---------: | --------------: | --------------: |
| **ResNet50**   |     **98.13%** |        **100.00%** | **0.9998** |           **7** |           **0** |
| VGG16          |         92.27% |            100.00% |     0.9893 |              29 |               0 |
| EfficientNetB0 |         93.87% |             97.44% |     0.9918 |              23 |               2 |

### Final Model

**ResNet50** was selected as the final model based on the project's evaluation criteria.

Test-set performance:

* TB Sensitivity: **98.13%**
* Normal Specificity: **100.00%**
* ROC-AUC: **0.9998**
* False Negatives: **7**
* False Positives: **0**

The final model is stored as:

```text
final_tb_model.keras
```

## Project Structure

```text
Tuberculosis_Detection/
│
├── Dataset of Tuberculosis Chest X-rays Images/
│   ├── Normal Chest X-rays/
│   └── TB Chest X-rays/
│
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── 1split_dataset.py
├── b_data_loader.py
├── c_model_resnet50.py
├── d_evaluate_resnet50.py
├── e_model_vgg16.py
├── f_evaluate_vgg16.py
├── g_model_efficientnet.py
├── h_evaluate_efficientnet.py
├── i_compare_models.py
├── e_verify_dataset.py
│
├── app.py
├── requirements.txt
│
├── resnet50_corrected.keras
├── vgg16_corrected.keras
├── efficientnetb0_corrected.keras
├── final_tb_model.keras
│
├── .gitignore
└── .gitattributes
```

The dataset directories are kept locally for training but are excluded from GitHub.

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Pillow
* Streamlit
* Git
* GitHub
* Git LFS
* AWS

## Training Pipeline

```text
Chest X-ray Dataset
        ↓
Dataset Verification
        ↓
Train / Validation / Test Split
        ↓
Image Preprocessing
        ↓
Data Loading
        ↓
Transfer Learning
        ↓
ResNet50 / VGG16 / EfficientNetB0
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Final ResNet50 Model
        ↓
Streamlit Prediction App
        ↓
AWS Deployment
```

## Streamlit Application

The application provides the following sections:

### Home

Provides an overview of the project, objective, workflow, and model performance.

### Dataset

Displays dataset statistics, class distribution, and train/validation/test split information.

### Models

Shows the performance comparison between ResNet50, VGG16, and EfficientNetB0.

### Prediction

Allows the user to upload a chest X-ray image and receive a model prediction:

* **TB**
* **Normal**

The prediction is generated using the final ResNet50 model.

### About

Contains project information, technologies, model details, and the medical-use disclaimer.

## Running the Application Locally

Clone the repository:

```bash
git clone https://github.com/rao274563-cpu/Tuberculosis_Detection.git
```

Move into the project directory:

```bash
cd Tuberculosis_Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

The application will open in the browser.

## Model Files

The trained `.keras` model files are stored using **Git LFS** because of their size.

Tracked model files:

```text
resnet50_corrected.keras
vgg16_corrected.keras
efficientnetb0_corrected.keras
final_tb_model.keras
```

If Git LFS is not installed, install it before cloning/downloading the repository.

## Prediction Logic

The final ResNet50 model produces a sigmoid output between 0 and 1.

The current classification threshold is:

```text
Score >= 0.5  →  TB
Score < 0.5   →  Normal
```

The displayed model confidence represents the model's output score and should not be interpreted as a medically validated probability.

## Limitations

This project has several limitations:

* The model is trained on a specific chest X-ray dataset.
* Dataset distribution may not represent all patient populations or imaging conditions.
* Model performance on external clinical datasets may differ.
* A machine learning prediction cannot replace clinical examination, laboratory testing, radiological interpretation, or professional medical diagnosis.

## Future Improvements

Possible future improvements include:

* External dataset validation
* Fine-tuning pretrained layers
* Threshold optimization based on validation data
* Explainable AI using Grad-CAM
* More extensive data augmentation
* Model calibration
* Clinical validation
* Cloud deployment and monitoring

## Author

**Sachin Kumar Rao**

B.Tech Computer Science Engineering

### Profiles

* LinkedIn: https://linkedin.com/in/sachin-rao-535b0b331
* GitHub: https://github.com/rao274563-cpu

## Disclaimer

This project is intended for **educational and research purposes**. It is a machine learning-based screening experiment and should not be used as a standalone medical diagnostic system.
