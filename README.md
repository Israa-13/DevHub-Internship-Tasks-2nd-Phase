# DevHub-Internship-Tasks-2nd-Phase
DevelopersHub Corporation AI/ML Internship Tasks.

# Task 1: News Topic Classifier Using BERT

### **Objective of the Task**
- To fine-tune a pretrained BERT transformer model to classify news headlines into topic categories using the AG News dataset.

### **Methodology / Approach**
- Loaded and preprocessed the AG News dataset using Hugging Face Datasets.
- Tokenized text using bert-base-uncased tokenizer.
- Fine-tuned BERT using Hugging Face Transformers Trainer API.
- Evaluated the model using accuracy and weighted F1-score.
- Deployed the model using Gradio for real-time predictions in Google Colab.

### **Key Results or Observations**
- Achieved high accuracy and F1-score (~93%).
- Transformer-based models outperform traditional NLP approaches.
- Gradio deployment enabled interactive and shareable inference.


# Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API

### **Objective of the Task**
- To build an end-to-end machine learning pipeline for predicting customer churn using Scikit-learn’s Pipeline API.

### **Methodology / Approach**
-Loaded and cleaned the Telco Customer Churn dataset.
- Applied preprocessing using ColumnTransformer for numerical and categorical features.
- Built reusable pipelines combining preprocessing and model training.
- Tuned hyperparameters using GridSearchCV.
- Exported the final trained pipeline using joblib.

### **Key Results or Observations**
- Pipeline architecture prevents data leakage and improves reproducibility.
- Random Forest achieved better F1-score compared to Logistic Regression.
- The exported pipeline enables seamless deployment and inference.


# Task 3: Multimodal ML – Housing Price Prediction Using Images + Tabular
Data

