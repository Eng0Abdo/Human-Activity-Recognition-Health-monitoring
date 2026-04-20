# Human Activity Recognition (HAR) using Smartphones

## 📌 Project Overview
This project, developed for the Supervised Learning course at Cairo University (FCAI), implements a machine learning pipeline to classify human activities based on smartphone sensor data. The project emphasizes Intelligent Data Analysis (IDA) and the comparative performance of various supervised learning models.

## 📂 Project Structure
```text
.
├── project.ipynb           # Main Jupyter Notebook (Data loading & EDA)
├── ida.py                  # Script for signal visualization and correlation analysis
├── ml_modes.py             # Script for model training, evaluation, and metrics
├── train_features.csv      # Processed training features and labels
├── test_features.csv       # Processed testing features and labels
├── README.md               # Project documentation
└── UCI HAR Dataset/        # Original raw dataset folder
    ├── activity_labels.txt 
    ├── features.txt        
    ├── test/               # Raw test signal files
    └── train/              # Raw training signal files