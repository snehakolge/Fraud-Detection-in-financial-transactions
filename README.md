# 💳 Fraud Transaction Detection Using Machine Learning

## 📌 Project Overview

This project develops a machine learning based fraud transaction detection system using supervised learning techniques The pipeline includes data preprocessing exploratory data analysis feature engineering class imbalance handling model training and evaluation

Behavior based features such as origin and destination balance differences transaction amount ratios and balance consistency indicators help detect abnormal transaction patterns Models are evaluated using Precision Recall F1 score ROC AUC score and Confusion Matrix with emphasis on recall due to dataset imbalance

This project demonstrates an interpretable and practical approach for building real world financial transaction fraud detection systems suitable for banking analytics workflows


## 🎯 Objectives

- Detect fraudulent financial transactions using machine learning models
- Perform feature engineering to capture hidden fraud behavior patterns
- Handle imbalanced datasets effectively
- Evaluate model performance using classification metrics
- Build an interpretable fraud detection pipeline


## 📂 Dataset

The dataset contains financial transaction records with features such as

- Transaction type
- Transaction amount
- Origin account balance
- Destination account balance
- Fraud label indicator

Additional behavioral features were engineered for improving fraud detection capability

Example engineered features

- Origin balance difference
- Destination balance difference
- Amount ratio
- Zero balance indicators
- Balance consistency validation features


## ⚙️ Machine Learning Workflow

### Data Preprocessing

- Handling missing values
- Encoding categorical variables
- Feature scaling if required


### Exploratory Data Analysis

- Fraud vs non fraud distribution
- Transaction type analysis
- Balance movement visualization
- Outlier detection


### Feature Engineering

Created behavioral fraud indicators such as

- Balance difference features
- Transaction amount ratio
- Zero balance anomaly flags


### Handling Class Imbalance

Applied techniques such as

- Resampling methods
- Class weighting


### Model Training

Models used in this project

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier


## 📊 Model Evaluation Metrics

Since fraud datasets are highly imbalanced evaluation focuses on

- Precision
- Recall
- F1 Score
- ROC AUC Score
- Confusion Matrix

Primary focus is given to Recall because detecting fraud transactions is critical in banking systems


## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit learn
- Jupyter Notebook


## 📁 Project Structure

fraud-transaction-detection/

│

├── data/

├── notebooks/

├── src/

├── requirements.txt

├── README.md

└── .gitignore


## 🚀 How to Run the Project

Step 1 Clone repository

git clone https://github.com/yourusername/fraud-transaction-detection.git


Step 2 Navigate to project folder

cd fraud-transaction-detection


Step 3 Install dependencies

pip install -r requirements.txt


Step 4 Run notebook

jupyter notebook


## 📈 Future Improvements

- Add Autoencoder based anomaly detection
- Deploy model using Flask or FastAPI
- Build fraud risk scoring dashboard
- Integrate real time transaction monitoring pipeline


## 👩‍💻 Author

Sneha Kolge

Machine Learning Enthusiast focused on financial fraud detection systems and applied data science projects
