# Credit-Card-Fraud-Detection-Docker

[Web Deployment of Credit Card Fraud Detection Model](https://github.com/user-attachments/assets/b5496b39-a03f-4931-9182-3ed3d560e535)

# Credit Card Fraud Detection

## Project Overview

This project aims to predict whether a given credit card transaction is fraudulent using machine learning classification algorithms. The dataset contains transactions made by credit cards in September 2013 by European cardholders, with 492 fraud cases out of 284,807 transactions, resulting in a highly unbalanced dataset where the positive class (frauds) accounts for only 0.172% of all transactions.

## Dataset Description

The dataset includes:

- **Numerical Features**: The input variables are the result of PCA transformations, including features V1, V2, …, V28. Due to confidentiality, original features are not provided.
- **Time**: The seconds elapsed since the first transaction in the dataset.
- **Amount**: The transaction amount, which can be used for example-dependent cost-sensitive learning.
- **Class**: The response variable, taking a value of 1 in case of fraud and 0 otherwise.

## Project Pipeline
```
.

├── Plots/                                      : Contains all visualizations and plots
├── Credit_Fraud_Analysis.ipynb                 : Contains EDA, data preprocessing, and model development (including K-Means clustering)
├── Flask_API.ipynb                             : Implementation of the Flask API
├── Frontend/                                   : Contains HTML and CSS files for the UI
│   ├── index.html                              : Main HTML file for user interaction
│   └── result.html                             : HTML file to display prediction results
├── Dockerfile                                   : Docker configuration file for containerization
├── main.yml                                    : GitHub Actions configuration for CI/CD
├── render.yml                                  : Render deployment configuration
├── requirements.txt                            : Python package dependencies
├── LICENSE                                      : License
└── README.md                                    : Project documentation

```
<br/>


### Class Imbalance

Given the class imbalance ratio, measuring the model's accuracy using the Area Under the Precision-Recall Curve (AUPRC) is recommended. Traditional confusion matrix accuracy is not meaningful in unbalanced classification contexts.

## Key Features

- **Fraud Prediction**: Utilizes multiple machine learning algorithms to identify fraudulent transactions.
- **K-Means Clustering**: Employed to predict fraud for new customer transactions based on existing data.
- **Flask API**: A RESTful API for interaction with the model.
- **User Interface**: Frontend built using HTML and CSS for user-friendly interaction.
- **Data Processing**: Includes EDA, preprocessing, and handling of imbalanced data using SMOTE.
- **Model Evaluation**: Selected models based on recall and ROC score.
- **CI/CD Pipeline**: Implemented using GitHub Actions for continuous integration and deployment with Docker.
- **Deployment**: Deployed on Render for easy accessibility.

## Algorithms Used

- **Isolation Forest**
- **Local Outlier Factor**
- **Random Forest**
- **CatBoost**
- **AdaBoost**
- **XGBoost**

## Project Workflow

1. **Data Collection**: Gathered and explored the credit card transactions dataset.
2. **Exploratory Data Analysis (EDA)**: Analyzed transaction patterns and visualized data distributions.
3. **Data Preprocessing**: Cleaned the dataset, applied PCA for dimensionality reduction, and handled class imbalance using SMOTE.
4. **Model Development**: Developed and evaluated multiple machine learning models for fraud detection.
5. **K-Means Clustering**: Applied clustering to predict fraud on new customer transactions.
6. **API Development**: Created a Flask API to expose endpoints for transaction predictions.
7. **Frontend Development**: Built a user-friendly interface using HTML and CSS.
8. **Containerization**: Dockerized the application for consistent deployment.
9. **CI/CD Pipeline**: Set up GitHub Actions to automate testing and deployment.
10. **Deployment**: Deployed the application on Render for public access.





