# Titanic Survival Prediction

## Overview
This repository contains a Jupyter Notebook demonstrating how to build a machine learning model to predict passenger survival on the Titanic using the Titanic dataset from Seaborn. The project focuses on classification techniques with scikit-learn, including:

- Data preprocessing
- Model pipelines
- Hyperparameter tuning
- Model comparison

Key models explored include **K-Nearest Neighbors (KNN)**, **Random Forest**, and **Logistic Regression**.

The notebook covers exploratory data analysis (EDA), feature engineering, model training, evaluation, and interpretation of feature importances. It highlights the use of pipelines for streamlined workflows and cross-validation for robust performance assessment.

---

## Objectives

- Use scikit-learn to build a model to solve a classification problem.  
- Implement a pipeline to combine preprocessing steps with a machine learning model.  
- Interpret the results of the model.  
- Update the pipeline with a different machine learning model.  
- Compare the performances of the classifiers.  

---

## Dataset
The Titanic dataset is loaded via Seaborn and includes features such as:

- `survived`: Target variable (0 = No, 1 = Yes)  
- `pclass`: Ticket class  
- `sex`: Gender  
- `age`: Age in years  
- `sibsp`: Number of siblings/spouses aboard  
- `parch`: Number of parents/children aboard  
- `fare`: Passenger fare  
- `embarked`: Port of embarkation  

Additional categorical features include `class`, `who`, `adult_male`, etc.  

The dataset has **891 rows** and is preprocessed to handle missing values, encode categorical variables, and scale numerical features.

---

## Usage

Clone the repository:

```bash
git clone https://github.com/LUTHFI007/Titanic_Survival_Prediction.git
```
## Key Sections in the Notebook

## Data Loading and EDA
 - Load the dataset and perform basic visualizations (e.g., survival rates by class, sex, age)

## Preprocessing Pipeline
 - Use ColumnTransformer for handling:
   - Numerical features: imputation, scaling
   - Categorical features: one-hot encoding

## Model Training
 - KNN with PCA and GridSearchCV
 - Random Forest with feature importance visualization
 - Logistic Regression with coefficient analysis

## Evaluation
 - Cross-validation scores, confusion matrices, classification reports, and test accuracy (~82%)

## Model Comparison

 - Compares KNN, Random Forest, and Logistic Regression performances
___ 
## Results

## Best Models:
 - Random Forest: Achieves ~82% test accuracy; key features include who_man, pclass, fare
 - Logistic Regression: Similar accuracy; emphasizes features like sex_male, pclass_3

## Visualizations:
 - Confusion matrices
 - Feature importances
 - Coefficient magnitudes

# Insights:
Models perform similarly, but feature importances differ, suggesting potential multicollinearity (e.g., between who_man and sex).

```bash
pip install numpy pandas matplotlib scikit-learn seaborn
