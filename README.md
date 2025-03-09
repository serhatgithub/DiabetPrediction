# Diabetes Dataset Machine Learning Example

This repository contains a simple Machine Learning example using the Diabetes dataset, designed for beginners in Machine Learning (ML), Artificial Intelligence (AI), or Data Science. The project demonstrates basic data analysis, preprocessing, model training, evaluation, and prediction using Python and popular libraries like Pandas, Scikit-learn, and Matplotlib.

## Overview

The goal of this project is to predict whether a patient has diabetes based on features like glucose levels, blood pressure, BMI, and more. It uses a Decision Tree Classifier, optimizes it with GridSearchCV, and evaluates its performance with metrics like accuracy, confusion matrix, and classification report.

### What You'll Learn
- Loading and exploring a dataset with Pandas.
- Visualizing data distributions with Seaborn and Matplotlib.
- Preprocessing data (scaling features).
- Training and evaluating ML models.
- Hyperparameter tuning with GridSearchCV.
- Making predictions on new data.

## Dataset

The dataset used is the **Pima Indians Diabetes Dataset**, which contains 768 records and 9 features:
- **Pregnancies**: Number of times pregnant.
- **Glucose**: Plasma glucose concentration.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skin fold thickness (mm).
- **Insulin**: 2-hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg/(height in m)^2).
- **DiabetesPedigreeFunction**: Diabetes pedigree function.
- **Age**: Age (years).
- **Outcome**: Class variable (0 = no diabetes, 1 = diabetes).

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) or [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes).

## Requirements

To run the notebook, you'll need the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install them using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
