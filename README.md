# Consumer Credit Worthiness Prediction
The Project used a dataset provided by the Caspian Investment firm for prediction of loan approval.
This repository contains code and documentation for predicting consumer credit worthiness using various classification models. The project encompasses a comprehensive data analysis pipeline, including hypothesis making, exploratory data analysis, data quality checking, hypothesis validation, preprocessing, feature engineering, feature selection, and model evaluation. Stratified 5-fold cross-validation is employed to make predictions for a test dataset using the following classification models:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XG Boost
5. AdaBoost
6. KNeighbors
7. Naïve Bayes
8. Support Vector Machine

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Quality Checking](#data-quality-checking)
- [Hypothesis Validation](#hypothesis-validation)
- [Preprocessing and Feature Engineering](#preprocessing-and-feature-engineering)
- [Feature Selection](#feature-selection)
- [Classification Models](#classification-models)
- [Installation](#Installation)
- [Optimizations](#Optimizations)
- [Contributing](#contributing)

## Project Overview
The goal of this project is to predict consumer credit worthiness based on various features and evaluate the performance of different classification models in making accurate predictions. The project is structured into several phases, each with specific tasks and objectives.

The project divided into four milestones:

## Milestone-01: Hypothesis Making

- **Objective**: Formulate initial hypotheses about the relationships between features and the target variable, consumer credit worthiness.

## Milestone-02: Exploratory Data Analysis

- **Objective**: Explore the dataset to gain insights into the data distribution, relationships between variables, and potential patterns. Formulate more specific hypotheses based on data exploration.

## Milestone-03: Data Quality Checking, Preprocessing, Feature Engineering

- **Objective**: Prepare the dataset for modeling. Address data quality issues, perform preprocessing tasks (handling missing values, encoding categorical variables, scaling numerical features), and engineer new features to enhance model performance.

## Milestone-04: Feature Selection, Model Development, and Validation

- **Objective**: Select the most relevant features to improve model efficiency and interpretability. Develop and validate classification models using stratified 5-fold cross-validation. Evaluate model performance using appropriate metrics.

## Dataset
The dataset used for this project is [Consumer Credit Worthiness]. It contains information about 13 features described below. You can download the dataset (https://docs.google.com/spreadsheets/d/1Tp0_0XMTuh3ZOM_DYdohZU-e9c8BRme6/edit?usp=drive_link)] or use your own dataset.

- Loan_ID: Unique Loan ID issued on every loan for a applicant, dtype: Object.<input>
- Gender: Gender of a applicant whether male or female, dtype: string.<input>
- Married: Martial status of a applicant i.e., Yes for married and NO for single, dtype: string.<input>
- Dependents: Number of individuals who are financially dependent on applicant, dtype: integer.<input>
- Education: Highest Education of applicant i.e, Bachelor, Post Graduation etc, dtype: string.<input>
- Self_employed: Whether the applicant is self employed or not i.e, Yes for self employed or else NO, dtype: string.<input>
- ApplicantIncome: Income of the applicant, dtype: integer.<input>
- CoApplicantIncome: Applicant have to put one nominee name that is called CoApplicant. So, it is column releated to coapplicant income, dtype: Integer.<input>
- Loan Amount: Amount of loan applicant wants to issue from the bank.dtype: float.<input>
- Loan_Amount_Term: The amount of time the lender gives you to repay your whole loan, dtype: float.<input>
- Credit_History: It tells about the credit done in the past by the applicant, dtype: Integer.<input>
- Property_Area: This tells about the applicant property is in which area i.e., Rural or Urban, dtype: String.<input>
- Loan_status: It is a target variable column which tells about whether the applicant application for loan approval is passed or not, dtype: String.<Output>

## Exploratory Data Analysis
In this phase, we explore the dataset to gain insights into the data distribution, relationships between variables, and potential patterns. This step helps us understand the data better and formulate initial hypotheses about the relationships between features and the target variable.

## Data Quality Checking
Data quality is essential for building reliable models. This phase involves checking for missing values, data inconsistencies, and outliers. Any issues found will be addressed to ensure the data is clean and suitable for modeling.

## Hypothesis Validation
The hypotheses formulated during exploratory data analysis will be validated through statistical tests and visualizations. This step helps us confirm or reject our initial assumptions about the data.

## Preprocessing and Feature Engineering
Data preprocessing involves preparing the data for modeling. This includes handling missing values, encoding categorical variables, scaling numerical features, and more. Feature engineering aims to create new features or transform existing ones to improve model performance.

## Feature Selection
Selecting the most relevant features is crucial for model efficiency and interpretability. Various feature selection techniques will be applied to identify the subset of features that contribute the most to the prediction task.

## Classification Models
We will train and evaluate the following classification models:
- Logistic Regression
- Decision Tree
- Random Forest
- XG Boost
- AdaBoost
- KNeighbors
- Naïve Bayes
- Support Vector Machine

Each model will be trained and evaluated using stratified 5-fold cross-validation to ensure robustness and generalization of the results.

![Screenshot 2024-05-29 224148](https://github.com/mallicksubhransu/Consumer-Credit-Worthiness-Prediction/assets/114018899/b057509e-9d14-4566-9e9b-409830023cd2)

## Installation

Install Machine Learning Libraries

```bash
pip install scikit-learn
pip install xgboost

```
Install supporting libraries
```bash
pip install numpy
pip install pandas
```
Install visualization libraries
```bash
pip install matplotlib
pip install seaborn
```

## Optimizations

## Model Optimization and Hyperparameter Tuning

To enhance the predictive performance of our classification models, we conducted the following steps:

- **Cross-Validation**: We utilized stratified 5-fold cross-validation to assess the generalization performance of our models. This approach ensures that our models are robust and capable of making accurate predictions on unseen data.

- **Hyperparameter Tuning**: We performed hyperparameter tuning for each classification model to find the optimal set of hyperparameters that maximize model performance. Techniques such as grid search or random search were employed to systematically explore the hyperparameter space.

The combination of cross-validation and hyperparameter tuning ensures that our models are well-optimized and capable of achieving the best possible predictive accuracy for consumer credit worthiness.

## Contributing
Contributions to this project are welcome. If you have ideas for improvements or new features, please open an issue or submit a pull request.

