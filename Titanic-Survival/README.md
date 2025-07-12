
# Titanic Survival Prediction Model

## Overview

In this project, I build a machine learning model to predict whether a passenger on the Titanic would survive or not, based on the Titanic dataset. The dataset contains features such as age, sex, passenger class, fare, and number of family members aboard. The target variable is `Survived`, where 1 means the passenger survived and 0 means they did not.

The goal was to clean the data, analyze important patterns, engineer relevant features, and train a model—specifically a Random Forest Classifier—to make accurate predictions. I also evaluated the model using classification metrics such as accuracy, precision, recall, and F1-score.

---

## What I Did – Step-by-Step

### 1. Importing Libraries

I started by importing all the necessary libraries including `pandas`, `numpy`, `matplotlib`, `seaborn`, and modules from `scikit-learn` for model building and evaluation.

---

### 2. Loading and Previewing the Data

I loaded the Titanic dataset from `Titanic-Dataset.csv` and reviewed the first few rows using `head()` and `info()`. This helped me understand the structure of the data, types of features, and identify missing values.

---

### 3. Data Preprocessing

* I filled missing values in the `Age` column with the median and filled missing `Embarked` values with the mode.
* I dropped irrelevant columns such as `Cabin`, `Ticket`, and `Name`.
* I used label encoding to convert categorical features like `Sex` and `Embarked` into numeric form.

The result was a clean and numerical dataset ready for analysis and modeling.

---

### 4. Exploratory Data Analysis (EDA)

I explored the relationships between various features and the survival outcome using visualizations. This included:

* Count plots of survival rates
* Survival comparison by sex and class
* Histograms for age and fare
* A heatmap to understand correlations between features

From this analysis, I observed that women and first-class passengers were more likely to survive. Age also played a role, with children having higher survival rates.

---

### 5. Feature Engineering

I added new features such as:

* `FamilySize`: the sum of `SibSp` and `Parch`
* `IsAlone`: a binary feature indicating whether a passenger was traveling alone

These new features helped capture important patterns not directly present in the original columns.

---

### 6. Model Preparation

I split the dataset into training and testing sets using an 80-20 split. I also scaled numerical features like `Age` and `Fare` using `StandardScaler` to normalize their values.

---

### 7. Model Training

I trained a `RandomForestClassifier` using the training data. This ensemble method was chosen for its ability to handle feature importance and generalize well on unseen data.

---

### 8. Model Evaluation

I evaluated the model using:

* Accuracy score
* Confusion matrix
* Classification report (including precision, recall, and F1-score)

The model showed good performance in classifying whether a passenger survived or not. Most importantly, it was able to correctly predict a balanced number of survival and non-survival cases.

---

### 9. Additional Insights

After training, I analyzed feature importances from the Random Forest model. I found that `Sex`, `Pclass`, `Fare`, and `Age` were the most influential features in predicting survival.

---

## Highlights

* Cleaned and preprocessed the Titanic dataset with care
* Performed detailed exploratory data analysis to extract useful patterns
* Created meaningful new features to improve model performance
* Trained and evaluated a Random Forest classifier with strong results
* Gained insights into which features most affected survival


