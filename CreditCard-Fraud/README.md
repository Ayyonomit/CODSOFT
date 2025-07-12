
# 🛡️ Credit Card Fraud Detection Model

## 📌 Overview

This project builds a **machine learning model** to detect fraudulent credit card transactions using the **Kaggle credit card fraud dataset** (`creditcard.csv`). The dataset contains anonymized numerical features (V1–V28), along with `Time`, `Amount`, and a target variable `Class` (0 = genuine, 1 = fraud).

Given the **highly imbalanced nature** of the dataset (only \~0.17% fraud), we apply:

* Data preprocessing
* Class balancing with **SMOTE**
* Model training using **Random Forest Classifier**
* Performance evaluation using **precision**, **recall**, **F1-score**, **ROC AUC**, and visualizations

The code is built to run in **Google Colab**, includes **error handling**, and addresses common preprocessing issues like the `ValueError: Input y contains NaN` error during SMOTE. Each step in the code is clearly labeled and organized.

---

## 🧠 What We Did – Step-by-Step Explanation

### 1. 📂 Data Loading

* **Action**: Loaded the `creditcard.csv` dataset using `pandas.read_csv()` after uploading it to Colab.
* **Purpose**: Import the dataset (284,807 transactions with 30 features).
* **Outcome**: Data successfully loaded into a DataFrame for analysis.

---

### 2. 🔍 Data Exploration

* **Actions**:

  * Viewed shape, info, and summary with `.shape`, `.info()`, `.describe()`
  * Checked for missing values: `df.isnull().sum()`
  * Analyzed class distribution using `value_counts()`
  * Visualized:

    * Class imbalance (bar plot)
    * Correlation heatmap
    * Distribution of `Amount` and `Time` (histograms)

* **Purpose**: Understand structure, detect missing values, assess imbalance, explore feature relationships.

* **Outcome**: Found strong class imbalance, potential `NaN` risks, and skewed feature distributions.

---

### 3. 🧹 Data Preprocessing

#### 3.1. ✅ Handle Missing Values

* **Action**: Dropped rows with missing values using `df.dropna()`.
* **Purpose**: Resolve SMOTE error (`ValueError: Input y contains NaN`) and ensure model compatibility.
* **Outcome**: All `NaN` values removed; dataset clean and SMOTE-ready.

#### 3.2. 🔀 Separate Features and Target

* **Action**: Split data into features (`X`) and target (`y`), cast `y` to `int`.
* **Purpose**: Prepare input/output for model training and SMOTE.
* **Outcome**: `X` and `y` are properly formatted.

#### 3.3. 📏 Normalize Features

* **Action**: Used `StandardScaler` to scale all features (including `Time`, `Amount`).
* **Purpose**: Prevent bias from differing feature scales.
* **Outcome**: Standardized feature set improves model performance.

#### 3.4. ⚖️ Handle Class Imbalance with SMOTE

* **Action**:

  * Applied SMOTE from `imblearn`
  * Handled potential errors using `try-except` and checked for `NaN`/`inf`
  * Visualized new class distribution

* **Purpose**: Address severe imbalance by oversampling minority class.

* **Outcome**: Balanced dataset with equal fraud and genuine transactions.

---

### 4. ✂️ Train-Test Split

* **Action**: Used `train_test_split` (80/20 split, with stratification).
* **Purpose**: Create separate sets for training and evaluation.
* **Outcome**: `X_train`, `X_test`, `y_train`, `y_test` created.

---

### 5. 🧠 Model Training

* **Action**: Trained a **Random Forest Classifier** with `n_estimators=100`.
* **Purpose**: Build a robust model capable of learning fraud patterns.
* **Outcome**: Model trained and ready for evaluation.

---

### 6. 📊 Model Prediction

* **Action**: Generated predictions and probabilities using the trained model.
* **Purpose**: Classify new transactions and evaluate model performance.
* **Outcome**: Model outputs available for evaluation.

---

### 7. 📈 Model Evaluation

* **Actions**:

  * Confusion matrix (numeric + heatmap)
  * Classification report (precision, recall, F1-score)
  * ROC AUC score
  * Plots: ROC curve, Precision-Recall curve

* **Purpose**: Evaluate model’s fraud detection performance using key metrics.

* **Outcome**: Strong performance with ROC AUC > 0.9, insightful visualizations.

---

### 8. 🌟 Feature Importance

* **Action**: Extracted and visualized top 10 most important features.
* **Purpose**: Understand which features influence fraud detection most.
* **Outcome**: Bar plot of feature importances (e.g., `V14`, `Amount` often highly ranked).

---

### 9. 💾 Model Saving

* **Action**: Saved the trained model as `enhanced_fraud_detection_model.pkl` using `joblib`.
* **Purpose**: Store model for reuse or deployment.
* **Outcome**: Model file saved for future predictions.

---

## 🛠️ Addressing the NaN Error During SMOTE

* **Problem**: `ValueError: Input y contains NaN` when applying SMOTE.
* **Solution**:

  * Checked `df.isnull().sum()`
  * Dropped missing rows (`df.dropna()`)
  * Verified no `NaN` remains
  * Added error handling during SMOTE to check for `NaN`/`inf` values

---

## 💡 Key Features of This Project

* ✅ **Comprehensive Preprocessing**: Scaling, cleaning, balancing data
* ⚠️ **Error Handling**: Catches and resolves common preprocessing issues
* 📊 **Data Visualization**: Insightful plots at every step
* 🌲 **Robust Model**: Random Forest performs well on balanced datasets
* 📂 **Organized Workflow**: Step-wise code with bolded headings and explanations

---

Let me know if you want this in plain `.md` format or need help uploading this into your GitHub `README.md` file!
