# Movie Rating Prediction with Python

## Overview

In this project, I developed a regression model to predict IMDb movie ratings based on various movie attributes such as genre, cast, runtime, votes, and release year. The dataset contains a mix of categorical and numerical features that required preprocessing and transformation before modeling.

My approach included:

* Loading and preparing the data
* Conducting exploratory data analysis (EDA)
* Performing feature engineering
* Building and evaluating a baseline regression model

This project demonstrates how machine learning can be used to estimate public sentiment through numerical ratings.

---

## What I Did – Step-by-Step Breakdown

### 1. Data Preparation

I started by loading the dataset into a Pandas DataFrame. I inspected the structure, checked for missing values, and identified key columns relevant for predicting movie ratings.

Key steps included:

* Dropping irrelevant columns
* Renaming or formatting columns for consistency
* Reviewing null values and overall dataset shape

---

### 2. Exploratory Data Analysis (EDA)

To understand the data better, I visualized:

* The distribution of IMDb ratings
* The number of movies released per year
* The most common genres and their relation to rating
* Correlations among numerical variables

This analysis helped identify important trends and relationships. For example, I noticed that movies with more votes tended to have more stable ratings, and certain genres showed consistent popularity.

---

### 3. Top 10 Movies

As part of the EDA, I filtered and displayed the top 10 highest-rated movies based on IMDb score and the number of votes, which provided insight into what kind of content audiences favor.

---

### 4. Data Preprocessing

I handled missing values, converted categorical columns into numeric format using encoding techniques, and scaled numerical features where necessary. This ensured that the dataset was suitable for training a regression model.

Actions included:

* One-hot encoding for genres or categorical tags
* Filling or dropping null values
* Converting data types to numeric where applicable

---

### 5. Feature Engineering

I created or transformed features to help the model learn better. For example:

* Extracted release year from the date column if available
* Combined or modified genre tags into binary flags
* Created new numerical indicators from existing columns

These engineered features were based on domain intuition and helped improve the model’s ability to capture patterns.

---

### 6. Modeling

I trained a baseline regression model using the cleaned and preprocessed data. Specifically, I used **Linear Regression** as a starting point to measure how well basic features could explain IMDb scores.

---

### 7. Baseline Model - Linear Regression

Using `LinearRegression` from `sklearn`, I fit the model on the training data and evaluated it using metrics such as:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R-squared (R²)

These metrics helped quantify how close the predicted ratings were to the actual ratings. The results provided a benchmark to improve upon in future iterations (e.g., using decision trees or ensemble models).

---

## Highlights

* Explored and visualized key patterns in IMDb movie data
* Engineered features that capture important movie attributes
* Built and evaluated a baseline regression model
* Gained insights into how factors like genre, year, and vote count affect ratings



