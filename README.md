#  MOVIE RATING PREDICTION WITH PYTHON


# Project Overview

This project focuses on predicting IMDb ratings of Indian movies using a dataset that includes features like movie name, year, duration, genre, director, votes, and main actors. It involves data preprocessing, exploratory data analysis, feature engineering, and regression model training.

# Objectives

Clean and preprocess raw movie data

Perform exploratory analysis to extract trends and patterns

Build regression models to predict IMDb ratings

Evaluate and compare model performance

# Data Understanding
The dataset includes the following columns:

| Column     | Description                   |
| ---------- | ----------------------------- |
| `Name`     | Movie title                   |
| `Year`     | Release year                  |
| `Duration` | Runtime in minutes            |
| `Genre`    | Genre(s) of the movie         |
| `Rating`   | IMDb rating (target variable) |
| `Votes`    | Number of IMDb votes          |
| `Director` | Name of the director          |
| `Actor 1`  | Lead actor                    |
| `Actor 2`  | Supporting actor              |
| `Actor 3`  | Additional actor              |

# Exploratory Data Analysis (EDA)
The notebook includes visualizations such as:

Top 10 movies based on rating

Top 10 directors by number of movies directed

Top 10 actors by number of appearances

Distribution of IMDb ratings

Distribution of number of votes

# Modeling
The following regression models were trained to predict IMDb ratings:

Linear Regression


