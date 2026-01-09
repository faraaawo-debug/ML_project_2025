# Clash Royale Match Outcome Prediction

This project focuses on predicting the outcome of a **1v1 Clash Royale match** using machine learning.  
The objective is to determine whether **Player 1 wins or loses** based solely on the **deck composition** of both players.

The work was conducted as part of a **machine learning course project** and explores the limitations of prediction when only partial game information is available.

---

##  Project Objective

The goal is to evaluate how much information **deck composition alone** provides for predicting match outcomes, and to assess the impact of additional card-level statistics on model performance.

The project is divided into two phases:
- **Phase 1:** prediction using only deck composition
- **Phase 2:** prediction using deck composition enriched with card statistics

---

##  Dataset

Two datasets from **Kaggle** are used.

### 1️⃣ Match Dataset
- 2,311 recorded matches
- For each match:
  - 8 cards for Player 1
  - 8 cards for Player 2
  - number of crowns for each player

A binary target variable is created:
- `win_p1 = 1` if Player 1 wins
- `win_p1 = 0` otherwise

!! The dataset is **highly imbalanced**:
- approximately **72% of matches** are won by Player 1

---

### 2️ Card Statistics Dataset
Provides global statistics for each card:
- elixir cost
- rarity
- win rate
- usage rate

---

##  Data Preprocessing

- One-hot encoding applied to the **16 card slots**
- Final feature space of approximately **1,300 binary features**
- Train-test split:
  - 80% training
  - 20% testing
  - **stratified** to preserve class distribution

For Phase 2:
- missing card statistics are filled using **column-wise mean imputation**

---

##  Phase 1 — Baseline Models

The following baseline models are trained:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)

### Evaluation metrics
- Accuracy
- Precision
- Recall (for the losing class)
- F1-score
- ROC-AUC
- Confusion matrices

### Observations
- all models are strongly affected by class imbalance
- Logistic Regression provides the most balanced behavior
- KNN mostly predicts the majority class

---

##  Phase 2 — Advanced Models

To improve performance, more expressive models are trained:
- Random Forest
- XGBoost
- CatBoost

### Additional engineered features
- average elixir cost
- average win rate
- average usage rate
- average rarity

---

##  Results

- Baseline accuracy is already high (~72%) due to class imbalance
- Advanced models slightly improve overall discrimination
- **CatBoost achieves the best overall performance**
- Recall for Player 1 losses remains low across all models

These results highlight that **deck composition alone is insufficient** to accurately predict match outcomes without real in-game context (e.g. player skill, timing, placements).

---

##  Evaluation Summary

All models are evaluated using:
- Accuracy
- Precision
- Recall (minority class)
- F1-score
- ROC-AUC
- Confusion matrices

This evaluation framework enables a detailed analysis of model behavior under strong class imbalance.

---

##  Libraries Used

- pandas  
- numpy  
- scikit-learn  
- xgboost  
- catboost  
- matplotlib  
- seaborn  

---

##  References

- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.*
- Dorogush, A. V., Ershov, V., & Gulinostrov, A. (2018). *CatBoost: Gradient boosting with categorical features support.*
- He, H., & Garcia, E. A. (2009). *Learning from imbalanced data.*

---

##  Authors

- Faraa Awoyemi  
- Lilia Benabdallah  
- Ilyes Ben Younes  
- Lea Hadj-Said
