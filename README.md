# Credit Card Fraud Detection Using SVM

---

## Context

This dataset contains credit card transactions featuring both legitimate and fraudulent activities from January 1st, 2019, through December 31st, 2020. It captures the transaction behaviors of 1,000 synthetic customers interacting with a pool of 800 merchants. With roughly 1.85 million transactions in total, the dataset reflects the nature of real-world financial fraud, providing features such as

- Sr_no
- trans_date_trans_time
- cc_num
- merchant
- category
- amt
- first
- last
- gender
- street
- city
- state
- zip
- lat
- long
- city_pop
- job
- dob
- trans_num
- unix_time
- merch_lat
- merch_long is_fraud

---

## Dataset

Kaggle Link:  
https://www.kaggle.com/datasets/kaushalnandania/credit-card-fraud-detection/data

---

## Problem Statement

We are solving a **Binary Classification Problem**:

- **Target Variable:** `is_fraud`
  - `0` → Normal Transaction
  - `1` → Fraudulent Transaction

---

### Dropped Features (Not Useful)

- `Sr_no` → Index only
- `cc_num` → Unique identifier
- `first`, `last` → Personal names
- `street`, `zip` → Highly specific
- `trans_num` → Unique transaction ID

---

### Selected Features

#### Numerical Features:

- `amt`
- `lat`, `long`
- `city_pop`
- `unix_time`
- `merch_lat`, `merch_long`

#### Categorical Features:

- `merchant`
- `category`
- `gender`
- `city`
- `state`
- `job`

#### Time Feature:

- `trans_date_trans_time`

---

## Feature Engineering

We extracted new features from the dataset to improve model performance:

### Time-Based Features:

- Hour of transaction
- Day
- Month
- Weekend indicator

### Distance Feature:

Calculated distance between customer and merchant:

```bash
distance = sqrt((lat - merch_lat)^2 + (long - merch_long)^2)
```

Fraud transactions often occur far from the user's location.

---

## Project Pipeline

---

### Exploratory Data Analysis (EDA)

- Fraud vs Normal distribution
- Transaction amount analysis
- Fraud by category
- Fraud by time (hour/day)
- Distance vs fraud

---

### Data Cleaning

- Checked missing values
- Handled inconsistent data
- Avoided unnecessary row dropping

---

### Encoding

- **High-cardinality** features (`merchant`, `city`, `job`): **Label Encoding** — fit on train only, applied to both train and test
- **Low-cardinality** features (`category`, `gender`, `state`): **One-Hot Encoding** via `pd.get_dummies` with `drop_first=True`
- Test set columns are aligned to train to handle any missing dummy categories

---

### Handling Imbalanced Data

The dataset is highly imbalanced (fraud cases are a small minority).

- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** on the **training set only** to prevent data leakage
- Generates synthetic fraud samples to balance class distribution

---

### Feature Scaling

- Applied **StandardScaler**
- Required for SVM performance

---

### Dimensionality Reduction

- Applied **PCA (Principal Component Analysis)**
- Reduced feature dimensionality while preserving variance

---

### Model Building (SVM)

We will use:

```bash
SVC(kernel='rbf')
```

---

### Hyperparameter Tuning

use **GridSearchCV** to optimize:

- `C`
- `gamma`

---

### Model Evaluation

Due to class imbalance, we used:

- Precision
- Recall
- F1-score
- Confusion Matrix

Accuracy alone is NOT reliable

---

## Bonus (Optional)

- Deploy model using **Streamlit**
- Allow user to input transaction data and predict fraud

---

## Team Workflow

- Youssef Elgamal: EDA + Visualization
- Asser Youssef: Data cleaning + Feature engineering
- Mohamed Hesham: Encoding + Handling Imbalanced Data
- Ahmed Gamal: Feature Scaling + Dimensionality Reduction
- Seraj Eldeen: Model + Tuning + Evaluation

---

# NOTE: DATA IS SPLIT TO TEST AND TRAIN JUST FOR YOUR KNOWLEDGE
