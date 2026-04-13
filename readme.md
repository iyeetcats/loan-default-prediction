# Loan Default Prediction

A machine learning project to predict loan default risk using LightGBM, built as part of a data science portfolio targeting fintech roles.

## Business Problem

Financial institutions need to assess the risk of loan applicants defaulting on their payments. Missing a true defaulter (false negative) results in direct financial loss, while incorrectly flagging good borrowers (false positive) reduces revenue. This project builds a binary classifier to predict default probability for each applicant.

## Dataset

- **Source:** Loan Default Dataset (255,347 rows, 18 columns)
- **Target:** `Default` (0 = No Default, 1 = Default)
- **Class distribution:** 88% non-default, 12% default (imbalanced)

**Features include:**
- Demographic: Age, MaritalStatus, Education, HasDependents
- Financial: Income, LoanAmount, CreditScore, DTIRatio, InterestRate
- Employment: EmploymentType, MonthsEmployed
- Loan details: LoanTerm, LoanPurpose, NumCreditLines, HasMortgage, HasCoSigner

## Project Structure

```
loan-default-prediction/
├── data/
│   └── Loan_default.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
└── README.md
```

## Methodology

### 1. Exploratory Data Analysis
- Identified class imbalance (88/12 split)
- Found dataset is likely synthetic — most features show uniform distributions except Age
- Key finding from boxplots: defaulters tend to be **younger**, have **lower income**, and are charged **higher interest rates**
- Categorical analysis: **Unemployed** borrowers have the highest default rate (~13.5%); having a **co-signer** reduces default risk

### 2. Preprocessing
- Dropped `LoanID` (identifier, no predictive value)
- Label encoded all categorical features
- 80/20 stratified train/test split to preserve class ratio

### 3. Modelling
- **Algorithm:** LightGBM Classifier
- **Class imbalance handling:** `class_weight='balanced'`
- **Hyperparameters:** 500 estimators, learning rate 0.05

## Results

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.75 |
| Recall (Default) | 0.66 |
| Precision (Default) | 0.23 |
| Accuracy | 0.70 |

**Confusion Matrix:**
- True Positives (correctly predicted defaults): 3,924
- False Negatives (missed defaults): 2,007
- False Positives (good borrowers flagged): 13,112

### Top Predictors of Default Risk
1. Interest Rate
2. Loan Amount
3. Income
4. Credit Score
5. Months Employed
6. Age

## Key Insights

- **Interest rate is the strongest predictor** — higher rates strongly correlate with default
- **Age and income matter more than education** — education level shows almost no difference in default rates across categories
- **Co-signers reduce risk** — applicants with co-signers default less frequently
- **Recall is prioritised over precision** — in lending, missing a real defaulter is more costly than a false alarm

## Tools & Libraries

- Python 3.14
- pandas, matplotlib, seaborn
- scikit-learn
- LightGBM

## How to Run

```bash
pip install pandas matplotlib seaborn scikit-learn lightgbm
```

Open notebooks in order:
1. `01_eda.ipynb` — Exploratory Data Analysis
2. `02_preprocessing.ipynb` — Data Preprocessing
3. `03_modeling.ipynb` — Model Training & Evaluation
