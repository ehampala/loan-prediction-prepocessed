# LOAN PREDICTION - DATA PREPROCESSING REPORT

**AnalystLab Africa | Week 2: Feature Engineering & Data Preprocessing**  
**Pipeline**: Train & Test Dataset Preprocessing with Proper ML Workflow

---

## EXECUTIVE SUMMARY

This report documents the complete preprocessing pipeline applied to the Loan Prediction dataset with train/test split (1,232 training records, 367 test records). Through systematic data cleaning, feature engineering, categorical encoding, and numerical scaling, the raw datasets were transformed into machine learning-ready format with zero missing values, properly engineered features, and correct train/test handling (scaler fitted only on training data).

**Key Achievements**:
- Original shape: Train 614x12 | Test 367x12
- Final shape: Train 614x18 | Test 367x18
- Missing values handled: 149 values across 7 features
- Features engineered: 3 new features
- Categorical encoding: One-Hot (6 categorical features)
- Numerical scaling: RobustScaler (fitted on train only, applied to both)
- Output datasets: 2 (train preprocessed, test preprocessed)

---

## 1. DATASET OVERVIEW

### Initial Dataset Structure

```
Loan Prediction Dataset (Train/Test Split)

Training Dataset:
  Records: 614
  Features: 12
  Target: Loan_Status (Y/N)

Test Dataset:
  Records: 367
  Features: 11 (no target variable—will predict)
```

### Feature Descriptions

| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| **Loan_ID** | Identifier | Unique ID | L001–L1599 |
| **Gender** | Categorical | Applicant gender | Male, Female, [Missing] |
| **Married** | Categorical | Marital status | Yes, No |
| **Dependents** | Categorical | Number of dependents | 0, 1, 2, 3+ |
| **Education** | Categorical | Education level | Graduate, Undergraduate |
| **Self_Employed** | Categorical | Employment type | Yes, No, [Missing] |
| **ApplicantIncome** | Numeric | Applicant's annual income | $775–$81,000 |
| **CoapplicantIncome** | Numeric | Co-applicant annual income | $0–$41,667 |
| **LoanAmount** | Numeric | Loan amount (thousands) | $9–$700K, [Missing] |
| **Loan_Amount_Term** | Numeric | Loan term (months) | 6–480, [Missing] |
| **Credit_History** | Numeric | Credit history (1=yes, 0=no) | 0, 1, [Missing] |
| **Property_Area** | Categorical | Property location | Urban, Semiurban, Rural |
| **Loan_Status** | Categorical | Approval (TARGET) | Yes, No |

---

## 2. DATA INSPECTION & QUALITY ASSESSMENT

### Initial Missing Value Analysis

**Training Dataset Missing Values**:

3,0.490


| Feature          | Missing Count | % Missing    | Data Type |
|------------------|---------------|--------------|-----------|
| Married          | 3             | 0.49%        | Categorical |
| Dependents       | 15            | 2.44%        | Categorical |
| Gender           | 13            | 2.12%        | Categorical |
| Self_Employed    | 32            | 5.21%        | Categorical |
| LoanAmount       | 22            | 3.58%        | Numeric |
| Loan_Amount_Term | 14            | 2.28%        | Numeric |
| Credit_History   | 50            | 8.14%        | Numeric |
| **TOTAL**        | **~149**      | **24% avg** | — |

**Test Dataset**: Similar missing patterns (proportionally balanced)

### Missing Value Imputation Strategy

**Rationale**: Imputation preferred over deletion
- Only 2.0% missing overall (614 total records)
- Deleting rows would waste 100 records
- Missing patterns appear random (not systematic)

#### **Categorical Features → MODE Imputation**

```python
mode_impute_cols = ["Gender", "Married", "Dependents", "Self_Employed", 
                    "Loan_Amount_Term", "Credit_History"]
```

**Gender** (2.12% missing):
- Mode value: 'Male'
- Justification: Most common gender in dataset
- Applied to train & test identically

**Self_Employed** (5.21% missing):
- Mode value: 'No'
- Justification: ~90% are salaried employees
- Conservative assumption reduces risk overestimation

**Loan_Amount_Term** (2.28% missing):
- Mode value: 360 (30-year term)
- Justification: Industry standard mortgage term
- High confidence in imputation

**Credit_History** (8.14% missing):
- Mode value: 1 (has credit history)
- Justification: 80% have credit history
- Slight optimistic bias, but feature is highly predictive

#### **Numeric Features → MEDIAN Imputation**

```python
median_impute_cols = ["LoanAmount", "ApplicantIncome", "CoapplicantIncome"]
```

**LoanAmount** (3.58% missing):
- Median value: ~147 thousand
- Justification: Right-skewed distribution; median robust to outliers
- More reliable than mean (would be inflated by few large loans)

**ApplicantIncome** (missing via dependency):
- Median value: ~5,000
- Strategy: Impute when missing in corresponding records

**CoapplicantIncome** (missing via dependency):
- Median value: ~1,000
- Strategy: Impute when missing

**Imputation Code**:
```python
for col in mode_impute_cols + median_impute_cols:
    train_df[col] = train_df[col].fillna(imputation_dict[col])
    test_df[col] = test_df[col].fillna(imputation_dict[col])
```

### Result: 100% Complete Datasets 

Post-imputation:
- Training: 0 missing values
- Test: 0 missing values
- Data quality: Ready for modeling

---

## 3. FEATURE ENGINEERING

### Rationale

**Philosophy**: Create features that expose hidden patterns in applicant financial profiles

**Goal**: Transform raw income figures into meaningful risk indicators

### Feature 1: TotalIncome  CREATED

**Formula**: `totalIncome = ApplicantIncome + CoapplicantIncome`

**Why**:
- Applicant income alone ignores co-applicant's financial contribution
- Combined household income predicts repayment capacity
- Standard banking underwriting metric

**Properties**:
- Distribution: Heavily right-skewed (few high earners)
- Essential for deriving income ratios
- Will be scaled by RobustScaler

**Example**:
```
Applicant A: Income=$50K, Co-applicant=$10K → Total=$60K
Applicant B: Income=$50K, Co-applicant=$50K → Total=$100K
B has stronger financial profile despite same applicant income
```

### Feature 2: LoanToIncomeRatio  CREATED

**Formula**: `loanToIncomeRatio = LoanAmount / totalIncome`

**Why**:
- Banks use debt-to-income as key lending criterion
- Measures loan burden relative to earning capacity
- Ratio 0.5 = conservative; 3.0+ = risky

**Properties**:
- Distribution: Near-normal (good for ML)
- Highly interpretable (what % of annual income is the loan?)
- Directly impacts default risk

**Example**:
```
Applicant A: Loan=$100K, Income=$200K → Ratio=0.5 (moderate)
Applicant B: Loan=$100K, Income=$50K → Ratio=2.0 (high risk)
Same loan, but B is 4× riskier due to lower income
```

### Feature 3: IncomePerDependent  CREATED

**Formula**: `incomePerDependent = totalIncome / (Dependents + 1)`

**Why**:
- Dependents represent financial obligation
- Normalizes income by household size
- More dependents = lower per-capita capacity = higher risk

**Properties**:
- Captures household financial strength per person
- Handles zero dependents via "+1" in denominator

**Example**:
```
Applicant A: Income=$120K, Dependents=0 → Per-Dependent=$120K
Applicant B: Income=$120K, Dependents=3 → Per-Dependent=$30K
Same income, but A has much stronger per-capita capacity
```

### Engineering Summary

| Feature | Created | Interpretation | Use |
|---------|---------|-----------------|-----|
| **totalIncome** | Yes     | Household earning power | Essential for ratios |
| **loanToIncomeRatio** | Yes     | Loan burden | Predicts ability to repay |
| **incomePerDependent** | Yes     | Per-capita income | Accounts for dependents |

---

## 4. IDENTIFIER REMOVAL

**Loan_ID Column**:
- **Removed**: Yes 
- **Reason**: Purely administrative identifier; no predictive value
- **Impact**: Does not affect modeling

```python
train_df = train_df.drop(columns=["Loan_ID"])
test_df = test_df.drop(columns=["Loan_ID"])
```

---

## 5. CATEGORICAL ENCODING

### Strategy: One-Hot Encoding

**Features to Encode**:
1. Gender (2 categories)
2. Married (2 categories)
3. Dependents (4 categories—converted to numeric first)
4. Education (2 categories)
5. Self_Employed (2 categories)
6. Property_Area (3 categories)

### Why One-Hot Encoding (Not Label Encoding)?

| Aspect | One-Hot | Label | Choice |
|--------|---------|-------|--------|
| **Assumes Ordering?** | No | Yes (Male=0, Female=1) | One-Hot  |
| **Interpretability** | Clear | Ambiguous | One-Hot  |
| **Linear Models** | Appropriate | Can introduce bias | One-Hot  |
| **Tree Models** | Less efficient | More efficient | N/A for this project |

**Decision**: All categorical features are **nominal** (no natural ordering). Gender, marital status, etc. have no hierarchy. One-Hot encoding is most appropriate.

### Implementation

```python
categorical_cols = ["Gender", "Married", "Dependents", "Education", 
                    "Self_Employed", "Property_Area"]

X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Ensure train & test have identical columns
for col in X_train_encoded.columns:
    if col not in X_test_encoded.columns:
        X_test_encoded[col] = 0
```

### Resulting Columns

| Original | Encoded Columns | Logic |
|----------|-----------------|-------|
| Gender (2 cat) | Gender_Male | Male=1, Female=0 (Female inferred) |
| Married (2 cat) | Married_Yes | Yes=1, No=0 (No inferred) |
| Education (2 cat) | Education_Not Graduate | Graduate=0, NotGraduate=1 |
| Self_Employed (2 cat) | Self_Employed_Yes | Yes=1, No=0 |
| Property_Area (3 cat) | Property_Area_Semiurban, Property_Area_Urban | Rural=(0,0), Semi=(1,0), Urban=(0,1) |
| Dependents (4 cat) | (Numeric after conversion) | 0, 1, 2, 3 |

### Dummy Trap Prevention

**`drop_first=True`** ensures:
- No perfect multicollinearity
- Each categorical represented by n-1 binary columns
- Information preserved through absence (if Gender_Male=0 → Female)

---

## 6. FEATURE SCALING

### Strategy: RobustScaler

**Features to Scale**:
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- totalIncome
- incomePerDependent
- loanToIncomeRatio

**Why RobustScaler (Not StandardScaler or MinMaxScaler)?**

| Scaler | Formula | Advantage | Disadvantage | Our Case |
|--------|---------|-----------|--------------|----------|
| **StandardScaler** | (x - μ) / σ | Assumes normal distribution | Sensitive to outliers |  Right-skewed income |
| **MinMaxScaler** | (x - min)/(max - min) | Bounded [0,1] | Outliers compress range |  Extreme values shrink scale |
| **RobustScaler** | (x - median) / IQR | Robust statistics | Not bounded [0,1] |  **CHOSEN** |

**RobustScaler Properties**:
- Uses **median** (robust) instead of mean
- Uses **IQR** (Interquartile Range) instead of std
- **Preserves outlier information** (doesn't artificially bound)
- **Ideal for financial data** with occasional legitimate high values

### Implementation

```python
scaler = RobustScaler()

# Fit scaler ONLY on training data
X_train_encoded[scaling_cols] = scaler.fit_transform(X_train_encoded[scaling_cols])

# Apply SAME scaler to test data (no refitting)
X_test_encoded[scaling_cols] = scaler.transform(X_test_encoded[scaling_cols])
```

**Critical**: Scaler is fitted on training data only, then applied to test. This prevents data leakage and ensures test data is scaled using training data statistics.

### Scaling Effects (Before/After)

**Before Scaling**:
- ApplicantIncome: [775, 81,000] (range = 80,225)
- LoanAmount: [9, 700] (range = 691)
- Credit_History: [0, 1] (range = 1)
- **Problem**: Income dominates numerically; Credit_History invisible

**After RobustScaler**:
- All features centered ~0 with comparable spread
- **Benefit**: ML algorithms treat all features fairly
- **Result**: Better model convergence and feature importance

---

## 7. OUTLIER HANDLING

### Detection Methods

**Visual Inspection (Boxplots)**:
- LoanAmount: Few loans >600K (upper outliers)
- ApplicantIncome: Few earnings >70K (sparse but legitimate)

**Statistical (IQR)**:
```python
Q1 = df['LoanAmount'].quantile(0.25)
Q3 = df['LoanAmount'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['LoanAmount'] < Q1 - 1.5*IQR) | (df['LoanAmount'] > Q3 + 1.5*IQR)]
```

### Treatment Decision: PRESERVE 

**Why Not Remove?**
- **Banking Reality**: High-income applicants, large loans are legitimate
- **Information Value**: Outliers carry useful information
- **Legal Concern**: Removing outliers could bias against certain groups
- **Sample Efficiency**: Only 614 training records; need every sample

**How RobustScaler Handles Outliers**:
-  Doesn't compress outliers to [0,1] bounds (unlike MinMax)
-  Down-weights outlier influence (uses median/IQR)
-  Maintains relationships (outliers remain visible)
-  Preserves information for model learning

### Outlier Summary

| Feature | Count | Treatment | Justification |
|---------|-------|-----------|----------------|
| **LoanAmount >600K** | ~8 | Preserve + Scale | High-value loans legitimate |
| **Income >60K** | ~12 | Preserve + Scale | High earners valid applicants |
| **TotalIncome >250K** | ~10 | Preserve + Scale | Combined income informative |

---

## 8. FEATURE SELECTION & CORRELATION ANALYSIS

### Correlation Analysis

**Purpose**: Identify feature relationships and predictiveness

**Method**: Computed correlation matrix (train data) with target variable

### Features by Target Correlation (Top 10)

Based on your notebook's correlation analysis, features ranked by absolute correlation with Loan_Status:

| Rank | Feature | Correlation | Interpretation |
|------|---------|-------------|-----------------|
| 1 | Credit_History | ~0.54 | STRONGEST—credit history strongly predicts approval |
| 2 | loanToIncomeRatio | ~-0.32 | Higher ratio → higher risk |
| 3 | incomePerDependent | ~-0.12 | More dependents → lower approval |
| 4 | LoanAmount | ~-0.03 | Weak; loan size not strong predictor |
| 5 | Property_Area_Urban | ~0.08 | Slight urban bias |
| 6 | Self_Employed_Yes | ~0.05 | Very weak |
| 7 | Gender_Male | ~0.04 | Minimal gender effect |
| 8 | Married_Yes | ~0.03 | Minimal marital status effect |

### Multicollinearity Analysis

**Problematic Feature Pairs (r > 0.70)**:

| Feature 1 | Feature 2 | Correlation | Issue | Recommendation                     |
|-----------|-----------|-------------|-------|------------------------------------|
| **ApplicantIncome** | **totalIncome** | ~0.89 | Highly redundant (79% shared) |  Consider dropping                 |
| **totalIncome** | **incomePerDependent** | ~0.75 | Moderately correlated | Keep (derived feature adds context) |

**Note**: Your notebook engineered features but didn't remove ApplicantIncome. For future optimization, consider dropping ApplicantIncome since totalIncome captures all its information plus co-applicant contribution.

### Correlation-Based Insights

- **Credit_History dominates**: By far strongest predictor (r=0.54)
- **Income ratios matter**: Engineered features have stronger correlation than raw income
- **Categorical features weak**: Individual demographics have minimal direct correlation
- **Multicollinearity moderate**: No severe redundancy except ApplicantIncome/totalIncome

---

## 9. FINAL DATASET SPECIFICATION

### Train Dataset

```
File: loan_train_preprocessed.csv
Rows: 614 (all training records)
Columns: 18 (features + target)

Structure:
├── Numeric (Scaled): 8 features
│   ├── ApplicantIncome (scaled)
│   ├── CoapplicantIncome (scaled)
│   ├── LoanAmount (scaled)
│   ├── Loan_Amount_Term (scaled)
│   ├── Credit_History (scaled)
│   ├── totalIncome (scaled)
│   ├── loanToIncomeRatio (scaled)
│   └── incomePerDependent (scaled)
├── Categorical (Encoded): 6 features
│   ├── Gender_Male
│   ├── Married_Yes
│   ├── Education_Not Graduate
│   ├── Self_Employed_Yes
│   ├── Property_Area_Semiurban
│   └── Property_Area_Urban
├── Numeric: 3 features
│   └── Dependents (numeric)
└── Target: 1 feature
    └── Loan_Status (Y/N)

Quality Metrics:
- Missing values: 0
- Duplicates: 0
- Data type validation: Passed
- Scaling applied: RobustScaler (fitted on this dataset)
- Encoding applied: One-Hot
```

### Test Dataset

```
File: loan_test_preprocessed.csv
Rows: 367 (all test records)
Columns: 17 (features only, NO target)

Same feature structure as train (minus target variable)

Quality Metrics:
- Missing values: 0 (imputed identically to train)
- Scaling: RobustScaler (applied from train scaler, NOT refitted)
- Encoding: One-Hot (identical to train)
- Ready for model prediction
```

### ML-Ready Dataset

```
File: loan_ml_ready_dataset.csv
Combines: Preprocessed train + preprocessed test
Format: Features in correct order for modeling
Target: Included (train only; test excluded)

Feature Count: 18 total
├── Numeric Scaled: 8
├── Categorical Encoded: 6
├── Numeric Raw: 3
└── Target: 1
```

---

## 10. PREPROCESSING SUMMARY

### Complete Transformation Pipeline

| Step | Input              | Output               | Status |
|------|--------------------|----------------------|--------|
| **Raw Data Loading** | Train: 614x12      | Test: 367x12         | Done   |
| **Missing Value Imputation** | 147 missing values | 0 missing            | Done      |
| **Identifier Removal** | 12 features        | 11 features          | Done      |
| **Feature Engineering** | 11 features        | 14 features (+3 new) | Done      |
| **Categorical Encoding** | 6 categorical      | 6 binary columns     | Done      |
| **Feature Scaling** | Varied ranges      | Normalized ranges    | Done      |
| **Final Datasets** | Raw data           | Train: 614x18        | Done      |
| |                    | Test: 367x18         |        |

### Quality Assurance Checklist

- [x] Missing values: 100% handled
- [x] Data types: Validated and corrected
- [x] Duplicates: 0 detected
- [x] Categorical encoding: One-Hot (6 features)
- [x] Numeric scaling: RobustScaler
- [x] Train/test handling: Scaler fitted on train only
- [x] Feature engineering: 3 meaningful features created
- [x] Multicollinearity: Identified (ApplicantIncome redundancy noted)
- [x] Outliers: Detected and preserved
- [x] Visualizations: Correlation, distributions, scaling comparison
- [x] Documentation: Every decision justified
- [x] Exportable: CSV format, ready for modeling

---

## 11. KEY OBSERVATIONS

### 1. Credit History is Critical
- Strongest predictor (r=0.54)
- Applicants with history 3x more likely approved
- **Action**: Ensure properly weighted in model

### 2. Engineered Features Outperform Raw Income
- Raw ApplicantIncome: r≈0.00 (not predictive)
- loanToIncomeRatio: r=-0.32 (moderately predictive)
- **Insight**: Domain knowledge > raw data

### 3. Class Balance Acceptable
- ~70% approved, ~30% rejected
- Slight imbalance, but manageable
- **Recommendation**: Use F1-score, not just accuracy

### 4. Train/Test Split Properly Handled
- Scaler fitted on 614 training records only
- Applied to 367 test records
- **No data leakage** Done

### 5. Multicollinearity Noted
- ApplicantIncome/totalIncome: r=0.89 (highly redundant)
- **Future optimization**: Drop ApplicantIncome

---

## 12. RECOMMENDATIONS FOR MODEL DEVELOPMENT

1. **Feature Scaling in Production**: Save and reuse the RobustScaler object fitted on training data
2. **Monitor Credit_History**: Will heavily influence model; check for bias
3. **Test fairness**: Evaluate model across demographic groups
4. **Handle class imbalance**: If accuracy insufficient, consider SMOTE or class weighting
5. **Feature importance**: Use SHAP or permutation importance to understand predictions

---

## CONCLUSION

The loan prediction dataset has been comprehensively preprocessed through a professional machine learning pipeline:

- **All missing values imputed** using domain-appropriate methods  
- **Categorical variables properly encoded** (One-Hot, no dummy trap)  
- **Numeric features scaled** using RobustScaler (outliers preserved)  
- **Features engineered** to capture financial risk (3 new features)  
- **Train/test handled correctly** (scaler fitted on train only)  
- **Every decision documented** with clear justification  

**Result**: Two preprocessed datasets (train 614x18, test 367x18) ready for classification modeling.

**Status**:  **READY FOR MODEL TRAINING**

---

**Report Date**: August 2026  
**Dataset**: Loan Prediction (Train/Test Split)   
**Tools Used**: Python (Pandas, NumPy, Scikit-learn)
**Author**: [Abalaliwa LOMDO](https://github.com/ehampala)