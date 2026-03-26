# 🦠 COVID-19 Early Case Trend Analysis & Recovery Insights

> **"This project was completed as part of my training under @ediglobe"**
> 📧support@ediglobe.in
---

## 📌 Project Overview

This project performs a comprehensive **Exploratory Data Analysis (EDA)** and **Machine Learning** study on early-stage COVID-19 patient data for **HealthGuard Analytics Pvt. Ltd.**

The goal is to help the **National Public Health Authority** make data-driven decisions by analyzing:
- Patient demographics (age, gender, region)
- Infection spread patterns
- Recovery timelines
- Regional impact
- Factors influencing recovery duration

---

## 🏢 Company Details

| Field | Details |
|---|---|
| **Company** | HealthGuard Analytics Pvt. Ltd. |
| **Client** | National Public Health Authority |
| **Project** | COVID-19 Early Case Trend Analysis & Recovery Insights |
| **Tech Stack** | Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn |

---

## ❓ Problem Statement

The public health authority wanted data-driven answers to 5 key questions:

| # | Question | Analysis Done |
|---|---|---|
| 1 | Who is getting infected? | Age, gender, region demographics |
| 2 | How are infections spreading? | Infection reasons, order, contact exposure |
| 3 | What are the recovery trends? | Time from confirmation to release |
| 4 | Which regions are most impacted? | Confirmed vs released cases by region |
| 5 | What factors influence recovery time? | Age, contact number, infection order |

---

## ⚙️ System Architecture & Workflow

| Step | Phase | Description | Output |
|---|---|---|---|
| 1 | Data Ingestion | Load CSV using Pandas | Raw DataFrame |
| 2 | EDA | Shape, dtypes, missing values | Data quality report |
| 3 | Data Cleaning | Date parsing, standardisation | Clean DataFrame |
| 4 | Feature Engineering | `age`, `recovery_days`, `sex_encoded` | 3 new features |
| 5 | Visualisation | 10 charts across all dimensions | PNG figures |
| 6 | Correlation Analysis | Pearson matrix | Heatmap |
| 7 | Linear Regression | Predict recovery days | R² score, MAE |
| 8 | Residual Analysis | Model evaluation | Residual plot |

---

## 🚀 Logical Flow

```
START

Load Dataset (covid19_cases.csv)

EDA:
  → Check shape, dtypes, missing values
  → Descriptive statistics

Data Cleaning:
  → Parse date columns
  → Standardise text fields

Feature Engineering:
  → age = 2020 - birth_year
  → recovery_days = released_date - confirmed_date
  → sex_encoded = LabelEncoder(sex)

Visualisations:
  → Gender distribution (pie chart)
  → Age distribution (histogram + boxplot)
  → Regional analysis (bar charts)
  → Infection reasons (horizontal bar)
  → Infection order (bar chart)
  → Case outcomes (bar chart)
  → Recovery timelines (histogram + trend)
  → Recovery by gender (boxplot)
  → Correlation heatmap

Linear Regression:
  → Features: age, contact_number, infection_order, sex_encoded
  → Target: recovery_days
  → Split: 80% train / 20% test
  → Evaluate: R² score + MAE
  → Plot: Residuals + Actual vs Predicted

END
```

---

## 🛠️ Installation & Usage

### 1️⃣ Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### 2️⃣ Dataset

Download the dataset from the link below and save it as `covid19_cases.csv` inside the `data/` folder:

📥 [Download Dataset][https://drive.google.com/file/d/1TXoqikmE0S3LGem8IgGktaJJyZPGMgN/view?usp=sharing](https://drive.google.com/file/d/1TXoqikmE0S3LGem8Ig-GktaJJyZPGMgN/view))

### 3️⃣ Running the Project

```bash
python covid19_analysis.py
```

---

## 📊 Key Visualisations Generated

| File | Description |
|---|---|
| `fig1_gender_distribution.png` | Pie chart — Male vs Female case split |
| `fig2_age_distribution.png` | Histogram + boxplot of age by gender |
| `fig3_regional_analysis.png` | Confirmed vs released cases by region |
| `fig4_infection_reasons.png` | Top infection sources bar chart |
| `fig5_infection_order.png` | Infection generation distribution |
| `fig6_outcomes.png` | Case outcome breakdown |
| `fig7_recovery_timelines.png` | Recovery duration + monthly trend |
| `fig8_recovery_by_gender.png` | Recovery days split by gender |
| `fig9_correlation_heatmap.png` | Pearson correlation matrix |
| `fig10_regression_analysis.png` | Residual plot + Actual vs Predicted |

---

## 🧠 Linear Regression Model Details

The optional extension builds a **Multi-Variable Linear Regression** model to predict `recovery_days`:

```
Target:   recovery_days
Features: age, contact_number, infection_order, sex_encoded

Results:
  age              → 0.0650  (older = longer recovery)
  contact_number   → -0.0093
  infection_order  → -0.2285
  sex_encoded      → -2.1492
  Intercept        → 15.5085 (~15.5 base recovery days)
```

### Model Evaluation
- ✅ **R² Score** — measures how well the model explains variance
- ✅ **MAE** — average error in days
- ✅ **Residual Plot** — checks model assumptions
- ✅ **Actual vs Predicted Chart** — visual fit quality

---

## 📂 Project Structure

```
covid19-trend-analysis/
│
├── data/
│   └── covid19_cases.csv          ← Dataset (download separately)
│
├── covid19_analysis.py            ← Main analysis script
├── requirements.txt               ← Python dependencies
└── README.md                      ← Project documentation
```

---

## 📈 Key Findings

- 👥 **Adults aged 40–65** are the most frequently infected group
- ✈️ **Overseas travel** was the primary infection source in early outbreak data
- ⏱️ **Average recovery: ~17 days** | Median: ~14 days
- 📍 **A few high-density regions** carry disproportionate case loads
- 🔬 **Age** is the strongest predictor of recovery duration

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

If you find this project useful, consider giving it a ⭐

---

> **"This project was completed as part of my training under @ediglobe"**
> 📧support@ediglobe.in
