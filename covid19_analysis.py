# =============================================================================
# COVID-19 Early Case Trend Analysis & Recovery Insights
# HealthGuard Analytics Pvt. Ltd.
# =============================================================================

# ─── SETUP & IMPORTS ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Global style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
# Download the dataset from the Drive link and save locally as 'covid19_cases.csv'
# Dataset link:
# https://drive.google.com/file/d/1TXoqikmE0S3LGem8IgGktaJJyZPGMgN/view?usp=sharing

CSV_PATH = "covid19_cases.csv"    # ← update path if needed

df = pd.read_csv(CSV_PATH)
print("✅ Dataset loaded  →  Shape:", df.shape)
print("\nColumns:", df.columns.tolist())


# ─── 2. EXPLORATORY DATA ANALYSIS (EDA) ──────────────────────────────────────
print("\n" + "="*60)
print("SECTION 2: EDA")
print("="*60)

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct}))

print("\n--- Descriptive Statistics (numeric) ---")
print(df.describe())


# ─── 3. DATA CLEANING & FEATURE ENGINEERING ───────────────────────────────────
print("\n" + "="*60)
print("SECTION 3: DATA CLEANING")
print("="*60)

# Parse dates
date_cols = ['confirmed_date', 'released_date', 'deceased_date']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Derive age from birth_year (using 2020 as reference – early outbreak)
if 'birth_year' in df.columns:
    df['age'] = 2020 - df['birth_year']

# Derive recovery duration (days from confirmation to release)
if 'confirmed_date' in df.columns and 'released_date' in df.columns:
    df['recovery_days'] = (df['released_date'] - df['confirmed_date']).dt.days

# Standardise text columns
for col in ['sex', 'state', 'infection_reason']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

print("Feature engineering complete.")
print("New columns added:", [c for c in ['age', 'recovery_days'] if c in df.columns])


# ─── 4. DEMOGRAPHIC ANALYSIS ──────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 4: DEMOGRAPHICS")
print("="*60)

# 4a. Gender Distribution
if 'sex' in df.columns:
    gender_counts = df['sex'].value_counts()
    print("\nGender distribution:\n", gender_counts)

    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ['#4A90D9', '#E8736B']
    ax.pie(gender_counts, labels=gender_counts.index.str.title(),
           autopct='%1.1f%%', colors=colors, startangle=140,
           wedgeprops=dict(edgecolor='white', linewidth=2))
    ax.set_title("Gender Distribution of COVID-19 Cases")
    plt.tight_layout()
    plt.savefig("fig1_gender_distribution.png")
    plt.show()
    print("Saved: fig1_gender_distribution.png")

# 4b. Age Distribution
if 'age' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    axes[0].hist(df['age'].dropna(), bins=20, color='#4A90D9', edgecolor='white')
    axes[0].set(title="Age Distribution of Confirmed Cases",
                xlabel="Age (years)", ylabel="Number of Cases")
    axes[0].axvline(df['age'].median(), color='red', linestyle='--',
                    label=f"Median: {df['age'].median():.0f}")
    axes[0].legend()

    # Box plot by gender
    if 'sex' in df.columns:
        gender_age = df[df['sex'].isin(['male', 'female'])]
        axes[1].boxplot(
            [gender_age[gender_age['sex'] == g]['age'].dropna() for g in ['male', 'female']],
            labels=['Male', 'Female'], patch_artist=True,
            boxprops=dict(facecolor='#AED6F1'),
            medianprops=dict(color='red', linewidth=2)
        )
        axes[1].set(title="Age Distribution by Gender", ylabel="Age (years)")
    else:
        axes[1].axis('off')

    plt.tight_layout()
    plt.savefig("fig2_age_distribution.png")
    plt.show()
    print("Saved: fig2_age_distribution.png")

    print(f"\nAge Stats:\n  Mean: {df['age'].mean():.1f}\n  Median: {df['age'].median():.1f}"
          f"\n  Min: {df['age'].min():.0f}  Max: {df['age'].max():.0f}")


# ─── 5. REGIONAL ANALYSIS ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 5: REGIONAL ANALYSIS")
print("="*60)

if 'region' in df.columns and 'state' in df.columns:
    region_total   = df['region'].value_counts().head(15)
    region_release = df[df['state'] == 'released']['region'].value_counts()
    region_compare = pd.DataFrame({
        'Confirmed': region_total,
        'Released':  region_release
    }).fillna(0).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Bar – total confirmed
    region_total.plot(kind='barh', ax=axes[0], color='#5DADE2', edgecolor='white')
    axes[0].set(title="Top 15 Regions by Confirmed Cases",
                xlabel="Number of Cases", ylabel="Region")
    axes[0].invert_yaxis()

    # Grouped bar – confirmed vs released
    region_compare.head(10).plot(kind='bar', ax=axes[1],
                                  color=['#5DADE2', '#58D68D'], edgecolor='white')
    axes[1].set(title="Confirmed vs Released Cases (Top 10 Regions)",
                xlabel="Region", ylabel="Cases")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("fig3_regional_analysis.png")
    plt.show()
    print("Saved: fig3_regional_analysis.png")

elif 'country' in df.columns:
    country_counts = df['country'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    country_counts.plot(kind='bar', ax=ax, color='#5DADE2', edgecolor='white')
    ax.set(title="Cases by Country (Top 10)", xlabel="Country", ylabel="Cases")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig("fig3_regional_analysis.png")
    plt.show()
    print("Saved: fig3_regional_analysis.png")


# ─── 6. INFECTION SOURCE ANALYSIS ─────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 6: INFECTION SOURCES")
print("="*60)

if 'infection_reason' in df.columns:
    reason_counts = df['infection_reason'].value_counts().head(12)
    print("\nTop infection reasons:\n", reason_counts)

    fig, ax = plt.subplots(figsize=(10, 6))
    reason_counts.sort_values().plot(kind='barh', ax=ax,
                                      color='#F0A500', edgecolor='white')
    ax.set(title="Primary Infection Sources",
           xlabel="Number of Cases", ylabel="Infection Reason")
    plt.tight_layout()
    plt.savefig("fig4_infection_reasons.png")
    plt.show()
    print("Saved: fig4_infection_reasons.png")

if 'infection_order' in df.columns:
    order_counts = df['infection_order'].value_counts().sort_index()
    print("\nInfection order distribution:\n", order_counts)

    fig, ax = plt.subplots(figsize=(7, 4))
    order_counts.plot(kind='bar', ax=ax, color='#8E44AD', edgecolor='white')
    ax.set(title="Distribution of Infection Order",
           xlabel="Infection Order (Generation)", ylabel="Number of Cases")
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig("fig5_infection_order.png")
    plt.show()
    print("Saved: fig5_infection_order.png")


# ─── 7. CASE OUTCOME ANALYSIS ─────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 7: CASE OUTCOMES")
print("="*60)

if 'state' in df.columns:
    outcome_counts = df['state'].value_counts()
    print("\nOutcome distribution:\n", outcome_counts)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {'released': '#58D68D', 'isolated': '#F0A500', 'deceased': '#E8736B'}
    bar_colors = [colors.get(s, '#95A5A6') for s in outcome_counts.index]
    outcome_counts.plot(kind='bar', ax=ax, color=bar_colors, edgecolor='white')
    ax.set(title="Case Outcome Distribution",
           xlabel="Outcome", ylabel="Number of Cases")
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig("fig6_outcomes.png")
    plt.show()
    print("Saved: fig6_outcomes.png")


# ─── 8. RECOVERY TIMELINE ANALYSIS ───────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 8: RECOVERY TIMELINES")
print("="*60)

if 'recovery_days' in df.columns:
    rec = df[df['recovery_days'] > 0]['recovery_days']
    print(f"\nRecovery Days:\n  Mean:   {rec.mean():.1f} days"
          f"\n  Median: {rec.median():.1f} days"
          f"\n  Std:    {rec.std():.1f} days"
          f"\n  Min:    {rec.min():.0f}   Max: {rec.max():.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram of recovery days
    axes[0].hist(rec, bins=25, color='#58D68D', edgecolor='white')
    axes[0].axvline(rec.mean(), color='red', linestyle='--',
                    label=f"Mean: {rec.mean():.1f}")
    axes[0].axvline(rec.median(), color='blue', linestyle='-.',
                    label=f"Median: {rec.median():.1f}")
    axes[0].set(title="Distribution of Recovery Duration",
                xlabel="Days to Recovery", ylabel="Frequency")
    axes[0].legend()

    # Monthly trend of confirmed cases
    if 'confirmed_date' in df.columns:
        monthly = df.groupby(df['confirmed_date'].dt.to_period('M')).size()
        monthly.plot(ax=axes[1], color='#5DADE2', marker='o', linewidth=2)
        axes[1].set(title="Monthly Confirmed Cases Trend",
                    xlabel="Month", ylabel="Confirmed Cases")
        axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("fig7_recovery_timelines.png")
    plt.show()
    print("Saved: fig7_recovery_timelines.png")

    # Recovery days by gender
    if 'sex' in df.columns:
        rec_gender = df[df['recovery_days'] > 0][['sex', 'recovery_days']]
        rec_gender = rec_gender[rec_gender['sex'].isin(['male', 'female'])]
        fig, ax = plt.subplots(figsize=(6, 5))
        rec_gender.boxplot(column='recovery_days', by='sex', ax=ax,
                           patch_artist=True)
        ax.set(title="Recovery Days by Gender",
               xlabel="Gender", ylabel="Days to Recovery")
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig("fig8_recovery_by_gender.png")
        plt.show()
        print("Saved: fig8_recovery_by_gender.png")


# ─── 9. DESCRIPTIVE STATISTICS SUMMARY ───────────────────────────────────────
print("\n" + "="*60)
print("SECTION 9: CORRELATION & STATISTICAL SUMMARY")
print("="*60)

num_cols = [c for c in ['age', 'contact_number', 'infection_order', 'recovery_days']
            if c in df.columns]

if len(num_cols) >= 2:
    corr_df = df[num_cols].dropna()
    corr_matrix = corr_df.corr()
    print("\nCorrelation Matrix:\n", corr_matrix.round(3))

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                center=0, ax=ax, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig("fig9_correlation_heatmap.png")
    plt.show()
    print("Saved: fig9_correlation_heatmap.png")


# ─── 10. LINEAR REGRESSION – OPTIONAL EXTENSION ──────────────────────────────
print("\n" + "="*60)
print("SECTION 10: LINEAR REGRESSION (Recovery Time Prediction)")
print("="*60)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

feature_cols = [c for c in ['age', 'contact_number', 'infection_order'] if c in df.columns]

if 'recovery_days' in df.columns and len(feature_cols) >= 1:
    # Optional: encode gender as binary
    if 'sex' in df.columns:
        le = LabelEncoder()
        df['sex_encoded'] = le.fit_transform(df['sex'].fillna('unknown'))
        feature_cols.append('sex_encoded')

    reg_df = df[feature_cols + ['recovery_days']].dropna()
    reg_df = reg_df[reg_df['recovery_days'] > 0]

    X = reg_df[feature_cols]
    y = reg_df['recovery_days']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\nLinear Regression Results:")
    print(f"  R² Score : {r2:.4f}")
    print(f"  MAE      : {mae:.2f} days")
    print(f"\nCoefficients:")
    for feat, coef in zip(feature_cols, model.coef_):
        print(f"  {feat:20s}: {coef:.4f}")
    print(f"  {'Intercept':20s}: {model.intercept_:.4f}")

    # Residual plot
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.5, color='#5DADE2', edgecolors='none')
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set(title="Residual Plot", xlabel="Predicted Recovery Days",
                ylabel="Residuals")

    axes[1].scatter(y_test, y_pred, alpha=0.5, color='#F0A500', edgecolors='none')
    axes[1].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], 'r--')
    axes[1].set(title=f"Actual vs Predicted  (R²={r2:.3f})",
                xlabel="Actual Recovery Days", ylabel="Predicted Recovery Days")

    plt.tight_layout()
    plt.savefig("fig10_regression_analysis.png")
    plt.show()
    print("Saved: fig10_regression_analysis.png")

else:
    print("Insufficient data for regression. Skipping.")


# ─── DONE ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("✅ Analysis Complete! All figures saved.")
print("="*60)