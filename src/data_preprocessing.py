import pandas as pd
import numpy as np
import random
from scipy import stats

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of records
num_records = 1500

# Step 1: Generate Clean Synthetic Data
patient_ids = [f"PID{1000+i}" for i in range(num_records)]
ages = np.clip(np.random.normal(loc=50, scale=15, size=num_records), 18, 90).astype(int)
genders = np.random.choice(['Male', 'Female'], size=num_records)
smoking_status = np.random.choice(['Yes', 'No'], size=num_records, p=[0.3, 0.7])
alcohol_use = np.random.choice(['Yes', 'No'], size=num_records, p=[0.4, 0.6])
family_history = np.random.choice(['Yes', 'No'], size=num_records, p=[0.2, 0.8])
marker1 = np.round(np.random.normal(loc=5, scale=2, size=num_records), 2)
marker2 = np.round(np.random.normal(loc=100, scale=20, size=num_records), 1)
symptom_score = np.random.randint(0, 11, size=num_records)

# Generate Diagnosis label based on marker thresholds + noise
diagnosis = []
for i in range(num_records):
    if marker1[i] > 7 and marker2[i] > 130:
        diagnosis.append(1 if random.random() > 0.2 else 0)
    else:
        diagnosis.append(0 if random.random() > 0.2 else 1)

# Assemble DataFrame
df = pd.DataFrame({
    'Patient_ID': patient_ids,
    'Age': ages,
    'Gender': genders,
    'Smoking_Status': smoking_status,
    'Alcohol_Use': alcohol_use,
    'Family_History': family_history,
    'Blood_Marker_1': marker1,
    'Blood_Marker_2': marker2,
    'Symptom_Score': symptom_score,
    'Cancer_Diagnosis': diagnosis
})

# Step 2: Inject Anomalies

## 2.1 Inject NULLs
for col in ['Age', 'Gender', 'Blood_Marker_1']:
    null_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[null_indices, col] = np.nan

## 2.2 Inject Duplicates
duplicate_rows = df.sample(n=15, random_state=42)
df = pd.concat([df, duplicate_rows], ignore_index=True)

## 2.3 Inject Outliers
df.loc[np.random.choice(df.index, 10, replace=False), 'Age'] = 150  # Unrealistic age
df.loc[np.random.choice(df.index, 10, replace=False), 'Blood_Marker_1'] = 25  # Abnormally high

## 2.4 Inject Format Issues
df.loc[np.random.choice(df.index, 10, replace=False), 'Age'] = 'forty'  # Age as string
df.loc[np.random.choice(df.index, 5, replace=False), 'Blood_Marker_2'] = 'one hundred'  # Text instead of float

# Save dataset with anomalies
df.to_excel("AI_Alula_Dataset_With_Anomalies.xlsx", index=False)

print("✅ Dataset generated and saved as 'AI_Alula_Dataset_With_Anomalies.xlsx'")

# Load the dataset with anomalies
df = pd.read_excel("AI_Alula_Dataset_With_Anomalies.xlsx")

# -------------------------
# STEP 1: Fix Format Issues
# -------------------------
# Convert to numeric, coercing errors (invalid formats like "forty" → NaN)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Blood_Marker_2'] = pd.to_numeric(df['Blood_Marker_2'], errors='coerce')

# --------------------------
# STEP 2: Remove Duplicates
# --------------------------
df.drop_duplicates(inplace=True)

# -----------------------------------
# STEP 3: Handle Missing Values (Central Tendency)
# -----------------------------------
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Blood_Marker_1'].fillna(df['Blood_Marker_1'].median(), inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)

# -----------------------------------
# STEP 4: Outlier Removal Using IQR
# -----------------------------------
def remove_iqr_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

df = remove_iqr_outliers(df, 'Age')
df = remove_iqr_outliers(df, 'Blood_Marker_1')

# ------------------------------------
# STEP 5: Outlier Removal Using Z-score
# ------------------------------------
z_scores = np.abs(stats.zscore(df['Blood_Marker_2'].dropna()))
threshold = 3
valid_indices = df['Blood_Marker_2'].dropna().index[z_scores < threshold]
df = df.loc[valid_indices]

# ------------------------------------
# STEP 6: Finalize
# ------------------------------------
df.reset_index(drop=True, inplace=True)

# Save the cleaned dataset
df.to_excel("AI_Alula_CleanedDataset.xlsx", index=False)

print("✅ Cleaned dataset saved as 'AI_Alula_CleanedDataset.xlsx'")
