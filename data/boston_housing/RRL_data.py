"""
RRL Data Processing Script
Processes Boston Housing dataset and saves in RRL format (.data and .info files)
"""

import pandas as pd

# File paths
# input_file = './HousingData.csv'
# output_base = 'boston_housing'
input_file = './bh_decent.csv'
output_base = 'bh_decent'
output_data_file = f'{output_base}.data'
output_info_file = f'{output_base}.info'

# Feature names
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Feature types: continuous or discrete
# CHAS is binary (0/1) - discrete
# RAD is categorical - discrete
# All others are continuous
feature_types = {
    'CRIM': 'continuous',
    'ZN': 'continuous',
    'INDUS': 'continuous',
    'CHAS': 'discrete',
    'NOX': 'continuous',
    'RM': 'continuous',
    'AGE': 'continuous',
    'DIS': 'continuous',
    'RAD': 'discrete',
    'TAX': 'continuous',
    'PTRATIO': 'continuous',
    'B': 'continuous',
    'LSTAT': 'continuous',
    'MEDV': 'continuous'  # Target variable
}

# Class label column name
class_label = 'MEDV'

print("=" * 60)
print("RRL Data Processing")
print("=" * 60)

# 1. Load data
print(f"\n1. Loading data from {input_file}")
data = pd.read_csv(input_file)
df = pd.DataFrame(data, columns=feature_names)
print(f"   Loaded {df.shape[0]} samples with {df.shape[1]} features")

# 2. Fill NA values with median
print("\n2. Filling NA values with median")
df_filled = df.copy()

na_before = df_filled.isna().sum()
na_columns = na_before[na_before > 0]

if len(na_columns) > 0:
    print(f"   Found NA values in {len(na_columns)} columns")
    for col in df_filled.columns:
        if df_filled[col].isna().sum() > 0:
            median_value = df_filled[col].median()
            na_count = df_filled[col].isna().sum()
            df_filled[col].fillna(median_value, inplace=True)
            print(f"   Column '{col}': {na_count} NA values filled with median {median_value:.4f}")
else:
    print("   No NA values found")

# Verify no NA values remain
remaining_na = df_filled.isna().sum().sum()
if remaining_na > 0:
    print(f"   WARNING: {remaining_na} NA values still remain!")
else:
    print("   ✓ All NA values have been filled")

# Convert discrete features to integers
print("\n3. Converting discrete features to integers")
for col in df_filled.columns:
    if feature_types.get(col, 'continuous') == 'discrete':
        df_filled[col] = df_filled[col].astype(int)
        print(f"   Column '{col}': converted to integer")

# 4. Save .data file
print(f"\n4. Saving data to {output_data_file}")
# Save without header, space-separated values
# Use custom formatting: integers for discrete, floats for continuous
with open(output_data_file, 'w') as f:
    for idx, row in df_filled.iterrows():
        values = []
        for col in df_filled.columns:
            if feature_types.get(col, 'continuous') == 'discrete':
                values.append(str(int(row[col])))
            else:
                values.append(f"{row[col]:.6f}")
        f.write(','.join(values) + '\n')
print(f"   ✓ Saved {df_filled.shape[0]} rows and {df_filled.shape[1]} columns")

# 5. Save .info file
print(f"\n5. Saving info to {output_info_file}")

# Find the position of class label column
class_label_pos = df_filled.columns.get_loc(class_label)

with open(output_info_file, 'w') as f:
    # Write feature information (one row per feature)
    for col in df_filled.columns:
        feature_type = feature_types.get(col, 'continuous')
        f.write(f"{col} {feature_type}\n")
    
    # Write class label position (last row)
    f.write(f"LABEL_POS {class_label_pos}")

print(f"   ✓ Saved feature information for {len(df_filled.columns)} features")
print(f"   ✓ Class label '{class_label}' is at position {class_label_pos}")

# 6. Summary
print("\n" + "=" * 60)
print("Processing Summary")
print("=" * 60)
print(f"Input file: {input_file}")
print(f"Output files: {output_data_file}, {output_info_file}")
print(f"Total samples: {df_filled.shape[0]}")
print(f"Total features: {df_filled.shape[1]}")
print(f"Class label: {class_label} (position {class_label_pos})")
print(f"Continuous features: {sum(1 for t in feature_types.values() if t == 'continuous')}")
print(f"Discrete features: {sum(1 for t in feature_types.values() if t == 'discrete')}")
print("\n✓ Processing complete!")

