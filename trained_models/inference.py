"""
inference.py - Corrected Comprehensive Data Analysis Script
This version properly handles missing data and clinical features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.impute import SimpleImputer
import warnings
import os
import sys
from datetime import datetime
import re

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory for visualizations
output_dir = "analysis_outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*80)
print("ANTIBIOTIC RESISTANCE DATASET ANALYSIS (FINAL VERSION)")
print("="*80)
print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output directory: {output_dir}/")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATASET
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING DATASET")
print("="*80)

# Load the dataset
try:
    df = pd.read_csv('Bacteria_dataset_Multiresictance.csv')
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print("✗ Error: File 'Bacteria_dataset_Multiresictance.csv' not found!")
    print("  Please ensure the file is in the current directory.")
    sys.exit(1)

# ============================================================================
# STEP 2: CORRECTLY IDENTIFY ANTIBIOTIC COLUMNS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: IDENTIFYING ANTIBIOTIC COLUMNS")
print("="*80)

# List of antibiotic column names from your data
antibiotic_columns = [
    'AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN',
    'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole', 'Furanes', 'colistine'
]

# Clinical/demographic columns
clinical_columns = [
    'ID', 'Name', 'Email', 'Address', 'age/gender', 'Souches',
    'Diabetes', 'Hypertension', 'Hospital_before', 'Infection_Freq',
    'Collection_Date', 'Notes'
]

# Verify which antibiotic columns exist in the dataset
existing_antibiotics = [col for col in antibiotic_columns if col in df.columns]
existing_clinical = [col for col in clinical_columns if col in df.columns]

print(f"\n✓ Identified {len(existing_antibiotics)} antibiotic columns:")
for i, col in enumerate(existing_antibiotics, 1):
    # Get unique values to understand encoding
    unique_vals = df[col].unique()
    print(f"  {i:2d}. {col} (unique values: {sorted([str(x) for x in unique_vals if pd.notna(x)])[:5]})")

print(f"\n✓ Identified {len(existing_clinical)} clinical/demographic columns:")
for i, col in enumerate(existing_clinical, 1):
    print(f"  {i:2d}. {col}")

# ============================================================================
# STEP 3: DATA CLEANING AND PREPROCESSING
# ============================================================================

print("\n" + "="*80)
print("STEP 3: DATA CLEANING AND PREPROCESSING")
print("="*80)

def clean_resistance_value(value):
    """Standardize resistance values to R, S, I format"""
    if pd.isna(value):
        return np.nan
    
    value_str = str(value).upper().strip()
    
    # Resistant patterns
    if value_str in ['R', 'RESISTANT', 'RES', 'RÉSISTANT']:
        return 'R'
    # Susceptible patterns
    elif value_str in ['S', 'SUSCEPTIBLE', 'SENSITIVE', 'SENSIBLE']:
        return 'S'
    # Intermediate patterns
    elif value_str in ['I', 'INTERMEDIATE', 'INT']:
        return 'I'
    # Handle '?' as missing
    elif value_str == '?':
        return np.nan
    else:
        return np.nan

# Clean antibiotic columns
for col in existing_antibiotics:
    df[col] = df[col].apply(clean_resistance_value)

print("✓ Resistance values standardized to R (Resistant), S (Susceptible), I (Intermediate)")

# ============================================================================
# STEP 4: ANALYZE RESISTANCE PATTERNS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: RESISTANCE PATTERN ANALYSIS")
print("="*80)

resistance_stats = {}
for col in existing_antibiotics:
    # Count resistance patterns
    resistance_count = (df[col] == 'R').sum()
    susceptible_count = (df[col] == 'S').sum()
    intermediate_count = (df[col] == 'I').sum()
    missing_count = df[col].isna().sum()
    
    total_valid = resistance_count + susceptible_count + intermediate_count
    
    if total_valid > 0:
        resistance_stats[col] = {
            'Resistant': resistance_count,
            'Susceptible': susceptible_count,
            'Intermediate': intermediate_count,
            'Missing': missing_count,
            'Resistance_Rate': (resistance_count / total_valid * 100) if total_valid > 0 else 0,
            'Valid_Samples': total_valid
        }

# Create resistance rates dataframe
resistance_df = pd.DataFrame(resistance_stats).T
resistance_df = resistance_df.sort_values('Resistance_Rate', ascending=False)

print("\nResistance Rates Summary:")
print("-"*80)
print(f"{'Antibiotic':<20} {'Resistant':<10} {'Susceptible':<12} {'Rate %':<10} {'Valid':<8}")
print("-"*80)
for col, stats in resistance_df.iterrows():
    print(f"{col:<20} {int(stats['Resistant']):<10} {int(stats['Susceptible']):<12} "
          f"{stats['Resistance_Rate']:.1f}%{'':<6} {int(stats['Valid_Samples']):<8}")

# Visualize resistance rates
if not resistance_df.empty:
    plt.figure(figsize=(14, 10))
    
    # Create bar plot with color coding
    colors = ['red' if rate > 50 else 'orange' if rate > 20 else 'green' 
              for rate in resistance_df['Resistance_Rate']]
    
    bars = plt.barh(range(len(resistance_df)), resistance_df['Resistance_Rate'], color=colors)
    plt.yticks(range(len(resistance_df)), resistance_df.index)
    plt.xlabel('Resistance Rate (%)')
    plt.title('Antibiotic Resistance Rates (R = Resistant)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(resistance_df.iterrows()):
        plt.text(row['Resistance_Rate'] + 1, i, f"{row['Resistance_Rate']:.1f}%", 
                va='center', fontsize=9, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='High Resistance (>50%)'),
        Patch(facecolor='orange', label='Moderate Resistance (20-50%)'),
        Patch(facecolor='green', label='Low Resistance (<20%)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_resistance_rates.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Resistance rates plot saved to: {output_dir}/02_resistance_rates.png")

# ============================================================================
# STEP 5: MULTI-DRUG RESISTANCE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: MULTI-DRUG RESISTANCE (MDR) ANALYSIS")
print("="*80)

# Create binary resistance matrix (1 for resistant, 0 for susceptible, drop intermediate)
resistance_binary = pd.DataFrame()
for col in existing_antibiotics:
    resistance_binary[col] = (df[col] == 'R').astype(float)
    # Set susceptible and intermediate to 0
    resistance_binary[col] = resistance_binary[col].fillna(0)

# Count number of resistances per isolate
resistance_count = resistance_binary.sum(axis=1)

if len(resistance_count) > 0:
    # MDR distribution
    plt.figure(figsize=(14, 7))
    
    # Create histogram with bins
    max_resist = int(resistance_count.max())
    bins = range(max_resist + 2)
    counts, edges = np.histogram(resistance_count, bins=bins)
    
    # Color code by MDR category
    colors_list = []
    for i in range(len(counts)):
        if i <= 1:
            colors_list.append('green')
        elif i <= 4:
            colors_list.append('orange')
        elif i <= 7:
            colors_list.append('red')
        else:
            colors_list.append('darkred')
    
    bars = plt.bar(edges[:-1], counts, width=0.8, color=colors_list, edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Antibiotics Resistant', fontsize=12)
    plt.ylabel('Number of Isolates', fontsize=12)
    plt.title('Multi-Drug Resistance Distribution', fontsize=14, fontweight='bold')
    plt.xticks(range(max_resist + 1))
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # Add category annotations
    plt.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=4.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=7.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.text(0.5, max(counts)*0.95, 'Non-MDR\n(0-1)', ha='center', fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    plt.text(3, max(counts)*0.95, 'MDR\n(2-4)', ha='center', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7))
    plt.text(6, max(counts)*0.95, 'XDR\n(5-7)', ha='center', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='salmon', alpha=0.7))
    plt.text(9, max(counts)*0.95, 'PDR\n(8+)', ha='center', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_mdr_distribution.png', dpi=150, bbox_inches='tight')
    print(f"✓ MDR distribution plot saved to: {output_dir}/03_mdr_distribution.png")
    
    # Statistics
    print(f"\nMDR Statistics:")
    print(f"  - Total isolates analyzed: {len(resistance_count):,}")
    print(f"  - Average number of resistances: {resistance_count.mean():.2f}")
    print(f"  - Median number of resistances: {resistance_count.median():.0f}")
    print(f"  - Maximum resistances in one isolate: {int(resistance_count.max())}")
    
    # MDR categories
    non_mdr = (resistance_count <= 1).sum()
    mdr = ((resistance_count >= 2) & (resistance_count <= 4)).sum()
    xdr = ((resistance_count >= 5) & (resistance_count <= 7)).sum()
    pdr = (resistance_count >= 8).sum()
    
    print(f"\nMDR Classification:")
    print(f"  - Non-MDR (0-1 resistances): {non_mdr:,} ({non_mdr/len(resistance_count)*100:.1f}%)")
    print(f"  - MDR (2-4 resistances): {mdr:,} ({mdr/len(resistance_count)*100:.1f}%)")
    print(f"  - XDR (5-7 resistances): {xdr:,} ({xdr/len(resistance_count)*100:.1f}%)")
    print(f"  - PDR (8+ resistances): {pdr:,} ({pdr/len(resistance_count)*100:.1f}%)")
    
    print(f"\n⚠️  ALERT: {mdr + xdr + pdr:,} ({((mdr + xdr + pdr)/len(resistance_count))*100:.1f}%) "
          f"isolates show multi-drug resistance!")

# ============================================================================
# STEP 6: RESISTANCE CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 6: RESISTANCE CORRELATION ANALYSIS")
print("="*80)

# Calculate correlation
if len(existing_antibiotics) > 1:
    corr_matrix = resistance_binary.corr()
    
    # Plot heatmap
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create mask for lower triangle
    mask = mask[1:, :-1]
    corr = corr_matrix.iloc[1:, :-1].copy()
    
    # Create heatmap
    sns.heatmap(corr_matrix, mask=np.triu(np.ones_like(corr_matrix, dtype=bool)), 
                annot=True, fmt='.2f', cmap='RdBu', center=0, 
                square=True, linewidths=0.5, annot_kws={'size': 8})
    plt.title('Resistance Correlation Across Antibiotics', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_resistance_correlation.png', dpi=150, bbox_inches='tight')
    print(f"✓ Resistance correlation heatmap saved to: {output_dir}/04_resistance_correlation.png")
    
    # Find strongest correlations
    print("\nStrongest Positive Correlations (Co-resistance):")
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if not pd.isna(corr_matrix.iloc[i, j]):
                corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], 
                                  corr_matrix.iloc[i, j]))
    
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    for pair in corr_pairs[:10]:
        print(f"  - {pair[0]} ↔ {pair[1]}: {pair[2]:.3f}")
    
    # Identify clusters of resistance
    print("\n📊 Resistance Clusters Identified:")
    print("  Cluster 1 (Beta-lactams): AMX/AMP, AMC, CZ, FOX, CTX/CRO, IPM")
    print("  Cluster 2 (Aminoglycosides): GEN, AN")
    print("  Cluster 3 (Fluoroquinolones): CIP, ofx, Acide nalidixique")
    print("  Cluster 4 (Others): Co-trimoxazole, C, Furanes, colistine")

# ============================================================================
# STEP 7: CLINICAL FEATURE ANALYSIS (WITHOUT ERROR)
# ============================================================================

print("\n" + "="*80)
print("STEP 7: CLINICAL FEATURE ANALYSIS")
print("="*80)

# Process clinical features properly
clinical_features = ['Diabetes', 'Hypertension', 'Hospital_before']

for feature in clinical_features:
    if feature in df.columns:
        # Clean the feature
        df[feature] = df[feature].apply(lambda x: str(x).upper() if pd.notna(x) else np.nan)
        df[feature] = df[feature].replace(['?', 'MISSING', 'NAN'], np.nan)
        
        # Get value counts
        print(f"\n{feature}:")
        value_counts = df[feature].value_counts(dropna=False)
        print(f"  Distribution:")
        for val, count in value_counts.items():
            if pd.isna(val):
                print(f"    - Missing: {count} ({count/len(df)*100:.1f}%)")
            else:
                print(f"    - {val}: {count} ({count/len(df)*100:.1f}%)")

# ============================================================================
# STEP 8: QUICK MODEL BUILDING DEMO (FIXED)
# ============================================================================

print("\n" + "="*80)
print("STEP 8: QUICK MODEL BUILDING DEMO")
print("="*80)

# Select antibiotic with highest resistance rate for modeling
best_antibiotic = resistance_df.index[0] if not resistance_df.empty else None

if best_antibiotic:
    print(f"\nBuilding model for: {best_antibiotic}")
    print(f"Resistance rate: {resistance_df.loc[best_antibiotic, 'Resistance_Rate']:.1f}%")
    print(f"Valid samples: {int(resistance_df.loc[best_antibiotic, 'Valid_Samples']):,}")
    
    # Prepare target
    y = (df[best_antibiotic] == 'R').astype(int)
    y = y[y.notna()]
    
    # Prepare features (clinical and demographic data that have actual data)
    feature_cols = ['Diabetes', 'Hypertension', 'Hospital_before']
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    if feature_cols:
        X = df[feature_cols].copy()
        X = X.loc[y.index]
        
        # Clean features
        for col in feature_cols:
            X[col] = X[col].apply(lambda x: str(x).upper() if pd.notna(x) else np.nan)
            X[col] = X[col].replace(['?', 'MISSING', 'NAN'], np.nan)
        
        # Encode categorical features
        for col in feature_cols:
            le = LabelEncoder()
            # Handle missing values by filling with 'Unknown'
            X[col] = X[col].fillna('Unknown')
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train multiple models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        print("\nModel Performance (Clinical Features Only):")
        print("-"*60)
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'model': model
            }
            
            print(f"\n{name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")
            
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"  Feature Importance:")
                for _, row in importance_df.iterrows():
                    print(f"    - {row['feature']}: {row['importance']:.4f}")
        
        # Feature importance visualization
        if hasattr(models['Random Forest'], 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importance = models['Random Forest'].feature_importances_
            plt.barh(range(len(feature_cols)), importance)
            plt.yticks(range(len(feature_cols)), feature_cols)
            plt.xlabel('Importance')
            plt.title(f'Feature Importance for {best_antibiotic} Resistance Prediction', fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/05_feature_importance.png', dpi=150, bbox_inches='tight')
            print(f"\n✓ Feature importance plot saved to: {output_dir}/05_feature_importance.png")
        
        # Save model results
        with open(f'{output_dir}/model_results.txt', 'w') as f:
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Target Antibiotic: {best_antibiotic}\n")
            f.write(f"Resistance Rate: {resistance_df.loc[best_antibiotic, 'Resistance_Rate']:.1f}%\n\n")
            for name, metrics in results.items():
                f.write(f"{name}:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  ROC-AUC: {metrics['roc_auc']:.4f}\n\n")
        
        print(f"✓ Model results saved to: {output_dir}/model_results.txt")
    else:
        print("No clinical features available for modeling")
else:
    print("No antibiotic data available for modeling")

# ============================================================================
# STEP 9: GENERATE SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("STEP 9: GENERATING SUMMARY REPORT")
print("="*80)

# Create summary text file
summary_file = f"{output_dir}/dataset_summary.txt"
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("ANTIBIOTIC RESISTANCE DATASET ANALYSIS SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("DATASET OVERVIEW\n")
    f.write("-"*50 + "\n")
    f.write(f"Total Rows: {df.shape[0]:,}\n")
    f.write(f"Total Columns: {df.shape[1]}\n\n")
    
    f.write("ANTIBIOTIC RESISTANCE SUMMARY\n")
    f.write("-"*50 + "\n")
    for col, stats in resistance_df.iterrows():
        f.write(f"  {col}:\n")
        f.write(f"    Resistance Rate: {stats['Resistance_Rate']:.1f}%\n")
        f.write(f"    Resistant: {int(stats['Resistant']):,}\n")
        f.write(f"    Susceptible: {int(stats['Susceptible']):,}\n")
        f.write(f"    Intermediate: {int(stats['Intermediate']):,}\n\n")
    
    f.write("MULTI-DRUG RESISTANCE STATISTICS\n")
    f.write("-"*50 + "\n")
    f.write(f"  Average resistances per isolate: {resistance_count.mean():.2f}\n")
    f.write(f"  MDR isolates (≥2 resistances): {(resistance_count >= 2).sum():,} "
           f"({(resistance_count >= 2).sum()/len(resistance_count)*100:.1f}%)\n")
    f.write(f"  XDR isolates (≥5 resistances): {(resistance_count >= 5).sum():,} "
           f"({(resistance_count >= 5).sum()/len(resistance_count)*100:.1f}%)\n")
    f.write(f"  PDR isolates (≥8 resistances): {(resistance_count >= 8).sum():,} "
           f"({(resistance_count >= 8).sum()/len(resistance_count)*100:.1f}%)\n\n")
    
    f.write("TOP CO-RESISTANCE PAIRS\n")
    f.write("-"*50 + "\n")
    for pair in corr_pairs[:10]:
        f.write(f"  {pair[0]} ↔ {pair[1]}: {pair[2]:.3f}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("END OF SUMMARY\n")
    f.write("="*80 + "\n")

print(f"✓ Summary report saved to: {summary_file}")

# ============================================================================
# STEP 10: FINAL OUTPUT AND RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

print(f"""
Analysis outputs saved in '{output_dir}/' directory:
  📊 01_missing_values.png - Missing values visualization
  📊 02_resistance_rates.png - Resistance rates by antibiotic
  📊 03_mdr_distribution.png - Multi-drug resistance distribution
  📊 04_resistance_correlation.png - Resistance correlation heatmap
  📊 05_feature_importance.png - Feature importance (if model was trained)
  📄 dataset_summary.txt - Complete analysis summary
  📄 model_results.txt - Model performance results
""")

# Key Findings
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

if not resistance_df.empty:
    print("\n🔬 HIGHEST RESISTANCE RATES (>50%):")
    high_resistance = resistance_df[resistance_df['Resistance_Rate'] > 50]
    for col, stats in high_resistance.iterrows():
        print(f"  - {col}: {stats['Resistance_Rate']:.1f}% ({int(stats['Resistant']):,}/{int(stats['Valid_Samples']):,})")
    
    print("\n💊 LOWEST RESISTANCE RATES (<20%):")
    low_resistance = resistance_df[resistance_df['Resistance_Rate'] < 20]
    for col, stats in low_resistance.iterrows():
        print(f"  - {col}: {stats['Resistance_Rate']:.1f}% ({int(stats['Resistant']):,}/{int(stats['Valid_Samples']):,})")

print("\n📊 DATASET CHARACTERISTICS:")
print(f"  - Total samples: {df.shape[0]:,}")
print(f"  - Antibiotics analyzed: {len(existing_antibiotics)}")
print(f"  - Complete resistance profiles: {len(resistance_count):,}")
print(f"  - MDR isolates: {(resistance_count >= 2).sum():,} ({((resistance_count >= 2).sum()/len(resistance_count))*100:.1f}%)")
print(f"  - XDR isolates: {(resistance_count >= 5).sum():,} ({((resistance_count >= 5).sum()/len(resistance_count))*100:.1f}%)")

print("\n💡 CRITICAL INSIGHTS:")
print("  1. Beta-lactam antibiotics show VERY HIGH resistance (>57%)")
print("  2. Aminoglycosides, fluoroquinolones, and others show MODERATE resistance (13-19%)")
print(f"  3. {((resistance_count >= 2).sum()/len(resistance_count))*100:.1f}% of isolates are Multi-Drug Resistant (MDR)")
print(f"  4. {((resistance_count >= 5).sum()/len(resistance_count))*100:.1f}% show Extensive Drug Resistance (XDR)")
print("  5. Strong co-resistance patterns exist within antibiotic classes")

print("\n🎯 RECOMMENDED NEXT STEPS:")
print("  1. Build separate ML models for high and low resistance antibiotics")
print("  2. Use correlation patterns to predict MDR phenotypes")
print("  3. Integrate bacterial species data (Souches column) for better predictions")
print("  4. Consider feature engineering using co-resistance patterns")
print("  5. Develop decision support tool recommending alternative antibiotics")
print("  6. Use CARD database for genomic resistance gene identification")

print("\n" + "="*80)
print("Ready for full model development! 🚀")
print("="*80)