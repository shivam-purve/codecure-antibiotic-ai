import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import os
import re

# Create directories to store outputs
os.makedirs('artifacts', exist_ok=True)
os.makedirs('data/visualizations', exist_ok=True)

# Load Dataset
print("Loading data...")
try:
    df = pd.read_csv('data/Bacteria_dataset_Multiresictance.csv')
except FileNotFoundError:
    df = pd.read_csv('../data/Bacteria_dataset_Multiresictance.csv')

antibiotics = ['AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN', 
               'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole', 'Furanes', 'colistine']

# --- 1. Resistance Distribution Plot ---
print("Generating Resistance Distribution Plot...")
df_dist = pd.melt(df, value_vars=[c for c in antibiotics if c in df.columns], var_name='Antibiotic', value_name='Status')
df_dist['Status'] = df_dist['Status'].astype(str).str.lower().str.strip()
val_map = {'r': 'Resistant', 'resistant': 'Resistant', 's': 'Susceptible', 'susceptible': 'Susceptible'}
df_dist['Status'] = df_dist['Status'].map(val_map)
df_dist = df_dist.dropna(subset=['Status'])

plt.figure(figsize=(14, 7))
# Create clean countplot
sns.countplot(data=df_dist, x='Antibiotic', hue='Status', palette={'Resistant': '#e74c3c', 'Susceptible': '#2ecc71'}, edgecolor='black')
plt.title('Overall Resistance vs Susceptibility Patterns per Antibiotic', fontsize=16, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('Number of Clinical Strains', fontsize=12, fontweight='bold')
plt.xlabel('Antibiotic Tested', fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Outcome', title_fontsize='11', fontsize='10', frameon=True, shadow=True)
plt.tight_layout()
plt.savefig('artifacts/resistance_distribution.png', dpi=300)
plt.savefig('data/visualizations/resistance_distribution.png', dpi=300)
plt.close()


# --- 2. Heatmap of Bacteria vs Antibiotic Resistance ---
print("Generating Bacteria Heatmap...")
def clean_bacteria(x):
    """Normalize messy bacteria text to standard species strings."""
    if pd.isna(x): return 'Unknown'
    val = re.sub(r'^S\d+\s+', '', str(x).strip()) # Remove 'S290 ' prefix
    # Explicit typographical corrections found in dataset
    corrections = {
        'E.coi': 'Escherichia coli', 'E.cli': 'Escherichia coli', 'E. coli': 'Escherichia coli', 'E.coi': 'Escherichia coli',
        'Klbsiella': 'Klebsiella pneumoniae', 'Klebsie.lla': 'Klebsiella pneumoniae', 'Klpsiella': 'Klebsiella pneumoniae',
        'Prot.eus': 'Proteus mirabilis', 'Proeus': 'Proteus mirabilis', 'Protus': 'Proteus mirabilis',
        'Enteobacteria': 'Enterobacteria', 'Enter.bacteria': 'Enterobacteria'
    }
    for bad, good in corrections.items():
        if bad in val:
            val = val.replace(bad, good)
    return val

df['Clean_Bacteria'] = df['Souches'].apply(clean_bacteria)
df_heat = df.copy()

for col in antibiotics:
    if col in df_heat.columns:
        df_heat[col] = df_heat[col].astype(str).str.lower().str.strip()
        df_heat[col] = df_heat[col].replace({'r': 1, 'resistant': 1, 's': 0, 'susceptible': 0})
        df_heat[col] = pd.to_numeric(df_heat[col], errors='coerce')

# Get Top 10 strains by occurrence counts to avoid a massive illegible y-axis
top_strains = df['Clean_Bacteria'].value_counts().head(10).index
heatmap_data = df_heat[df_heat['Clean_Bacteria'].isin(top_strains)].groupby('Clean_Bacteria')[antibiotics].mean()

plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".2f", vmin=0, vmax=1,
            cbar_kws={'label': 'Resistance Rate (0=Susceptible, 1=Resistant)'},
            linewidths=0.5, linecolor='white')
plt.title('Antibiotic Resistance Rate Heatmap by Top Bacteria Strains', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Bacteria Strain', fontsize=12, fontweight='bold')
plt.xlabel('Antibiotic', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig('artifacts/bacteria_heatmap.png', dpi=300)
plt.savefig('data/visualizations/bacteria_heatmap.png', dpi=300)
plt.close()


# --- 3. Feature Importance Plot (From XGBoost) ---
print("Generating Feature Importance Plot...")
try:
    with open('model/model.pkl', 'rb') as f:
        pipeline = pickle.load(f)
        
    numeric_features = ['Age', 'Infection_Freq']
    categorical_features = ['Gender', 'Souches', 'Diabetes', 'Hypertension', 'Hospital_before']
    
    # Expose the specific names produced by our trained OneHotEncoder config
    cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_feature_names)
    
    importances = pipeline.named_steps['classifier'].feature_importances_
    
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=False).head(15) # Top 15 drivers of predictions
    
    # Standard Presentation Graphic (Seaborn/PNG)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_imp, x='Importance', y='Feature', palette='viridis', edgecolor='black')
    plt.title('Top 15 Predictive Features for Resistance (XGBoost)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Clinical Feature', fontsize=12, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('artifacts/feature_importance.png', dpi=300)
    plt.savefig('data/visualizations/feature_importance.png', dpi=300)
    plt.close()
    
    # Interactive HTML Graphic (Plotly)
    fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h', 
                 title='<b>Top 15 Predictive Features for AMX/AMP Resistance</b>',
                 color='Importance', color_continuous_scale='Viridis')
    # Make horizontal bar charts sort sensibly
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_white')
    fig.write_html('artifacts/feature_importance_interactive.html')
    fig.write_html('data/visualizations/feature_importance_interactive.html')

except FileNotFoundError:
    print("WARNING: model.pkl not found inside model/. Skipping feature importance plot.")
except Exception as e:
    print(f"Error when generating feature importance plot: {e}")

print("All plots successfully generated and saved to 'data/visualizations/' and 'artifacts/'.")
