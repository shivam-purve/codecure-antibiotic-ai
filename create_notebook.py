import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Antibiotic Resistance Analysis\n",
    "Visualizing resistance patterns, bacteria strains, and model feature importances."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import plotly.express as px\n",
    "import pickle\n",
    "import re\n",
    "\n",
    "# Load Dataset\n",
    "df = pd.read_csv('../data/Bacteria_dataset_Multiresictance.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Resistance Distribution Plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "antibiotics = ['AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN', \n",
    "               'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole', 'Furanes', 'colistine']\n",
    "\n",
    "df_dist = pd.melt(df, value_vars=[c for c in antibiotics if c in df.columns], var_name='Antibiotic', value_name='Status')\n",
    "df_dist['Status'] = df_dist['Status'].astype(str).str.lower().str.strip()\n",
    "val_map = {'r': 'Resistant', 'resistant': 'Resistant', 's': 'Susceptible', 'susceptible': 'Susceptible'}\n",
    "df_dist['Status'] = df_dist['Status'].map(val_map)\n",
    "df_dist = df_dist.dropna(subset=['Status'])\n",
    "\n",
    "plt.figure(figsize=(14, 7))\n",
    "sns.countplot(data=df_dist, x='Antibiotic', hue='Status', palette={'Resistant': '#e74c3c', 'Susceptible': '#2ecc71'}, edgecolor='black')\n",
    "plt.title('Overall Resistance vs Susceptibility Patterns per Antibiotic', fontsize=16, fontweight='bold', pad=15)\n",
    "plt.xticks(rotation=45, ha='right', fontsize=10)\n",
    "plt.ylabel('Number of Clinical Strains', fontsize=12, fontweight='bold')\n",
    "plt.xlabel('Antibiotic Tested', fontsize=12, fontweight='bold')\n",
    "plt.grid(axis='y', linestyle='--', alpha=0.7)\n",
    "plt.legend(title='Outcome', title_fontsize='11', fontsize='10', frameon=True, shadow=True)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Bacteria Heatmap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_bacteria(x):\n",
    "    if pd.isna(x): return 'Unknown'\n",
    "    val = re.sub(r'^S\\d+\\s+', '', str(x).strip())\n",
    "    corrections = {\n",
    "        'E.coi': 'Escherichia coli', 'E.cli': 'Escherichia coli', 'E. coli': 'Escherichia coli',\n",
    "        'Klbsiella': 'Klebsiella pneumoniae', 'Klebsie.lla': 'Klebsiella pneumoniae', 'Klpsiella': 'Klebsiella pneumoniae',\n",
    "        'Prot.eus': 'Proteus mirabilis', 'Proeus': 'Proteus mirabilis', 'Protus': 'Proteus mirabilis',\n",
    "        'Enteobacteria': 'Enterobacteria', 'Enter.bacteria': 'Enterobacteria'\n",
    "    }\n",
    "    for bad, good in corrections.items():\n",
    "        if bad in val:\n",
    "            val = val.replace(bad, good)\n",
    "    return val\n",
    "\n",
    "df['Clean_Bacteria'] = df['Souches'].apply(clean_bacteria)\n",
    "df_heat = df.copy()\n",
    "\n",
    "for col in antibiotics:\n",
    "    if col in df_heat.columns:\n",
    "        df_heat[col] = df_heat[col].astype(str).str.lower().str.strip()\n",
    "        df_heat[col] = df_heat[col].replace({'r': 1, 'resistant': 1, 's': 0, 'susceptible': 0})\n",
    "        df_heat[col] = pd.to_numeric(df_heat[col], errors='coerce')\n",
    "\n",
    "top_strains = df['Clean_Bacteria'].value_counts().head(10).index\n",
    "heatmap_data = df_heat[df_heat['Clean_Bacteria'].isin(top_strains)].groupby('Clean_Bacteria')[antibiotics].mean()\n",
    "\n",
    "plt.figure(figsize=(14, 8))\n",
    "sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=\".2f\", vmin=0, vmax=1,\n",
    "            cbar_kws={'label': 'Resistance Rate (0=Susceptible, 1=Resistant)'},\n",
    "            linewidths=0.5, linecolor='white')\n",
    "plt.title('Antibiotic Resistance Rate Heatmap by Top Bacteria Strains', fontsize=16, fontweight='bold', pad=15)\n",
    "plt.ylabel('Bacteria Strain', fontsize=12, fontweight='bold')\n",
    "plt.xlabel('Antibiotic', fontsize=12, fontweight='bold')\n",
    "plt.xticks(rotation=45, ha='right', fontsize=10)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Feature Importance (Random Forest)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "try:\n",
    "    with open('../model/model.pkl', 'rb') as f:\n",
    "        pipeline = pickle.load(f)\n",
    "        \n",
    "    numeric_features = ['Age', 'Infection_Freq']\n",
    "    categorical_features = ['Gender', 'Souches', 'Diabetes', 'Hypertension', 'Hospital_before']\n",
    "    \n",
    "    cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']\n",
    "    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)\n",
    "    feature_names = numeric_features + list(cat_feature_names)\n",
    "    \n",
    "    importances = pipeline.named_steps['classifier'].feature_importances_\n",
    "    \n",
    "    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})\n",
    "    df_imp = df_imp.sort_values(by='Importance', ascending=False).head(15)\n",
    "    \n",
    "    plt.figure(figsize=(12, 8))\n",
    "    sns.barplot(data=df_imp, x='Importance', y='Feature', palette='viridis', edgecolor='black')\n",
    "    plt.title('Top 15 Predictive Features for Resistance (Random Forest)', fontsize=16, fontweight='bold', pad=15)\n",
    "    plt.xlabel('Feature Importance Score', fontsize=12, fontweight='bold')\n",
    "    plt.ylabel('Clinical Feature', fontsize=12, fontweight='bold')\n",
    "    plt.grid(axis='x', linestyle='--', alpha=0.5)\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    \n",
    "    fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h', \n",
    "                 title='<b>Top 15 Predictive Features for AMX/AMP Resistance</b>',\n",
    "                 color='Importance', color_continuous_scale='Viridis')\n",
    "    fig.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_white')\n",
    "    fig.show()\n",
    "    \n",
    "except FileNotFoundError:\n",
    "    print(\"Model file not found. Ensure you run src/train.py first.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('notebooks/analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
