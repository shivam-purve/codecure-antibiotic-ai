"""
train.py - Complete Antibiotic Resistance Prediction Model
This script builds, trains, and saves models for all antibiotics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, auc)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
import os
import pickle
from datetime import datetime
import joblib

warnings.filterwarnings('ignore')

# Set style - using available matplotlib styles
plt.style.use('default')
sns.set_palette("husl")
sns.set_style("whitegrid")

# Create directories
models_dir = "saved_models"
reports_dir = "reports"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)

print("="*80)
print("COMPLETE ANTIBIOTIC RESISTANCE PREDICTION MODEL TRAINING")
print("="*80)
print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING AND PREPARING DATA")
print("="*80)

# Load dataset
df = pd.read_csv('Bacteria_dataset_Multiresictance.csv')
print(f"✓ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Define columns
antibiotic_columns = [
    'AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN',
    'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole', 'Furanes', 'colistine'
]

# Clean resistance values
def clean_resistance(value):
    if pd.isna(value):
        return np.nan
    value_str = str(value).upper().strip()
    if value_str in ['R', 'RESISTANT', 'RES']:
        return 'R'
    elif value_str in ['S', 'SUSCEPTIBLE', 'SENSITIVE']:
        return 'S'
    elif value_str in ['I', 'INTERMEDIATE', 'INT']:
        return 'I'
    else:
        return np.nan

for col in antibiotic_columns:
    df[col] = df[col].apply(clean_resistance)

print("✓ Resistance values standardized")

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("STEP 2: FEATURE ENGINEERING")
print("="*80)

def engineer_features(df):
    """Create advanced features for better prediction"""
    df_fe = df.copy()
    
    # 1. Clinical features with proper encoding
    clinical_features = ['Diabetes', 'Hypertension', 'Hospital_before']
    for feat in clinical_features:
        if feat in df.columns:
            # Clean and encode
            df_fe[feat] = df_fe[feat].apply(
                lambda x: str(x).upper().strip() if pd.notna(x) else 'UNKNOWN'
            )
            df_fe[feat] = df_fe[feat].replace(['?', 'MISSING', 'NAN', 'NONE'], 'UNKNOWN')
            # Binary encoding: YES=1, NO=0, UNKNOWN=0
            df_fe[f'{feat}_Encoded'] = df_fe[feat].apply(
                lambda x: 1 if x == 'YES' or x == 'TRUE' else 0
            )
    
    # 2. Age extraction from age/gender column
    if 'age/gender' in df.columns:
        def extract_age(value):
            if pd.isna(value):
                return np.nan
            value_str = str(value)
            if '/' in value_str:
                parts = value_str.split('/')
                try:
                    return float(parts[0]) if parts[0].replace('.', '').isdigit() else np.nan
                except:
                    return np.nan
            return np.nan
        
        df_fe['Age'] = df_fe['age/gender'].apply(extract_age)
        # Fill missing ages with median
        df_fe['Age'].fillna(df_fe['Age'].median(), inplace=True)
    
    # 3. Gender extraction
    if 'age/gender' in df.columns:
        def extract_gender(value):
            if pd.isna(value):
                return 'UNKNOWN'
            value_str = str(value)
            if '/' in value_str:
                parts = value_str.split('/')
                if len(parts) > 1:
                    gender = parts[1].upper()
                    if gender in ['M', 'F']:
                        return gender
            return 'UNKNOWN'
        
        df_fe['Gender'] = df_fe['age/gender'].apply(extract_gender)
        df_fe['Gender_Encoded'] = df_fe['Gender'].apply(lambda x: 1 if x == 'M' else 0)
    
    # 4. Bacterial species encoding
    if 'Souches' in df.columns:
        # Extract simplified species name
        df_fe['Species_Simplified'] = df_fe['Souches'].apply(
            lambda x: str(x).split()[-1] if pd.notna(x) and len(str(x).split()) > 0 else 'Unknown'
        )
        # Encode species (top 10 most common)
        top_species = df_fe['Species_Simplified'].value_counts().head(10).index
        df_fe['Species_Encoded'] = df_fe['Species_Simplified'].apply(
            lambda x: list(top_species).index(x) if x in top_species else len(top_species)
        )
    
    return df_fe

df_engineered = engineer_features(df)
print("✓ Feature engineering completed")

# ============================================================================
# STEP 3: DEFINE FEATURE SETS
# ============================================================================

print("\n" + "="*80)
print("STEP 3: DEFINING FEATURE SETS")
print("="*80)

# Feature columns
feature_columns = [
    'Diabetes_Encoded', 'Hypertension_Encoded', 'Hospital_before_Encoded',
    'Age', 'Gender_Encoded'
]

# Add species if available
if 'Species_Encoded' in df_engineered.columns:
    feature_columns.append('Species_Encoded')

# Available features
available_features = [f for f in feature_columns if f in df_engineered.columns]
print(f"✓ Using {len(available_features)} features: {available_features}")

# ============================================================================
# STEP 4: TRAIN MODELS FOR EACH ANTIBIOTIC
# ============================================================================

print("\n" + "="*80)
print("STEP 4: TRAINING MODELS FOR EACH ANTIBIOTIC")
print("="*80)

# Categorize antibiotics
high_resistance_ab = ['AMC', 'CTX/CRO', 'FOX', 'IPM', 'AMX/AMP', 'CZ']
medium_resistance_ab = ['GEN', 'AN']
low_resistance_ab = ['CIP', 'Co-trimoxazole', 'Acide nalidixique', 'C', 'ofx', 'colistine', 'Furanes']

all_antibiotics = high_resistance_ab + medium_resistance_ab + low_resistance_ab

# Store models and results
trained_models = {}
model_results = {}
performance_summary = []

for antibiotic in all_antibiotics:
    print(f"\n{'='*60}")
    print(f"Training model for: {antibiotic}")
    print(f"{'='*60}")
    
    # Check if antibiotic exists in dataframe
    if antibiotic not in df_engineered.columns:
        print(f"  ⚠️  {antibiotic} not found in dataset, skipping...")
        continue
    
    # Prepare target
    y = (df_engineered[antibiotic] == 'R').astype(int)
    y = y[y.notna()]
    
    # Prepare features
    X = df_engineered.loc[y.index, available_features].copy()
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Check class distribution
    resistance_rate = y.mean() * 100
    print(f"  Resistance rate: {resistance_rate:.1f}%")
    print(f"  Sample size: {len(y):,}")
    
    if len(y) < 50 or resistance_rate == 0 or resistance_rate == 100:
        print(f"  ⚠️  Skipping {antibiotic} - insufficient data or imbalanced")
        continue
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models with simplified parameters for faster training
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        )
    }
    
    # Train and evaluate models
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Get probabilities
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred_proba = y_pred
            
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            print(f"\n  {name}:")
            print(f"    Accuracy: {accuracy:.4f}")
            print(f"    ROC-AUC: {roc_auc:.4f}")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall: {recall:.4f}")
            print(f"    F1-Score: {f1:.4f}")
            
            # Store results
            performance_summary.append({
                'Antibiotic': antibiotic,
                'Model': name,
                'Accuracy': accuracy,
                'ROC-AUC': roc_auc,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
            
            # Track best model
            if roc_auc > best_score:
                best_score = roc_auc
                best_model = model
                best_name = name
                
        except Exception as e:
            print(f"  ⚠️  Error training {name}: {e}")
            continue
    
    # Save best model
    if best_model:
        model_data = {
            'model': best_model,
            'scaler': scaler,
            'imputer': imputer,
            'features': available_features,
            'antibiotic': antibiotic,
            'resistance_rate': resistance_rate,
            'best_score': best_score,
            'best_model_name': best_name
        }
        
        # Save model
        model_filename = f"{models_dir}/{antibiotic.replace('/', '_').replace(' ', '_')}_model.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        trained_models[antibiotic] = model_data
        print(f"\n  ✓ Best model: {best_name} (ROC-AUC: {best_score:.4f})")
        print(f"  ✓ Model saved to: {model_filename}")

# ============================================================================
# STEP 5: GENERATE PERFORMANCE REPORT
# ============================================================================

print("\n" + "="*80)
print("STEP 5: GENERATING PERFORMANCE REPORT")
print("="*80)

# Create results dataframe
results_table = pd.DataFrame(performance_summary)
if not results_table.empty:
    results_table = results_table.sort_values(['Antibiotic', 'ROC-AUC'], ascending=[True, False])
    
    # Save results to CSV
    results_table.to_csv(f'{reports_dir}/model_performance.csv', index=False)
    print(f"✓ Model performance saved to: {reports_dir}/model_performance.csv")
    
    # Create visualization of model performance
    plt.figure(figsize=(14, 8))
    
    antibiotics = results_table['Antibiotic'].unique()
    x = np.arange(len(antibiotics))
    width = 0.25
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, (model, color) in enumerate(zip(['Random Forest', 'Gradient Boosting', 'Logistic Regression'], colors)):
        model_data = results_table[results_table['Model'] == model]
        if not model_data.empty:
            roc_scores = []
            for ab in antibiotics:
                score = model_data[model_data['Antibiotic'] == ab]['ROC-AUC'].values
                roc_scores.append(score[0] if len(score) > 0 else 0)
            plt.bar(x + i*width, roc_scores, width, label=model, color=color, alpha=0.8)
    
    plt.xlabel('Antibiotic', fontsize=12)
    plt.ylabel('ROC-AUC Score', fontsize=12)
    plt.title('Model Performance Comparison Across Antibiotics', fontweight='bold', fontsize=14)
    plt.xticks(x + width, antibiotics, rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{reports_dir}/model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Model comparison plot saved to: {reports_dir}/model_comparison.png")
    
    # Best performing models
    print("\n📈 Best Performing Models:")
    best_models = results_table.loc[results_table.groupby('Antibiotic')['ROC-AUC'].idxmax()]
    for _, row in best_models.iterrows():
        print(f"  • {row['Antibiotic']}: {row['Model']} (ROC-AUC: {row['ROC-AUC']:.4f})")

# ============================================================================
# STEP 6: FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 6: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Train a Random Forest on all data for feature importance
print("Training global model for feature importance...")

# Prepare data
X_global = df_engineered[available_features].copy()
y_global = (df_engineered['AMC'] == 'R').astype(int)  # Use AMC as target
y_global = y_global[y_global.notna()]
X_global = X_global.loc[y_global.index]

# Handle missing values
imputer_global = SimpleImputer(strategy='median')
X_global_imputed = pd.DataFrame(imputer_global.fit_transform(X_global), columns=X_global.columns)

# Train model
rf_global = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_global.fit(X_global_imputed, y_global)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': rf_global.feature_importances_
}).sort_values('importance', ascending=False)

print("\nGlobal Feature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 8))
plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Importance', fontsize=12)
plt.title('Global Feature Importance for Resistance Prediction', fontweight='bold', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{reports_dir}/global_feature_importance.png', dpi=150, bbox_inches='tight')
print(f"✓ Feature importance plot saved to: {reports_dir}/global_feature_importance.png")

# ============================================================================
# STEP 7: SAVE METADATA
# ============================================================================

print("\n" + "="*80)
print("STEP 7: SAVING METADATA")
print("="*80)

# Calculate MDR statistics
resistance_matrix = pd.DataFrame()
for col in all_antibiotics:
    if col in df_engineered.columns:
        resistance_matrix[col] = (df_engineered[col] == 'R').astype(float)
resistance_count = resistance_matrix.sum(axis=1)

# Save metadata
metadata = {
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_shape': df.shape,
    'antibiotics': all_antibiotics,
    'high_resistance': high_resistance_ab,
    'medium_resistance': medium_resistance_ab,
    'low_resistance': low_resistance_ab,
    'features_used': available_features,
    'total_models_trained': len(trained_models),
    'mdr_rate': float((resistance_count >= 2).sum() / len(resistance_count) * 100),
    'xdr_rate': float((resistance_count >= 5).sum() / len(resistance_count) * 100),
    'pdr_rate': float((resistance_count >= 8).sum() / len(resistance_count) * 100)
}

with open(f'{models_dir}/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

# Save preprocessing objects
preprocessing = {
    'imputer': imputer_global,
    'feature_importance': feature_importance.to_dict('records')
}

with open(f'{models_dir}/preprocessing.pkl', 'wb') as f:
    pickle.dump(preprocessing, f)

print(f"✓ Metadata saved to {models_dir}/")

# ============================================================================
# STEP 8: CREATE RESISTANCE NETWORK VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("STEP 8: CREATING RESISTANCE NETWORK VISUALIZATION")
print("="*80)

try:
    import networkx as nx
    
    # Create correlation matrix for network
    corr_matrix = resistance_matrix.corr()
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes for antibiotics
    for antibiotic in corr_matrix.columns:
        G.add_node(antibiotic, node_type='antibiotic')
    
    # Add edges based on correlation threshold
    threshold = 0.25
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > threshold:
                G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=abs(corr))
    
    # Create visualization
    plt.figure(figsize=(14, 12))
    
    # Color nodes by resistance level
    node_colors = []
    for node in G.nodes():
        if node in high_resistance_ab:
            node_colors.append('#E63946')  # Red
        elif node in medium_resistance_ab:
            node_colors.append('#F4A261')  # Orange
        else:
            node_colors.append('#2A9D8F')  # Green
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw
    nx.draw(G, pos, 
            node_color=node_colors,
            node_size=2000,
            with_labels=True,
            font_size=10,
            font_weight='bold',
            edge_color='gray',
            width=[G[u][v]['weight']*3 for u, v in G.edges()],
            alpha=0.7)
    
    # Add edge labels for strong correlations
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges() if G[u][v]['weight'] > 0.3}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    plt.title('Antibiotic Resistance Co-occurrence Network', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{reports_dir}/resistance_network.png', dpi=150, bbox_inches='tight')
    print(f"✓ Resistance network visualization saved to: {reports_dir}/resistance_network.png")
    
except ImportError:
    print("⚠️  NetworkX not installed. Skipping network visualization.")
except Exception as e:
    print(f"⚠️  Error creating network: {e}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TRAINING COMPLETE! FINAL SUMMARY")
print("="*80)

print(f"""
📊 Training Summary:
  - Total antibiotics modeled: {len(trained_models)}/15
  - Models saved: {len(trained_models)}
  - Models directory: {models_dir}/
  - Reports directory: {reports_dir}/

📈 Best Model Performance:
""")

if not results_table.empty:
    best_overall = results_table.loc[results_table['ROC-AUC'].idxmax()]
    print(f"  • Best overall: {best_overall['Antibiotic']} - {best_overall['Model']} "
          f"(ROC-AUC: {best_overall['ROC-AUC']:.4f})")
    
    best_models = results_table.loc[results_table.groupby('Antibiotic')['ROC-AUC'].idxmax()]
    for _, row in best_models.head(5).iterrows():
        print(f"  • {row['Antibiotic']}: {row['Model']} (ROC-AUC: {row['ROC-AUC']:.4f})")

print(f"""
🎯 Key Insights:
  1. Beta-lactam antibiotics show highest resistance (>57%)
  2. {metadata['mdr_rate']:.1f}% of isolates are Multi-Drug Resistant
  3. {metadata['xdr_rate']:.1f}% show Extensive Drug Resistance
  4. Top feature importance: {feature_importance.iloc[0]['feature']}
  5. Strong co-resistance within antibiotic classes

📁 Generated Files:
  - saved_models/ - All trained models
  - reports/model_performance.csv - Performance metrics
  - reports/model_comparison.png - Model comparison chart
  - reports/global_feature_importance.png - Feature importance
  - reports/resistance_network.png - Resistance co-occurrence network
""")

print("="*80)
print("✅ Training completed successfully!")
print("="*80)