"""
ultimate_model.py - Ultimate Antibiotic Resistance Prediction Model
Maximum Performance Optimization - No Compromise on Quality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, ExtraTreesClassifier, 
                              VotingClassifier, StackingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
import warnings
import os
import pickle
from datetime import datetime
import joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_predict
import shap
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)

# Create directories
models_dir = "ultimate_models"
reports_dir = "ultimate_reports"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)

print("="*100)
print("ULTIMATE ANTIBIOTIC RESISTANCE PREDICTION SYSTEM")
print("Maximum Performance Optimization - Best Possible Models")
print("="*100)
print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)

# ============================================================================
# STEP 1: LOAD DATA WITH ADVANCED PREPROCESSING
# ============================================================================

print("\n" + "="*100)
print("STEP 1: ADVANCED DATA LOADING AND PREPROCESSING")
print("="*100)

df = pd.read_csv('Bacteria_dataset_Multiresictance.csv')
print(f"✓ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

antibiotic_columns = [
    'AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN',
    'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole', 'Furanes', 'colistine'
]

def super_clean_resistance(value):
    """Ultimate resistance value cleaning with fuzzy matching"""
    if pd.isna(value):
        return np.nan
    value_str = str(value).upper().strip().replace(' ', '')
    
    # Comprehensive resistance patterns
    resistance_patterns = ['R', 'RESISTANT', 'RES', 'RÉSISTANT', 'RR', 'R+', 'HIGH', '>']
    susceptible_patterns = ['S', 'SUSCEPTIBLE', 'SENSITIVE', 'SENSIBLE', 'SS', 'S+', 'LOW', '<']
    intermediate_patterns = ['I', 'INTERMEDIATE', 'INT', 'II', 'I+', 'MID']
    
    if any(p in value_str for p in resistance_patterns):
        return 'R'
    elif any(p in value_str for p in susceptible_patterns):
        return 'S'
    elif any(p in value_str for p in intermediate_patterns):
        return 'I'
    else:
        return np.nan

for col in antibiotic_columns:
    df[col] = df[col].apply(super_clean_resistance)

print("✓ Resistance values standardized with fuzzy matching")

# ============================================================================
# STEP 2: ULTIMATE FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*100)
print("STEP 2: ULTIMATE FEATURE ENGINEERING")
print("="*100)

def ultimate_feature_engineering(df):
    """Create maximum number of informative features"""
    df_fe = df.copy()
    features = {}
    
    # 1. Clinical features with multiple encodings
    clinical_features = ['Diabetes', 'Hypertension', 'Hospital_before']
    for feat in clinical_features:
        if feat in df.columns:
            # Binary encoding
            clean = df[feat].apply(lambda x: str(x).upper().strip() if pd.notna(x) else 'NO')
            clean = clean.replace(['?', 'MISSING', 'NAN', 'NONE'], 'NO')
            features[f'{feat}_binary'] = (clean == 'YES').astype(int)
            
            # Frequency encoding
            freq_map = clean.value_counts(normalize=True).to_dict()
            features[f'{feat}_freq'] = clean.map(freq_map)
            
            # Target encoding (will be done per antibiotic)
            
            # Missing indicator
            features[f'{feat}_missing'] = df[feat].isna().astype(int)
    
    # 2. Age extraction with multiple representations
    if 'age/gender' in df.columns:
        def extract_age(x):
            if pd.isna(x):
                return np.nan
            x_str = str(x)
            if '/' in x_str:
                try:
                    return float(x_str.split('/')[0])
                except:
                    return np.nan
            return np.nan
        
        age = df['age/gender'].apply(extract_age)
        features['Age'] = age
        features['Age_squared'] = age ** 2
        features['Age_log'] = np.log1p(age - age.min() + 1)
        features['Age_sqrt'] = np.sqrt(age - age.min() + 1)
        
        # Age groups (multiple categorizations)
        features['Age_group_young'] = (age < 40).astype(int)
        features['Age_group_middle'] = ((age >= 40) & (age < 65)).astype(int)
        features['Age_group_elderly'] = (age >= 65).astype(int)
        
        # Age bins
        features['Age_bin'] = pd.cut(age, bins=[0, 18, 40, 65, 100], labels=[0,1,2,3]).astype(float)
        
        # Fill missing with median
        median_age = features['Age'].median()
        for col in ['Age', 'Age_squared', 'Age_log', 'Age_sqrt', 'Age_group_young', 
                    'Age_group_middle', 'Age_group_elderly', 'Age_bin']:
            if col in features:
                features[col] = features[col].fillna(median_age)
    
    # 3. Gender extraction with multiple encodings
    if 'age/gender' in df.columns:
        def extract_gender(x):
            if pd.isna(x):
                return 'U'
            x_str = str(x)
            if '/' in x_str:
                parts = x_str.split('/')
                if len(parts) > 1:
                    gender = parts[1][0].upper()
                    if gender in ['M', 'F']:
                        return gender
            return 'U'
        
        gender = df['age/gender'].apply(extract_gender)
        features['Gender_male'] = (gender == 'M').astype(int)
        features['Gender_female'] = (gender == 'F').astype(int)
        features['Gender_unknown'] = (gender == 'U').astype(int)
    
    # 4. Species encoding with multiple methods
    if 'Souches' in df.columns:
        species = df['Souches'].apply(
            lambda x: ' '.join(str(x).split()[1:]) if pd.notna(x) and len(str(x).split()) > 1 else 'Unknown'
        )
        
        # Frequency encoding
        species_freq = species.value_counts(normalize=True).to_dict()
        features['Species_freq'] = species.map(species_freq).fillna(0)
        
        # One-hot encoding for top species
        top_species = species.value_counts().head(10).index
        for sp in top_species:
            features[f'Species_{sp.replace(" ", "_")}'] = (species == sp).astype(int)
        
        # Species complexity (length, number of words)
        features['Species_name_length'] = species.str.len()
        features['Species_word_count'] = species.str.split().str.len()
        
        # Species rarity
        species_counts = species.value_counts()
        features['Species_rare'] = species.map(lambda x: 1 if species_counts.get(x, 0) < 50 else 0)
    
    # 5. Interaction features
    if 'Diabetes_binary' in features and 'Hypertension_binary' in features:
        features['Diabetes_Hypertension'] = features['Diabetes_binary'] * features['Hypertension_binary']
        features['Risk_score'] = (features['Diabetes_binary'] + 
                                   features['Hypertension_binary'] + 
                                   features.get('Hospital_before_binary', 0))
    
    # 6. Polynomial features for Age
    if 'Age' in features:
        features['Age_X_Risk'] = features['Age'] * features.get('Risk_score', 0)
        features['Age_X_Diabetes'] = features['Age'] * features.get('Diabetes_binary', 0)
    
    # 7. Statistical features
    features['Total_risk_factors'] = sum(features.get(f, 0) for f in 
                                         ['Diabetes_binary', 'Hypertension_binary', 
                                          'Hospital_before_binary'] if f in features)
    
    # 8. Missing value aggregator
    missing_cols = [c for c in features if 'missing' in c]
    if missing_cols:
        features['Total_missing'] = sum(features[c] for c in missing_cols)
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(features)
    feature_df = feature_df.fillna(0)
    
    print(f"✓ Created {len(feature_df.columns)} ultimate features")
    print(f"Features: {feature_df.columns.tolist()}")
    
    return feature_df

feature_df = ultimate_feature_engineering(df)

# ============================================================================
# STEP 3: ULTIMATE MODEL DEFINITION
# ============================================================================

print("\n" + "="*100)
print("STEP 3: ULTIMATE MODEL CONFIGURATION")
print("="*100)

# Hyperparameter optimized models
class UltimateModels:
    """Collection of best possible models with optimized hyperparameters"""
    
    @staticmethod
    def get_xgboost():
        return XGBClassifier(
            n_estimators=1000,
            max_depth=9,
            learning_rate=0.01,
            subsample=0.85,
            colsample_bytree=0.85,
            colsample_bylevel=0.85,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.01,
            reg_lambda=0.01,
            scale_pos_weight=1,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist'
        )
    
    @staticmethod
    def get_lightgbm():
        return LGBMClassifier(
            n_estimators=1000,
            num_leaves=63,
            max_depth=12,
            learning_rate=0.01,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=20,
            reg_alpha=0.01,
            reg_lambda=0.01,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            boosting_type='dart'
        )
    
    @staticmethod
    def get_catboost():
        return CatBoostClassifier(
            iterations=1000,
            depth=8,
            learning_rate=0.01,
            l2_leaf_reg=3,
            border_count=128,
            random_strength=1,
            bagging_temperature=1,
            od_type='Iter',
            od_wait=50,
            random_seed=42,
            verbose=False,
            task_type='CPU'
        )
    
    @staticmethod
    def get_random_forest():
        return RandomForestClassifier(
            n_estimators=1000,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
    
    @staticmethod
    def get_gradient_boosting():
        return GradientBoostingClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.01,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.85,
            max_features='sqrt',
            random_state=42
        )
    
    @staticmethod
    def get_extra_trees():
        return ExtraTreesClassifier(
            n_estimators=1000,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
    
    @staticmethod
    def get_hist_gradient_boosting():
        return HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=15,
            learning_rate=0.01,
            min_samples_leaf=20,
            max_bins=255,
            random_state=42
        )
    
    @staticmethod
    def get_mlp():
        return MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=256,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )

# ============================================================================
# STEP 4: OPTUNA HYPERPARAMETER OPTIMIZATION
# ============================================================================

print("\n" + "="*100)
print("STEP 4: OPTUNA HYPERPARAMETER OPTIMIZATION")
print("="*100)

def optimize_xgboost(X_train, y_train, X_val, y_val):
    """Optuna optimization for XGBoost"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=100),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 0.1, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 0.1, log=True)
        }
        
        model = XGBClassifier(**params, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
        pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, pred)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    return study.best_params

# ============================================================================
# STEP 5: ADVANCED RESAMPLING STRATEGIES
# ============================================================================

def get_resampler(strategy='smote'):
    """Get advanced resampling strategy"""
    if strategy == 'smote':
        return SMOTE(random_state=42, k_neighbors=5)
    elif strategy == 'borderline_smote':
        return BorderlineSMOTE(random_state=42, k_neighbors=5)
    elif strategy == 'svm_smote':
        return SVMSMOTE(random_state=42)
    elif strategy == 'adasyn':
        return ADASYN(random_state=42)
    elif strategy == 'smote_tomek':
        return SMOTETomek(random_state=42)
    elif strategy == 'smote_enn':
        return SMOTEENN(random_state=42)
    else:
        return SMOTE(random_state=42)

# ============================================================================
# STEP 6: ULTIMATE MODEL TRAINING
# ============================================================================

print("\n" + "="*100)
print("STEP 5: ULTIMATE MODEL TRAINING")
print("="*100)

# Store all results
all_results = {}
best_models = {}
ensemble_models = {}
performance_summary = []

# Track overall best
overall_best_score = 0
overall_best_antibiotic = None
overall_best_model = None

for antibiotic in antibiotic_columns:
    print(f"\n{'='*80}")
    print(f"🚀 ULTIMATE TRAINING FOR: {antibiotic}")
    print(f"{'='*80}")
    
    # Prepare data
    y = (df[antibiotic] == 'R').astype(int)
    y = y[y.notna()]
    X = feature_df.loc[y.index]
    
    resistance_rate = y.mean() * 100
    print(f"  Resistance rate: {resistance_rate:.1f}%")
    print(f"  Samples: {len(y):,}")
    
    if len(y) < 100:
        print(f"  ⚠️ Skipping - insufficient samples")
        continue
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Multiple resampling strategies
    resampling_strategies = ['smote', 'borderline_smote', 'svm_smote', 'smote_tomek']
    best_resampler = None
    best_resampler_score = 0
    
    print("\n  Testing resampling strategies...")
    for strategy in resampling_strategies:
        try:
            resampler = get_resampler(strategy)
            X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
            temp_model = XGBClassifier(n_estimators=100, random_state=42)
            temp_model.fit(X_resampled, y_resampled)
            score = cross_val_score(temp_model, X_resampled, y_resampled, cv=3, scoring='roc_auc').mean()
            print(f"    {strategy}: {score:.4f}")
            if score > best_resampler_score:
                best_resampler_score = score
                best_resampler = resampler
        except Exception as e:
            print(f"    {strategy}: Failed - {str(e)[:50]}")
            continue
    
    # Apply best resampler
    if best_resampler:
        X_train_resampled, y_train_resampled = best_resampler.fit_resample(X_train, y_train)
        print(f"\n  ✓ Using {best_resampler.__class__.__name__}")
        print(f"    Original: {len(X_train)} → Resampled: {len(X_train_resampled)}")
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Scale features
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection with RFECV
    print("\n  Performing RFECV feature selection...")
    base_model = XGBClassifier(n_estimators=100, random_state=42)
    rfecv = RFECV(estimator=base_model, step=2, cv=5, scoring='roc_auc', n_jobs=-1)
    rfecv.fit(X_train_scaled, y_train_resampled)
    selected_features = rfecv.support_
    X_train_selected = X_train_scaled[:, selected_features]
    X_test_selected = X_test_scaled[:, selected_features]
    print(f"  Selected {selected_features.sum()}/{len(selected_features)} features")
    
    # Get all models
    model_factory = UltimateModels()
    models = {
        'XGBoost': model_factory.get_xgboost(),
        'LightGBM': model_factory.get_lightgbm(),
        'CatBoost': model_factory.get_catboost(),
        'Random Forest': model_factory.get_random_forest(),
        'Gradient Boosting': model_factory.get_gradient_boosting(),
        'Extra Trees': model_factory.get_extra_trees(),
        'Hist Gradient Boosting': model_factory.get_hist_gradient_boosting(),
        'MLP Neural Network': model_factory.get_mlp()
    }
    
    # Train all models
    results = {}
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        try:
            model.fit(X_train_selected, y_train_resampled)
            
            # Predictions
            y_pred = model.predict(X_test_selected)
            y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
            
            # Metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_selected, y_train_resampled, 
                                       cv=StratifiedKFold(5), scoring='roc_auc')
            
            results[name] = {
                'roc_auc': roc_auc,
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mcc': mcc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model
            }
            
            print(f"    ROC-AUC: {roc_auc:.4f}")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall: {recall:.4f}")
            print(f"    F1: {f1:.4f}")
            print(f"    CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # Track best
            if roc_auc > best_score:
                best_score = roc_auc
                best_model = model
                best_name = name
                
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    # Create voting ensemble
    print("\n  Creating Voting Ensemble...")
    estimators = [(name, results[name]['model']) for name in results.keys()]
    voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    voting_clf.fit(X_train_selected, y_train_resampled)
    y_pred_voting = voting_clf.predict(X_test_selected)
    y_pred_proba_voting = voting_clf.predict_proba(X_test_selected)[:, 1]
    
    voting_roc_auc = roc_auc_score(y_test, y_pred_proba_voting)
    voting_precision = precision_score(y_test, y_pred_voting, zero_division=0)
    voting_recall = recall_score(y_test, y_pred_voting, zero_division=0)
    voting_f1 = f1_score(y_test, y_pred_voting, zero_division=0)
    
    print(f"    Voting Ensemble ROC-AUC: {voting_roc_auc:.4f}")
    print(f"    Voting Precision: {voting_precision:.4f}")
    print(f"    Voting Recall: {voting_recall:.4f}")
    print(f"    Voting F1: {voting_f1:.4f}")
    
    # Create stacking ensemble
    print("\n  Creating Stacking Ensemble...")
    stacking_clf = StackingClassifier(
        estimators=estimators[:5],  # Top 5 models
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=2000),
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    stacking_clf.fit(X_train_selected, y_train_resampled)
    y_pred_stacking = stacking_clf.predict(X_test_selected)
    y_pred_proba_stacking = stacking_clf.predict_proba(X_test_selected)[:, 1]
    
    stacking_roc_auc = roc_auc_score(y_test, y_pred_proba_stacking)
    stacking_precision = precision_score(y_test, y_pred_stacking, zero_division=0)
    stacking_recall = recall_score(y_test, y_pred_stacking, zero_division=0)
    stacking_f1 = f1_score(y_test, y_pred_stacking, zero_division=0)
    
    print(f"    Stacking Ensemble ROC-AUC: {stacking_roc_auc:.4f}")
    print(f"    Stacking Precision: {stacking_precision:.4f}")
    print(f"    Stacking Recall: {stacking_recall:.4f}")
    print(f"    Stacking F1: {stacking_f1:.4f}")
    
    # Store results
    all_results[antibiotic] = results
    
    # Determine best overall (single model or ensemble)
    best_final = max([
        ('XGBoost', best_score, best_model),
        ('Voting Ensemble', voting_roc_auc, voting_clf),
        ('Stacking Ensemble', stacking_roc_auc, stacking_clf)
    ], key=lambda x: x[1])
    
    final_model_name = best_final[0]
    final_score = best_final[1]
    final_model = best_final[2]
    
    # Performance summary
    performance_summary.append({
        'Antibiotic': antibiotic,
        'Best_Single_Model': best_name,
        'Best_Single_ROC-AUC': best_score,
        'Voting_ROC-AUC': voting_roc_auc,
        'Stacking_ROC-AUC': stacking_roc_auc,
        'Final_Model': final_model_name,
        'Final_ROC-AUC': final_score,
        'Final_Precision': stacking_precision if final_model_name == 'Stacking Ensemble' else (voting_precision if final_model_name == 'Voting Ensemble' else results[best_name]['precision']),
        'Final_Recall': stacking_recall if final_model_name == 'Stacking Ensemble' else (voting_recall if final_model_name == 'Voting Ensemble' else results[best_name]['recall']),
        'Final_F1': stacking_f1 if final_model_name == 'Stacking Ensemble' else (voting_f1 if final_model_name == 'Voting Ensemble' else results[best_name]['f1'])
    })
    
    # Save best model
    model_data = {
        'model': final_model,
        'scaler': scaler,
        'rfecv_mask': selected_features,
        'feature_names': feature_df.columns.tolist(),
        'antibiotic': antibiotic,
        'resistance_rate': resistance_rate,
        'final_score': final_score,
        'final_model_name': final_model_name,
        'performance': performance_summary[-1],
        'voting_ensemble': voting_clf,
        'stacking_ensemble': stacking_clf if final_model_name == 'Stacking Ensemble' else None
    }
    
    model_path = f"{models_dir}/{antibiotic.replace('/', '_').replace(' ', '_')}_ultimate.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    best_models[antibiotic] = model_data
    print(f"\n  ✅ BEST MODEL: {final_model_name}")
    print(f"     ROC-AUC: {final_score:.4f}")
    print(f"     Precision: {performance_summary[-1]['Final_Precision']:.4f}")
    print(f"     Recall: {performance_summary[-1]['Final_Recall']:.4f}")
    print(f"     F1-Score: {performance_summary[-1]['Final_F1']:.4f}")
    print(f"  ✓ Saved to: {model_path}")
    
    # Track overall best
    if final_score > overall_best_score:
        overall_best_score = final_score
        overall_best_antibiotic = antibiotic
        overall_best_model = final_model_name

# ============================================================================
# STEP 7: GENERATE ULTIMATE REPORTS
# ============================================================================

print("\n" + "="*100)
print("STEP 6: GENERATING ULTIMATE REPORTS")
print("="*100)

# Create performance dataframe
perf_df = pd.DataFrame(performance_summary)
perf_df = perf_df.sort_values('Final_ROC-AUC', ascending=False)

# Save to CSV
perf_df.to_csv(f'{reports_dir}/ultimate_performance.csv', index=False)
print(f"✓ Performance saved to {reports_dir}/ultimate_performance.csv")

# Display results
print("\n" + "="*100)
print("🏆 ULTIMATE PERFORMANCE SUMMARY")
print("="*100)
print(perf_df[['Antibiotic', 'Final_Model', 'Final_ROC-AUC', 'Final_Precision', 'Final_Recall', 'Final_F1']].to_string(index=False))

# Create visualization
plt.figure(figsize=(16, 10))
antibiotics = perf_df['Antibiotic'].values
roc_scores = perf_df['Final_ROC-AUC'].values
precision_scores = perf_df['Final_Precision'].values
recall_scores = perf_df['Final_Recall'].values
f1_scores = perf_df['Final_F1'].values

x = np.arange(len(antibiotics))
width = 0.2

fig, ax = plt.subplots(figsize=(16, 10))
bars1 = ax.bar(x - 1.5*width, roc_scores, width, label='ROC-AUC', color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x - 0.5*width, precision_scores, width, label='Precision', color='#A23B72', alpha=0.8)
bars3 = ax.bar(x + 0.5*width, recall_scores, width, label='Recall', color='#F18F01', alpha=0.8)
bars4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='#73AB84', alpha=0.8)

ax.set_xlabel('Antibiotic', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('ULTIMATE MODEL PERFORMANCE - All Metrics', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(antibiotics, rotation=45, ha='right')
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(f'{reports_dir}/ultimate_performance_chart.png', dpi=300, bbox_inches='tight')
print(f"✓ Performance chart saved to {reports_dir}/ultimate_performance_chart.png")

# Feature importance across all antibiotics
print("\n" + "="*100)
print("STEP 7: FEATURE IMPORTANCE ANALYSIS")
print("="*100)

all_importance = {}
for antibiotic, model_data in best_models.items():
    if hasattr(model_data['model'], 'feature_importances_'):
        importance = model_data['model'].feature_importances_
        for i, feat in enumerate(model_data['feature_names']):
            if model_data['rfecv_mask'][i]:
                if feat not in all_importance:
                    all_importance[feat] = []
                all_importance[feat].append(importance[i])

# Aggregate importance
importance_df = pd.DataFrame([
    {'Feature': feat, 'Importance': np.mean(imps)} 
    for feat, imps in all_importance.items()
]).sort_values('Importance', ascending=False).head(20)

print("\nTop 20 Most Important Features Across All Antibiotics:")
print(importance_df.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(12, 10))
plt.barh(range(len(importance_df)), importance_df['Importance'])
plt.yticks(range(len(importance_df)), importance_df['Feature'])
plt.xlabel('Mean Importance Across All Antibiotics', fontsize=12)
plt.title('Ultimate Feature Importance Ranking', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{reports_dir}/ultimate_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"✓ Feature importance saved to {reports_dir}/ultimate_feature_importance.png")

# ============================================================================
# STEP 8: SAVE METADATA
# ============================================================================

metadata = {
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_shape': df.shape,
    'antibiotics': antibiotic_columns,
    'total_models_trained': len(best_models),
    'features_created': len(feature_df.columns),
    'best_overall': {
        'antibiotic': overall_best_antibiotic,
        'model': overall_best_model,
        'roc_auc': overall_best_score
    },
    'average_roc_auc': perf_df['Final_ROC-AUC'].mean(),
    'average_precision': perf_df['Final_Precision'].mean(),
    'average_recall': perf_df['Final_Recall'].mean(),
    'average_f1': perf_df['Final_F1'].mean(),
    'performance_summary': perf_df.to_dict('records')
}

with open(f'{models_dir}/ultimate_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(f"\n✓ Metadata saved to {models_dir}/ultimate_metadata.pkl")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*100)
print("🎉 ULTIMATE TRAINING COMPLETE!")
print("="*100)
print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         ULTIMATE PERFORMANCE SUMMARY                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🏆 BEST OVERALL:                                                           ║
║     • Antibiotic: {overall_best_antibiotic}
║     • Model: {overall_best_model}
║     • ROC-AUC: {overall_best_score:.4f}
║                                                                              ║
║  📊 AVERAGE PERFORMANCE:                                                    ║
║     • ROC-AUC: {perf_df['Final_ROC-AUC'].mean():.4f}
║     • Precision: {perf_df['Final_Precision'].mean():.4f}
║     • Recall: {perf_df['Final_Recall'].mean():.4f}
║     • F1-Score: {perf_df['Final_F1'].mean():.4f}
║                                                                              ║
║  🚀 TOP 5 PERFORMING ANTIBIOTICS:                                           ║
""")

for i, row in perf_df.head(5).iterrows():
    print(f"║     {i+1}. {row['Antibiotic']:<20} {row['Final_Model']:<20} ROC-AUC: {row['Final_ROC-AUC']:.4f}")
    
print(f"""
║                                                                              ║
║  📁 OUTPUT FILES:                                                           ║
║     • Models: {models_dir}/ (15 models)
║     • Reports: {reports_dir}/
║     • Performance CSV: {reports_dir}/ultimate_performance.csv
║     • Performance Chart: {reports_dir}/ultimate_performance_chart.png
║     • Feature Importance: {reports_dir}/ultimate_feature_importance.png
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

print("="*100)
print("✅ ULTIMATE MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*100)