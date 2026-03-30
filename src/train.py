"""
Training Module for CodeCure AI.

This module cleans raw clinical data, constructs an XGBoost pipeline with preprocessing,
and trains independent models for all specified antibiotics.
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

def clean_data(df, target_col):
    """Clean data focusing specifically on providing valid inputs for a target."""
    df = df.copy()
    cols_to_drop = ['ID', 'Name', 'Email', 'Address', 'Collection_Date', 'Notes']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    if 'age/gender' in df.columns:
        df[['Age', 'Gender']] = df['age/gender'].str.split('/', n=1, expand=True)
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df = df.drop(columns=['age/gender'])
    else:
        df['Age'] = np.nan
        df['Gender'] = 'unknown'
        
    if target_col not in df.columns:
        return pd.DataFrame()
        
    df[target_col] = df[target_col].astype(str).str.lower().str.strip()
    val_map = {'r': 'resistant', 'resistant': 'resistant', 's': 'susceptible', 'susceptible': 'susceptible'}
    df[target_col] = df[target_col].map(val_map)
    df = df[df[target_col].isin(['resistant', 'susceptible'])]
    df[target_col] = df[target_col].apply(lambda x: 1 if x == 'resistant' else 0) # 1=Resistant, 0=Susceptible
    
    if 'Infection_Freq' in df.columns:
        df['Infection_Freq'] = pd.to_numeric(df['Infection_Freq'], errors='coerce')
        
    for col in ['Diabetes', 'Hypertension', 'Hospital_before']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            df[col] = df[col].map({'yes': '1', 'true': '1', '1': '1', 'no': '0', 'false': '0', '0': '0'})
            
    df['Gender'] = df['Gender'].astype(str)
    if 'Souches' in df.columns:
        df['Souches'] = df['Souches'].astype(str)

    return df

def main():
    print("Loading data...")
    df = pd.read_csv('data/Bacteria_dataset_Multiresictance.csv')
    
    antibiotics = ['AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN', 
                   'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole', 'Furanes', 'colistine']
                   
    numeric_features = ['Age', 'Infection_Freq']
    categorical_features = ['Gender', 'Souches', 'Diabetes', 'Hypertension', 'Hospital_before']
    
    models_dict = {}
    
    print("Training independent XGBoost models for ALL Antibiotics...")
    for target in antibiotics:
        df_clean = clean_data(df, target)
        if len(df_clean) < 100: # Skip if insufficient clean training data
            continue
            
        X = df_clean.drop(columns=[target] + [c for c in antibiotics if c in df_clean.columns])
        y = df_clean[target]
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        num_trans = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        cat_trans = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
                                    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
        preprocessor = ColumnTransformer(transformers=[('num', num_trans, numeric_features), ('cat', cat_trans, categorical_features)])
        
        # Lightweight model sufficient for UI
        neg_cases = (y_train == 0).sum()
        pos_cases = (y_train == 1).sum()
        scale_w = neg_cases / pos_cases if pos_cases > 0 else 1
        
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                                   ('classifier', XGBClassifier(n_estimators=50, random_state=42, scale_pos_weight=scale_w, eval_metric='logloss'))])

        pipeline.fit(X_train, y_train)
        models_dict[target] = pipeline
        print(f" -> Trained model for {target}")

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(models_dict, f)
    print("Saved dictionary of models to model/model.pkl!")

if __name__ == "__main__":
    main()
