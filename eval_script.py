import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.train import clean_data

def evaluate():
    print("Loading data...")
    df = pd.read_csv('data/Bacteria_dataset_Multiresictance.csv')
    with open('model/model.pkl', 'rb') as f:
        models_dict = pickle.load(f)
        
    antibiotics = ['AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN', 
                   'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole', 'Furanes', 'colistine']
                   
    results = []
    
    for target in models_dict.keys():
        df_clean = clean_data(df, target)
        if len(df_clean) < 100:
            continue
            
        X = df_clean.drop(columns=[target] + [c for c in antibiotics if c in df_clean.columns])
        y = df_clean[target]
        
        try:
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except Exception:
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
        model = models_dict[target]
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'Antibiotic': target,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'Test_Size': len(y_test)
        })
        
    res_df = pd.DataFrame(results)
    res_df.set_index('Antibiotic').to_csv('metrics.csv')

if __name__ == '__main__':
    evaluate()
