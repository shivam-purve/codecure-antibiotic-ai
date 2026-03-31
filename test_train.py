import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.train import clean_data

df = pd.read_csv('data/Bacteria_dataset_Multiresictance.csv')
target = 'AMX/AMP'
antibiotics = ['AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN', 
               'Acide nalidixique', 'ofx', 'CIP', 'C', 'Co-trimoxazole', 'Furanes', 'colistine']
df_clean = clean_data(df, target)
X = df_clean.drop(columns=[target] + [c for c in antibiotics if c in df_clean.columns])
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

numeric_features = ['Age', 'Infection_Freq']
categorical_features = ['Gender', 'Souches', 'Diabetes', 'Hypertension', 'Hospital_before']

num_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_trans, numeric_features), 
    ('cat', cat_trans, categorical_features)
])

neg_cases = (y_train == 0).sum()
pos_cases = (y_train == 1).sum()
scale_w = neg_cases / pos_cases if pos_cases > 0 else 1

xgb_clf = XGBClassifier(
    n_estimators=300, 
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=scale_w,
    eval_metric='logloss'
)

rf_clf = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    max_depth=10,
    random_state=42
)

ensemble_clf = VotingClassifier(estimators=[
    ('xgb', xgb_clf),
    ('rf', rf_clf)
], voting='soft')

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', ensemble_clf)
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print('Target:', target)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred, zero_division=0))
print('Recall:', recall_score(y_test, y_pred, zero_division=0))
print('F1:', f1_score(y_test, y_pred, zero_division=0))