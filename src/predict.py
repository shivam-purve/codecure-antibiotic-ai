import pickle
import pandas as pd
import numpy as np
import re

# Load the dictionary of pipelines
try:
    with open('model/model.pkl', 'rb') as f:
        models = pickle.load(f)
except FileNotFoundError:
    try:
        # For fallback if executed inside src/
        with open('../model/model.pkl', 'rb') as f:
            models = pickle.load(f)
    except FileNotFoundError:
        models = {}

def clean_bacteria_input(x):
    """Helper to clean the bacterial text input if necessary."""
    val = re.sub(r'^S\d+\s+', '', str(x).strip())
    return val

def get_model_explanation(model, input_data):
    """Extract feature importances and their values for explanation."""
    try:
        classifier = model.named_steps['classifier']
        preprocessor = model.named_steps['preprocessor']
        
        num_features = preprocessor.transformers_[0][2]
        cat_features = preprocessor.transformers_[1][2]
        
        cat_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
        cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
        
        all_feature_names = list(num_features) + list(cat_feature_names)
        importances = classifier.feature_importances_
        
        transformed_input = preprocessor.transform(input_data)
        if hasattr(transformed_input, "toarray"):
            transformed_input = transformed_input.toarray()
            
        transformed_input = transformed_input[0]
        
        explanation = []
        for name, imp, val in zip(all_feature_names, importances, transformed_input):
            if val > 0 or imp > 0.02:
                # Clean up feature names
                clean_name = name.replace('Souches_', 'Bacteria: ')
                clean_name = clean_name.replace('Gender_', 'Gender: ')
                explanation.append({'feature': clean_name, 'importance': float(imp), 'value': float(val)})
                
        explanation.sort(key=lambda x: x['importance'], reverse=True)
        return explanation[:5]
    except Exception as e:
        print(f"Explanation error: {e}")
        return []

def predict_resistance(bacteria_name, antibiotic, age=40, gender='Unknown', diabetes='0', hypertension='0', hospital='0', infection_freq=1.0):
    """Predict if a bacteria is Resistant or Susceptible to ONE antibiotic based on Patient Context."""
    # Create singleton dataframe mapped to feature schema
    input_data = pd.DataFrame({
        'Age': [age],
        'Infection_Freq': [infection_freq],
        'Gender': [gender],
        'Souches': [bacteria_name],
        'Diabetes': [str(diabetes)],
        'Hypertension': [str(hypertension)],
        'Hospital_before': [str(hospital)]
    })

    if antibiotic not in models:
        if not models:
            return "Model not found", 0.0, []
            
        # Broad-spectrum generalization approach for untrained antibiotics
        avg_prob_s = 0.0
        avg_prob_r = 0.0
        
        for abx, model in models.items():
            probs = model.predict_proba(input_data)[0]
            idx_s = list(model.classes_).index(0) if 0 in list(model.classes_) else -1
            idx_r = list(model.classes_).index(1) if 1 in list(model.classes_) else -1
            
            avg_prob_s += probs[idx_s] if idx_s != -1 else 0.0
            avg_prob_r += probs[idx_r] if idx_r != -1 else 0.0
            
        avg_prob_s /= len(models)
        avg_prob_r /= len(models)
        
        # Take an explanation from a widely used model as proxy
        proxy_model = models.get('AMX/AMP', list(models.values())[0])
        explanation = get_model_explanation(proxy_model, input_data)
        
        if avg_prob_r > 0.5:
            return "Resistant", float(avg_prob_r), explanation
        else:
            return "Susceptible", float(avg_prob_s), explanation

    model = models[antibiotic]
    
    # Extract prediction probability
    probs = model.predict_proba(input_data)[0]
    
    # XGBoost mapped 0=Susceptible, 1=Resistant in train.py
    try:
        idx_susceptible = list(model.classes_).index(0)
    except ValueError:
        # If model only ever saw exclusively resistant (edge cases)
        idx_susceptible = -1
        
    try:
        idx_resistant = list(model.classes_).index(1)
    except ValueError:
        idx_resistant = -1
    
    prob_s = probs[idx_susceptible] if idx_susceptible != -1 else 0.0  # type: ignore
    prob_r = probs[idx_resistant] if idx_resistant != -1 else 0.0  # type: ignore
    
    # Recommendation logic based on probability
    explanation = get_model_explanation(model, input_data)
    
    if prob_r > 0.5:
        return "Resistant", float(prob_r), explanation
    else:
        return "Susceptible", float(prob_s), explanation

def recommend_top_3_antibiotics(bacteria_name, exclude_antibiotic=None, age=40, gender='Unknown', diabetes='0', hypertension='0', hospital='0', infection_freq=1.0):
    """Return top 3 antibiotics with highest probability of being effective (Susceptible),
    excluding the currently assessed antibiotic so recommendations are genuine alternatives."""
    if not models:
        return []
        
    input_data = pd.DataFrame({
        'Age': [age],
        'Infection_Freq': [infection_freq],
        'Gender': [gender],
        'Souches': [bacteria_name],
        'Diabetes': [str(diabetes)],
        'Hypertension': [str(hypertension)],
        'Hospital_before': [str(hospital)]
    })
    
    recommendations = []
    
    # Scan through all trained antibiotic models
    for abx, model in models.items():
        # Skip the currently assessed antibiotic — recommendations must be true alternatives
        if exclude_antibiotic and abx == exclude_antibiotic:
            continue
        try:
            # Finding probability of index 0 (Susceptible)
            idx_susceptible = list(model.classes_).index(0)
            probs = model.predict_proba(input_data)[0]
            prob_s = float(probs[idx_susceptible]) # type: ignore
            explanation = get_model_explanation(model, input_data)
            recommendations.append((abx, prob_s, explanation))
        except ValueError:
            # Failed to map susceptible
            recommendations.append((abx, 0.0, []))
            
    # Sort strictly by highest Susceptible probability descending
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:3]

def get_all_antibiotic_probabilities(bacteria_name, age=40, gender='Unknown', diabetes='0', hypertension='0', hospital='0', infection_freq=1.0):
    """Return susceptibility probabilities for ALL available antibiotics."""
    input_data = pd.DataFrame({
        'Age': [age],
        'Infection_Freq': [infection_freq],
        'Gender': [gender],
        'Souches': [bacteria_name],
        'Diabetes': [str(diabetes)],
        'Hypertension': [str(hypertension)],
        'Hospital_before': [str(hospital)]
    })
    
    results = {}
    for abx, model in models.items():
        try:
            idx_susceptible = list(model.classes_).index(0)
            prob_s = float(model.predict_proba(input_data)[0][idx_susceptible])
            results[abx] = prob_s
        except ValueError:
            results[abx] = 0.0
    return results
