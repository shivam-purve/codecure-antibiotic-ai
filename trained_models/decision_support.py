"""
decision_support.py - Complete Clinical Decision Support System with Dashboard
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AMRDecisionSupport:
    """
    Clinical Decision Support System for Antibiotic Resistance Prediction
    """
    
    def __init__(self, models_dir='saved_models'):
        """Initialize the decision support tool"""
        self.models_dir = models_dir
        self.models = {}
        self.metadata = None
        self.preprocessing = None
        self.performance_data = None
        
        # Load performance data
        try:
            self.performance_data = pd.read_csv('reports/model_performance.csv')
            print("✓ Loaded model performance data")
        except:
            print("⚠️  Performance data not found")
        
        # Load metadata
        try:
            with open(f'{models_dir}/metadata.pkl', 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"✓ Loaded metadata for {len(self.metadata['antibiotics'])} antibiotics")
        except Exception as e:
            print(f"⚠️  Metadata not found: {e}")
        
        # Load preprocessing objects
        try:
            with open(f'{models_dir}/preprocessing.pkl', 'rb') as f:
                self.preprocessing = pickle.load(f)
            print("✓ Loaded preprocessing objects")
        except:
            print("⚠️  Preprocessing objects not found")
        
        # Load individual models
        antibiotics = ['AMC', 'AMX/AMP', 'CZ', 'FOX', 'CTX/CRO', 'IPM', 'GEN', 'AN', 
                       'CIP', 'Co-trimoxazole', 'Acide nalidixique', 'C', 'ofx', 
                       'colistine', 'Furanes']
        
        loaded_count = 0
        for antibiotic in antibiotics:
            try:
                model_filename = f"{models_dir}/{antibiotic.replace('/', '_').replace(' ', '_')}_model.pkl"
                with open(model_filename, 'rb') as f:
                    self.models[antibiotic] = pickle.load(f)
                loaded_count += 1
            except Exception as e:
                print(f"⚠️  Model for {antibiotic} not found: {e}")
        
        print(f"✓ Loaded {loaded_count} models")
    
    def predict_resistance(self, patient_data):
        """
        Predict resistance for all antibiotics based on patient data
        """
        predictions = {}
        probabilities = {}
        confidence_scores = {}
        
        # Create feature vector
        features = {
            'Diabetes_Encoded': 1 if patient_data.get('diabetes', '').upper() in ['YES', 'TRUE', 'Y'] else 0,
            'Hypertension_Encoded': 1 if patient_data.get('hypertension', '').upper() in ['YES', 'TRUE', 'Y'] else 0,
            'Hospital_before_Encoded': 1 if patient_data.get('hospital_before', '').upper() in ['YES', 'TRUE', 'Y'] else 0,
            'Age': float(patient_data.get('age', 50)),
            'Gender_Encoded': 1 if patient_data.get('gender', '').upper() == 'M' else 0,
            'Species_Encoded': 0  # Default for unknown species
        }
        
        # Create DataFrame
        X = pd.DataFrame([features])
        
        # Scale features
        try:
            if 'scaler' in self.models.get(list(self.models.keys())[0], {}):
                # Use first model's scaler
                first_model = list(self.models.values())[0]
                X_scaled = first_model['scaler'].transform(X)
            else:
                X_scaled = X.values
        except:
            X_scaled = X.values
        
        # Get predictions from each model
        for antibiotic, model_data in self.models.items():
            try:
                model = model_data['model']
                
                # Predict
                pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0]
                
                predictions[antibiotic] = "Resistant" if pred == 1 else "Susceptible"
                probabilities[antibiotic] = proba[1] if pred == 1 else proba[0]
                
                # Calculate confidence score
                if pred == 1:
                    confidence_scores[antibiotic] = proba[1]
                else:
                    confidence_scores[antibiotic] = proba[0]
                
            except Exception as e:
                print(f"Error predicting {antibiotic}: {e}")
                predictions[antibiotic] = "Unknown"
                probabilities[antibiotic] = 0.5
                confidence_scores[antibiotic] = 0.5
        
        return predictions, probabilities, confidence_scores
    
    def recommend_antibiotics(self, predictions, confidence_scores, threshold=0.7):
        """
        Generate antibiotic recommendations
        """
        recommendations = {
            'first_line': [],
            'second_line': [],
            'avoid': [],
            'reserve': []
        }
        
        # Antibiotic classes
        beta_lactams = ['AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM']
        aminoglycosides = ['GEN', 'AN']
        fluoroquinolones = ['CIP', 'ofx', 'Acide nalidixique']
        others = ['Co-trimoxazole', 'C', 'colistine', 'Furanes']
        
        for antibiotic, pred in predictions.items():
            conf = confidence_scores.get(antibiotic, 0.5)
            
            # Determine antibiotic class
            if antibiotic in beta_lactams:
                ab_class = "Beta-lactam"
            elif antibiotic in aminoglycosides:
                ab_class = "Aminoglycoside"
            elif antibiotic in fluoroquinolones:
                ab_class = "Fluoroquinolone"
            else:
                ab_class = "Other"
            
            if pred == "Susceptible":
                if conf > threshold:
                    recommendations['first_line'].append({
                        'antibiotic': antibiotic,
                        'class': ab_class,
                        'confidence': conf,
                        'prediction': pred
                    })
                else:
                    recommendations['second_line'].append({
                        'antibiotic': antibiotic,
                        'class': ab_class,
                        'confidence': conf,
                        'prediction': pred
                    })
            else:
                recommendations['avoid'].append({
                    'antibiotic': antibiotic,
                    'class': ab_class,
                    'confidence': conf,
                    'prediction': pred
                })
        
        # Sort by confidence
        for key in recommendations:
            recommendations[key] = sorted(
                recommendations[key],
                key=lambda x: x['confidence'],
                reverse=True
            )
        
        # Identify reserve antibiotics (last resort)
        recommendations['reserve'] = [rec for rec in recommendations['first_line'] 
                                     if rec['antibiotic'] in ['colistine', 'IPM', 'GEN']]
        
        return recommendations
    
    def generate_visual_report(self, patient_data, predictions, probabilities, confidence_scores):
        """
        Generate visual report with charts
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Resistance Profile
        ax1 = axes[0, 0]
        antibiotics = list(predictions.keys())
        colors = ['red' if pred == 'Resistant' else 'green' for pred in predictions.values()]
        y_pos = np.arange(len(antibiotics))
        
        ax1.barh(y_pos, [probabilities[ab] for ab in antibiotics], color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(antibiotics)
        ax1.set_xlabel('Resistance Probability')
        ax1.set_title('Antibiotic Resistance Profile', fontweight='bold')
        ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # 2. Confidence Scores
        ax2 = axes[0, 1]
        sorted_confidence = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        antibiotics_conf = [x[0] for x in sorted_confidence]
        conf_values = [x[1] for x in sorted_confidence]
        
        colors_conf = ['green' if c > 0.7 else 'orange' if c > 0.5 else 'red' for c in conf_values]
        ax2.bar(range(len(antibiotics_conf)), conf_values, color=colors_conf, alpha=0.7)
        ax2.set_xticks(range(len(antibiotics_conf)))
        ax2.set_xticklabels(antibiotics_conf, rotation=45, ha='right')
        ax2.set_ylabel('Confidence Score')
        ax2.set_title('Prediction Confidence by Antibiotic', fontweight='bold')
        ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High Confidence')
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium Confidence')
        ax2.legend()
        
        # 3. Risk Assessment
        ax3 = axes[1, 0]
        resistant_count = sum(1 for pred in predictions.values() if pred == 'Resistant')
        susceptible_count = len(predictions) - resistant_count
        
        labels = ['Resistant', 'Susceptible']
        sizes = [resistant_count, susceptible_count]
        colors_pie = ['#E63946', '#2A9D8F']
        
        ax3.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Overall Resistance Profile', fontweight='bold')
        
        # 4. Model Performance Summary
        ax4 = axes[1, 1]
        if self.performance_data is not None:
            # Get best model for each antibiotic
            best_models = self.performance_data.loc[
                self.performance_data.groupby('Antibiotic')['ROC-AUC'].idxmax()
            ]
            best_models = best_models.sort_values('ROC-AUC', ascending=False).head(10)
            
            ax4.barh(range(len(best_models)), best_models['ROC-AUC'], color='steelblue', alpha=0.7)
            ax4.set_yticks(range(len(best_models)))
            ax4.set_yticklabels(best_models['Antibiotic'])
            ax4.set_xlabel('ROC-AUC Score')
            ax4.set_title('Model Performance (Top 10 Antibiotics)', fontweight='bold')
            ax4.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='Good')
            ax4.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        
        plt.suptitle(f"Antibiotic Resistance Report - {patient_data.get('name', 'Patient')}", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        filename = f"reports/patient_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        return filename
    
    def generate_clinical_report(self, patient_data):
        """
        Generate complete clinical report with recommendations
        """
        # Get predictions
        predictions, probabilities, confidence_scores = self.predict_resistance(patient_data)
        recommendations = self.recommend_antibiotics(predictions, confidence_scores)
        
        # Generate visual report
        visual_report = self.generate_visual_report(patient_data, predictions, probabilities, confidence_scores)
        
        # Calculate MDR status
        resistant_count = sum(1 for pred in predictions.values() if pred == "Resistant")
        total_count = len(predictions)
        mdr_status = ""
        
        if resistant_count >= 8:
            mdr_status = "PDR (Pandrug-Resistant) - Resistant to 8+ antibiotics"
            mdr_color = "🔴 CRITICAL"
        elif resistant_count >= 5:
            mdr_status = "XDR (Extensively Drug-Resistant) - Resistant to 5+ antibiotics"
            mdr_color = "🟠 HIGH RISK"
        elif resistant_count >= 2:
            mdr_status = "MDR (Multi-Drug Resistant) - Resistant to 2+ antibiotics"
            mdr_color = "🟡 MODERATE RISK"
        else:
            mdr_status = "Non-MDR - Low resistance profile"
            mdr_color = "🟢 LOW RISK"
        
        # Generate text report
        report = []
        report.append("="*80)
        report.append("ANTIMICROBIAL RESISTANCE CLINICAL REPORT")
        report.append("="*80)
        report.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Patient: {patient_data.get('name', 'Unknown')}")
        report.append(f"Age/Gender: {patient_data.get('age', '?')}/{patient_data.get('gender', '?')}")
        report.append(f"Medical History:")
        report.append(f"  - Diabetes: {patient_data.get('diabetes', 'Unknown')}")
        report.append(f"  - Hypertension: {patient_data.get('hypertension', 'Unknown')}")
        report.append(f"  - Previous Hospitalization: {patient_data.get('hospital_before', 'Unknown')}")
        
        report.append(f"\n" + "="*80)
        report.append(f"RESISTANCE ASSESSMENT: {mdr_color}")
        report.append("="*80)
        report.append(f"\nResistance Profile:")
        report.append(f"  • Total Antibiotics Analyzed: {total_count}")
        report.append(f"  • Predicted Resistant: {resistant_count}/{total_count}")
        report.append(f"  • MDR Classification: {mdr_status}")
        
        # Treatment Recommendations
        report.append(f"\n" + "="*80)
        report.append("TREATMENT RECOMMENDATIONS")
        report.append("="*80)
        
        if recommendations['first_line']:
            report.append("\n✅ FIRST-LINE RECOMMENDED (High Confidence):")
            for rec in recommendations['first_line'][:5]:
                report.append(f"  • {rec['antibiotic']} ({rec['class']}) - "
                            f"Confidence: {rec['confidence']:.1%}")
        
        if recommendations['second_line']:
            report.append("\n⚠️  SECOND-LINE OPTIONS (Moderate Confidence):")
            for rec in recommendations['second_line'][:3]:
                report.append(f"  • {rec['antibiotic']} ({rec['class']}) - "
                            f"Confidence: {rec['confidence']:.1%}")
        
        if recommendations['reserve']:
            report.append("\n💊 RESERVE ANTIBIOTICS (Last Resort):")
            for rec in recommendations['reserve'][:3]:
                report.append(f"  • {rec['antibiotic']} ({rec['class']}) - "
                            f"Confidence: {rec['confidence']:.1%}")
        
        if recommendations['avoid']:
            report.append("\n❌ AVOID (High Resistance Risk):")
            for rec in recommendations['avoid'][:5]:
                report.append(f"  • {rec['antibiotic']} ({rec['class']}) - "
                            f"Resistance Probability: {rec['confidence']:.1%}")
        
        # Detailed predictions by class
        report.append(f"\n" + "="*80)
        report.append("DETAILED PREDICTIONS BY ANTIBIOTIC CLASS")
        report.append("="*80)
        
        by_class = {}
        for antibiotic, pred in predictions.items():
            # Determine class
            if antibiotic in ['AMX/AMP', 'AMC', 'CZ', 'FOX', 'CTX/CRO', 'IPM']:
                ab_class = "Beta-lactams"
            elif antibiotic in ['GEN', 'AN']:
                ab_class = "Aminoglycosides"
            elif antibiotic in ['CIP', 'ofx', 'Acide nalidixique']:
                ab_class = "Fluoroquinolones"
            else:
                ab_class = "Other Antibiotics"
            
            if ab_class not in by_class:
                by_class[ab_class] = []
            by_class[ab_class].append((antibiotic, pred, confidence_scores[antibiotic]))
        
        for ab_class, items in by_class.items():
            report.append(f"\n{ab_class}:")
            for antibiotic, pred, conf in items:
                symbol = "✓" if pred == "Susceptible" else "✗"
                report.append(f"  {symbol} {antibiotic}: {pred} (confidence: {conf:.1%})")
        
        report.append(f"\n" + "="*80)
        report.append("CLINICAL NOTES")
        report.append("="*80)
        report.append("• This is an AI-assisted prediction tool based on clinical data only")
        report.append("• All predictions should be confirmed with laboratory testing")
        report.append("• Consider local resistance patterns and antibiogram data")
        report.append("• Consult infectious disease specialist for complex cases")
        report.append("• Visual report saved to: {}".format(visual_report))
        report.append("="*80)
        
        return "\n".join(report)

# Interactive clinical interface
def interactive_clinical_session():
    """
    Interactive command-line interface for clinical use
    """
    print("\n" + "="*80)
    print("ANTIBIOTIC RESISTANCE PREDICTION SYSTEM - CLINICAL INTERFACE")
    print("="*80)
    
    # Initialize the system
    ds = AMRDecisionSupport()
    
    while True:
        print("\n" + "-"*50)
        print("PATIENT INFORMATION")
        print("-"*50)
        
        # Collect patient data
        patient = {}
        patient['name'] = input("Patient Name (or 'quit' to exit): ").strip()
        
        if patient['name'].lower() == 'quit':
            print("\nExiting system...")
            break
        
        patient['age'] = input("Age: ").strip()
        patient['gender'] = input("Gender (M/F): ").strip().upper()
        patient['diabetes'] = input("Diabetes (Yes/No): ").strip().upper()
        patient['hypertension'] = input("Hypertension (Yes/No): ").strip().upper()
        patient['hospital_before'] = input("Previous Hospitalization (Yes/No): ").strip().upper()
        
        # Generate report
        print("\n" + "="*80)
        print("GENERATING CLINICAL REPORT...")
        print("="*80)
        
        report = ds.generate_clinical_report(patient)
        print(report)
        
        # Save report to file
        filename = f"reports/clinical_report_{patient['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write(report)
        print(f"\n✓ Full report saved to: {filename}")
        
        # Ask for another patient
        another = input("\n\nWould you like to analyze another patient? (y/n): ").strip().lower()
        if another != 'y':
            print("\nThank you for using the system!")
            break

if __name__ == "__main__":
    # Run interactive session
    interactive_clinical_session()