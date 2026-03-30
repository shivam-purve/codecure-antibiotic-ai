"""
CodeCure AI: Clinical Decision Support Dashboard

This module is the core Streamlit interface for predicting antibiotic resistance.
It utilizes pre-trained XGBoost models to provide clinical insights, 3D molecular
renderings, and multi-objective optimization recommendations (efficacy vs. toxicity/cost).
"""
import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import urllib.parse
import datetime
from contextlib import suppress
# Ensure absolute imports work from root since Streamlit runs natively in app/
sys.path.append(os.getcwd())

from src.predict import (
    predict_resistance, 
    recommend_top_3_antibiotics, 
    get_all_antibiotic_probabilities,
    models
)

# Global configuration variables
antibiotics_list = list(models.keys()) if models else ['AMX/AMP', 'AMC', 'CZ', 'FOX', 'IPM']

bacteria_list = [
    'Escherichia coli',
    'Klebsiella pneumoniae',
    'Proteus mirabilis',
    'Enterobacteria spp.',
    'Citrobacter spp.',
    'Pseudomonas aeruginosa',
    'Serratia marcescens',
    'Acinetobacter baumannii',
    'Morganella morganii',
    'Staphylococcus aureus'
]

# Image URLs for bacteria
BACTERIA_REAL_IMAGES = {
    'Escherichia coli': 'https://images.unsplash.com/photo-1579684385127-1ef15d508118?ixlib=rb-4.0.3&w=800&q=80',
    'Klebsiella pneumoniae': 'https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?ixlib=rb-4.0.3&w=800&q=80',
    'Proteus mirabilis': 'https://images.unsplash.com/photo-1614728470691-1fa12e52bc81?ixlib=rb-4.0.3&w=800&q=80',
    'Enterobacteria spp.': 'https://images.unsplash.com/photo-1530026405186-ed1f139313f8?ixlib=rb-4.0.3&w=800&q=80',
    'Citrobacter spp.': 'https://images.unsplash.com/photo-1583324113626-70df0f4deaab?ixlib=rb-4.0.3&w=800&q=80',
    'Pseudomonas aeruginosa': 'https://images.unsplash.com/photo-1518152006812-edab29b069ac?ixlib=rb-4.0.3&w=800&q=80',
    'Serratia marcescens': 'https://images.unsplash.com/photo-1563200985-79d8ecbae23b?ixlib=rb-4.0.3&w=800&q=80',
    'Acinetobacter baumannii': 'https://images.unsplash.com/photo-1618051253017-f273b06e30ab?ixlib=rb-4.0.3&w=800&q=80',
    'Morganella morganii': 'https://images.unsplash.com/photo-1576086213369-97a306d36557?ixlib=rb-4.0.3&w=800&q=80',
    'Staphylococcus aureus': 'https://images.unsplash.com/photo-1632395627727-4bdaea284dd3?ixlib=rb-4.0.3&w=800&q=80'
}


# PubChem CIDs for antibiotics
PUBCHEM_MAPPING = {
    'AMX/AMP': 33613,      # Amoxicillin
    'AMC': 33613,          # Amoxicillin Base (for Combination 3D)
    'CZ': 54038,           # Cefazolin
    'FOX': 439050,         # Cefoxitin
    'CTX/CRO': 5479530,    # Ceftriaxone
    'IPM': 104838,         # Imipenem
    'GEN': 3467,           # Gentamicin
    'AN': 417855,          # Amikacin
    'Acide nalidixique': 4421, # Nalidixic Acid
    'ofx': 4584,           # Ofloxacin
    'CIP': 2764,           # Ciprofloxacin
    'C': 5959,             # Chloramphenicol
    'Co-trimoxazole': 53232, # Trimethoprim
    'Furanes': 6433,       # Nitrofurantoin
    'colistine': 9832267   # Polymyxin E1 (Colistin active base)
}

# Clinical metadata for recommendations (toxicity & cost considerations)
CLINICAL_METADATA = {
    'AMX/AMP': {'toxicity': 'Low', 'cost': '$'},
    'AMC': {'toxicity': 'Low', 'cost': '$$'},
    'CZ': {'toxicity': 'Medium', 'cost': '$$'},
    'FOX': {'toxicity': 'Medium', 'cost': '$$'},
    'CTX/CRO': {'toxicity': 'Medium', 'cost': '$$$'},
    'IPM': {'toxicity': 'High', 'cost': '$$$'},
    'GEN': {'toxicity': 'High', 'cost': '$'},
    'AN': {'toxicity': 'High', 'cost': '$$'},
    'Acide nalidixique': {'toxicity': 'Low', 'cost': '$'},
    'ofx': {'toxicity': 'Medium', 'cost': '$$'},
    'CIP': {'toxicity': 'Medium', 'cost': '$$'},
    'C': {'toxicity': 'High', 'cost': '$'},
    'Co-trimoxazole': {'toxicity': 'Low', 'cost': '$'},
    'Furanes': {'toxicity': 'Low', 'cost': '$'},
    'colistine': {'toxicity': 'Very High', 'cost': '$$$'}
}

def get_clinical_badge(level, type_str):
    color = "green" if level in ["Low", "$"] else "orange" if level in ["Medium", "$$"] else "red"
    return f"<span style='background-color:{color}; color:white; padding:4px 10px; border-radius:12px; font-size:0.85rem; margin-right:5px; font-weight: bold;'>{type_str}: {level}</span>"

def fetch_wiki_image(query):
    try:
        query_encoded = urllib.parse.quote(query)
        url = f"https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch={query_encoded}&gsrlimit=1&prop=pageimages&pithumbsize=800&format=json"
        headers = {'User-Agent': 'CodeCureAI/1.0'}
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_info in pages.items():
                if 'thumbnail' in page_info:
                    return page_info['thumbnail']['source']
    except Exception:
        pass
    # fallback generic bacteria image
    return "https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?ixlib=rb-4.0.3&w=800&q=80"

def fetch_pubchem_cid(query):
    try:
        query_encoded = urllib.parse.quote(query)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query_encoded}/cids/JSON"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            cids = data.get('IdentifierList', {}).get('CID', [])
            if cids:
                return cids[0]
    except Exception:
        pass
    return 33613 # default to Amoxicillin

# ---------- CSS styling layout ----------
st.set_page_config(page_title="CodeCure AI | Clinical Decision Support", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #eef5f9 0%, #ffffff 100%);
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 20px;
        border-top: 5px solid #005A9C;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-size: 1.1rem;
        color: #34495e;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.2rem;
        color: #005A9C;
        font-weight: 800;
        margin: 10px 0;
    }
    iframe {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        background-color: #fcfcfc;
    }
    .css-1v0mbdj > img {
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 CodeCure AI: Antibiotic Resistance Predictor")
st.markdown("An AI-powered dashboard estimating pathogen susceptibility and providing clinical recommendations based on patient demographics.")
st.markdown("---")

# ---------- Sidebar Patient Demographics ----------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100) # Medical logo
    st.header("👤 Patient Context")
    st.markdown("Adjust patient parameters to strictly update resistance predictions.")
    age = st.slider("Patient Age", 1, 100, 40)
    gender = st.selectbox("Biological Sex", ['M', 'F', 'Unknown'], index=2)
    st.markdown("---")
    st.subheader("🏥 Comorbidities & Medical History")
    history = st.checkbox("Previous Hospitalization History")
    diabetes = st.checkbox("Diabetes Mellitus")
    hypertension = st.checkbox("Hypertension")
    
    hospital_val = '1' if history else '0'
    diabetes_val = '1' if diabetes else '0'
    hypertension_val = '1' if hypertension else '0'

# ---------- Main Core Selection ----------
st.markdown("### Step 1: Input Infection Details")
col1, col2 = st.columns(2)
with col1:
    st.subheader("🦠 Select Pathogen")
    bacteria_dropdown = st.selectbox("Select Isolated Bacteria Strain", bacteria_list + ["Other (Custom...)"], help="The pathogen driving the infection.")
    
    if bacteria_dropdown == "Other (Custom...)":
        bacteria_input = st.text_input("Enter Custom Pathogen Name", "Streptococcus pneumoniae")
        img_url = fetch_wiki_image(bacteria_input)
        bacteria_dropdown = bacteria_input
    else:
        img_url = BACTERIA_REAL_IMAGES.get(bacteria_dropdown, "https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?ixlib=rb-4.0.3&w=800&q=80")
        
    st.markdown(f"**Strain: {bacteria_dropdown}**")
    st.image(img_url, use_container_width=True)

with col2:
    st.subheader("💊 Target Treatment")
    antibiotic_dropdown = st.selectbox("Assess Specific Target Antibiotic", antibiotics_list + ["Other (Custom...)"], help="Which treatment drug line are you initially considering?")
    
    if antibiotic_dropdown == "Other (Custom...)":
        antibiotic_input = st.text_input("Enter Custom Antibiotic Name", "Azithromycin")
        cid = fetch_pubchem_cid(antibiotic_input)
        antibiotic_dropdown = antibiotic_input
    else:
        cid = PUBCHEM_MAPPING.get(antibiotic_dropdown, 33613)
        
    # Interactive 3D Molecular iFrame
    st.markdown(f"**3D Structure: {antibiotic_dropdown}**")
    st.components.v1.html(
        f'<iframe style="width: 100%; height: 260px; border: none; overflow: hidden;" src="https://pubchem.ncbi.nlm.nih.gov/vw3d/vw3d.cgi?smin=0&cid={cid}&v=auto"></iframe>',
        height=270
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------- Execution Flow ----------
st.markdown("### Step 2: Run Analysis")
if st.button("🔬 Analyze Resistance", type="primary", use_container_width=True):
    with st.spinner("Analyzing patient demographic and resistance patterns..."):
        
        prediction, confidence, explanation = predict_resistance(
            bacteria_dropdown, antibiotic_dropdown, 
            age=age, gender=gender, hospital=hospital_val,
            diabetes=diabetes_val, hypertension=hypertension_val
        )
        
        # UI Breakdown
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.subheader("Primary Assessment")
            if antibiotic_dropdown not in antibiotics_list:
                st.info(f"**Clinical Insight:** Initiating broad-spectrum algorithmic proxy to evaluate generalized susceptibility for '{antibiotic_dropdown}'.", icon="ℹ️")
            if prediction == "Resistant":
                st.error(f"⚠️ **{antibiotic_dropdown}** is predicted **RESISTANT** against **{bacteria_dropdown}**.")
                st.metric(label="Confidence", value=f"{confidence*100:.1f}%", delta="-High Risk Context", delta_color="inverse")
                st.markdown("*Note: Treatment may fail. Consider alternatives.*")
            else:
                st.success(f"✅ **{antibiotic_dropdown}** is predicted **SUSCEPTIBLE** (Effective) against **{bacteria_dropdown}**.")
                st.metric(label="Confidence", value=f"{confidence*100:.1f}%", delta="Viable Option", delta_color="normal")
                
        with res_col2:
            st.subheader("Key Clinical Factors")
            st.markdown(f"The **{prediction}** prediction was primarily influenced by:")
            if explanation:
                exp_df = pd.DataFrame(explanation)
                fig = px.bar(exp_df, x='importance', y='feature', orientation='h', 
                             title="Feature Impact Profiling", 
                             color='importance', color_continuous_scale='Blues')
                fig.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Feature importance could not be calculated for this combination scope.")
            
        st.markdown("---")
        
        # 2. Recommendations
        st.subheader("💡 Top Recommended Alternatives")
        st.markdown(f"Alternative antibiotics with the highest probability of success against **{bacteria_dropdown}**, factoring in toxicity and cost:")
        
        top_3 = recommend_top_3_antibiotics(
            bacteria_dropdown, 
            age=age, gender=gender, hospital=hospital_val,
            diabetes=diabetes_val, hypertension=hypertension_val
        )
        
        if top_3:
            rec_cols = st.columns(3)
            for idx, (abx, prob, exp) in enumerate(top_3):
                meta = CLINICAL_METADATA.get(abx, {'toxicity': 'Unknown', 'cost': 'Unknown'})
                tox_badge = get_clinical_badge(meta['toxicity'], "Toxicity")
                cost_badge = get_clinical_badge(meta['cost'], "Cost")
                
                with rec_cols[idx]:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-title">Rank {idx+1}: {abx}</div>
                        <div class="metric-value">{prob*100:.1f}%</div>
                        <small>Statistical Efficacy</small><br><br>
                        <div style="margin-top: 10px;">{tox_badge} {cost_badge}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    with st.expander(f"View Feature Explainer for {abx}"):
                        if exp:
                            exp_df_top = pd.DataFrame(exp)
                            fig2 = px.bar(exp_df_top, x='importance', y='feature', orientation='h', height=250)
                            fig2.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=0, b=0))
                            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No functioning ML models loaded. Did you train the dataset under `src/train.py`?")

        st.markdown("---")
        
        # 3. Antibiogram Heatmap
        st.subheader("📊 Susceptibility Overview")
        st.markdown(f"Estimated susceptibility for **{bacteria_dropdown}** across all modeled antibiotics for this patient:")
        
        all_probs = get_all_antibiotic_probabilities(
            bacteria_dropdown, age=age, gender=gender, hospital=hospital_val,
            diabetes=diabetes_val, hypertension=hypertension_val
        )
        
        prob_df = pd.DataFrame(list(all_probs.items()), columns=['Antibiotic', 'Susceptibility Confidence'])
        prob_df['Susceptibility Confidence'] = prob_df['Susceptibility Confidence'] * 100
        prob_df = prob_df.sort_values(by='Susceptibility Confidence', ascending=False)
        
        fig_all = px.bar(prob_df, x='Antibiotic', y='Susceptibility Confidence',
                         color='Susceptibility Confidence', color_continuous_scale='Portland',
                         title=f"Predictive Antibiogram: {bacteria_dropdown}", text_auto='.1f')
        fig_all.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_all, use_container_width=True)
        
        # --- 4. Export Report ---
        st.markdown("---")
        st.subheader("📄 Download Report")
        st.markdown("Export a markdown summary of the analysis for record-keeping.")
        
        report_content = f"""# CodeCure AI - Patient Case Report
Date Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---------------------------------------------------------
PATIENT DEMOGRAPHICS
- Age: {age}
- Gender: {gender}
- Previous Hospitalization: {"Yes" if history else "No"}
- Diabetes Mellitus: {"Yes" if diabetes else "No"}
- Hypertension: {"Yes" if hypertension else "No"}

PATHOGEN & PRIMARY DIAGNOSTICS
- Isolated Strain: {bacteria_dropdown}
- Assessed Target Treatment: {antibiotic_dropdown}
>> Model Assessment: {prediction} (Confidence Metric: {confidence*100:.1f}%)

TOP 3 RECOMMENDED ALTERNATIVES (Efficacy vs Tradeoff Metrics)
"""
        for idx, (abx, prob, exp) in enumerate(top_3):
            meta = CLINICAL_METADATA.get(abx, {'toxicity': 'Unknown', 'cost': 'Unknown'})
            report_content += f"{idx+1}. {abx} - {prob*100:.1f}% Efficacy | Toxicity: {meta['toxicity']} | Cost: {meta['cost']}\n"
            
        report_content += "\n---------------------------------------------------------\n"
        report_content += "Disclaimer: This AI-generated report is for strict clinical decision support and should not unilaterally override physician judgement guidelines."
        
        st.download_button(
            label="📥 Download Clinical Case Report (.md / PDF-ready)",
            data=report_content,
            file_name=f"CodeCure_PatientReport_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.md",
            mime="text/markdown",
            use_container_width=True
        )

st.markdown("<br><hr><center><small>CodeCure AI • Powered by XGBoost</small></center>", unsafe_allow_html=True)
