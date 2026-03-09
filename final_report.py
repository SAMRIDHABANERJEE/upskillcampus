"""
Title: Prediction of Agriculture Crop Production in India
Author: Samridha Banerjee
Date: March 2026
Description: This script integrates 4 weeks of development, including Gradient Boosting 
             ML models, crop duration feature engineering, and a localized recommendation engine.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# =================================================================
# 1. DATA LOADING & PREPROCESSING
# =================================================================

def load_and_clean_data():
    """Loads all 5 datasets and strips whitespace from column names."""
    try:
        df_cost = pd.read_csv('datafile (1).csv').rename(columns=lambda x: x.strip())
        df_index = pd.read_csv('datafile.csv').rename(columns=lambda x: x.strip())
        df_variety = pd.read_csv('datafile (3).csv').rename(columns=lambda x: x.strip())
        df_time_series = pd.read_csv('produce.csv').rename(columns=lambda x: x.strip())
        print("✅ Datasets loaded successfully.")
        return df_cost, df_index, df_variety, df_time_series
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Ensure all CSV files are in the working directory.")
        return None, None, None, None

# =================================================================
# 2. FEATURE ENGINEERING (WEEK 2 & 3 MILESTONES)
# =================================================================

def extract_duration_days(text):
    """Converts unstructured season text into numerical days."""
    if pd.isna(text) or text in ['NA', '-', '']: 
        return 130.0  # Median fallback
    text = str(text)
    # Range handling (e.g. 120-130)
    nums = re.findall(r'(\d+)', text)
    if nums:
        return np.mean([float(n) for n in nums])
    # Mapping qualitative terms
    mapping = {'Medium': 135, 'Short': 90, 'Early': 100, 'Late': 160, 'Long': 175}
    for key, val in mapping.items():
        if key.lower() in text.lower():
            return val
    return 130.0

# =================================================================
# 3. ADVANCED MACHINE LEARNING (WEEK 4 MILESTONE)
# =================================================================

def train_production_model(df_cost):
    """Trains a Gradient Boosting Regressor to predict crop yield."""
    le_crop = LabelEncoder()
    le_state = LabelEncoder()
    
    df_model = df_cost.copy()
    df_model['Crop_Enc'] = le_crop.fit_transform(df_model['Crop'])
    df_model['State_Enc'] = le_state.fit_transform(df_model['State'])
    
    # Features: Crop, State, Cost | Target: Yield
    X = df_model[['Crop_Enc', 'State_Enc', 'Cost of Cultivation (`/Hectare) C2']]
    y = df_model['Yield (Quintal/ Hectare)']
    
    # Gradient Boosting handles non-linear agricultural variables effectively
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X, y)
    
    print(f"🚀 Model Trained. Baseline R2 Score: {r2_score(y, model.predict(X)):.4f}")
    return model, le_crop, le_state

# =================================================================
# 4. RECOMMENDATION ENGINE & UI LOGIC
# =================================================================

def get_insights(state, crop, budget, model, le_crop, le_state, df_variety):
    """Predicts yield and recommends specific seed varieties."""
    try:
        # 1. Prediction
        c_enc = le_crop.transform([crop.upper()])[0]
        s_enc = le_state.transform([state.title()])[0]
        prediction = model.predict([[c_enc, s_enc, budget]])[0]
        
        print(f"\n--- Results for {crop} in {state} ---")
        print(f"Estimated Yield: {prediction:.2f} Quintals per Hectare")
        
        # 2. Recommendations
        recs = df_variety[(df_variety['Crop'].str.contains(crop, case=False, na=False)) & 
                          (df_variety['Recommended Zone'].str.contains(state, case=False, na=False))]
        
        if not recs.empty:
            print("\nRecommended Seed Varieties for your Zone:")
            print(recs[['Variety', 'Duration_Days', 'Recommended Zone']].head(3).to_string(index=False))
        else:
            print("\nNo specific localized varieties found for this selection.")
            
    except Exception as e:
        print(f"Input Error: {e}. Please check your spelling for Crop or State.")

# =================================================================
# MAIN EXECUTION
# =================================================================

if __name__ == "__main__":
    # 1. Load data
    df_cost, df_index, df_variety, df_time_series = load_and_clean_data()
    
    if df_cost is not None:
        # 2. Process Features
        df_variety['Duration_Days'] = df_variety['Season/ duration in days'].apply(extract_duration_days)
        
        # 3. Train Model
        final_model, le_crop, le_state = train_production_model(df_cost)
        
        # 4. Run Example Prediction (Simulation)
        # In a real GitHub repo, you could wrap this in a Flask/Streamlit app
        get_insights('Punjab', 'Paddy', 45000, final_model, le_crop, le_state, df_variety)
        
        # 5. Generate Final Dashboard Plot
        plt.figure(figsize=(12, 6))
        df_dash = df_index.set_index('Crop').T
        for c in ['Rice', 'Wheat', 'Pulses', 'Oilseeds']:
            plt.plot(df_dash.index, df_dash[c], marker='o', label=c)
        plt.title('Final Project: Agriculture Growth Trends (2004-2012)')
        plt.ylabel('Growth Index')
        plt.legend()
        plt.savefig('final_production_dashboard.png')
        print("\n📈 Final dashboard saved as 'final_production_dashboard.png'")
