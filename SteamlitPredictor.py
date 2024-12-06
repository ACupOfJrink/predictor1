##########################################################################################分割线
##########################################################################################可段代码可能需要单独运行
#streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
Sepsismodel = joblib.load('CatboostSepsis.pkl')
Deathmodel = joblib.load('SVM.pkl')

# Define feature options
cp_options = {    
    1: 'Typical angina (1)',    
    2: 'Atypical angina (2)',    
    3: 'Non-anginal pain (3)',    
    4: 'Asymptomatic (4)'
    }
restecg_options = {    
    0: 'Normal (0)',    
    1: 'ST-T wave abnormality (1)',   
    2: 'Left ventricular hypertrophy (2)'
    }
slope_options = {    
    1: 'Upsloping (1)',   
    2: 'Flat (2)',
    3: 'Downsloping (3)'
    }
thal_options = {    
    1: 'Normal (1)',    
    2: 'Fixed defect (2)',    
    3: 'Reversible defect (3)'
    }


# Define feature names
#feature_names = [    "Age", "Sex", "Chest Pain Type", "Resting Blood Pressure", "Serum Cholesterol",    "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate", "Exercise Induced Angina",    "ST Depression", "Slope", "Number of Vessels", "Thal"]
# Streamlit user interface
st.title("Probability predictor of sepsis and mortality in ICU patients with COVID-19")
# Choose  the target
Choice=st.selectbox("Predict the target",options=['0','1'],format_func=lambda x: 'Sepsis' if x == 0 else 'mortality')
if Choice==0:
    feature_names = ['agp_min', 'bun', 'hospstay_seq', 'sbp_min0', 'po2', 'resp_avg', 'DOP', 'hr_max', 'agp_max', 'los_icu', 'resp_max', 'ast_max', 'ph', 'creatinine', 'inr_avg', 'ventilation', 'pco2', 'Gender', 'gcs', 'resp_var']
    #
    agp_min = st.number_input("",value=0.50)
    #
    bun = st.number_input("",value=0.50)
    #
    hospstay_seq = st.number_input("",value=0.50)
    #
    sbp_min_value = st.number_input("",value=0.50)
    #
    po2 = st.number_input("",value=0.50)
    #
    resp_rate_avg_value = st.number_input("",value=0.50)
    #
    Disease_Onset_Period = st.number_input("",value=0.50)
    #
    hr_max_value = st.number_input("",value=0.50)
    #
    agp_max = st.number_input("",value=0.50)
    #
    los_icu = st.number_input("",value=0.50)
    #
    resp_rate_max_value = st.number_input("",value=0.50)
    #
    ast_max = st.number_input("",value=0.50)
    #
    ph = st.number_input("",value=0.50)
    #
    creatinine = st.number_input("",value=0.50)
    # 
    inr_avg = st.number_input("",value=0.50)
    # 
    ventilation= st.number_input("",value=0.50)
    # 
    pco2 = st.number_input("",value=0.50)
    # 
    Gender = st.selectbox("Sex (0=Female, 1=Male):", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')
    # 
    gcs = st.number_input("",value=0.50)
    # 
    resp_rate_variance_value = st.number_input("",value=0.50)
    # 
    feature_values = [agp_min, bun, hospstay_seq, sbp_min_value, po2, resp_rate_avg_value, Disease_Onset_Period, hr_max_value, agp_max, los_icu, resp_rate_max_value, ast_max, ph, creatinine, inr_avg, ventilation, pco2, Gender, gcs, resp_rate_variance_value]
    features = np.array([feature_values])
    if st.button("Predict"):    
        # Predict class and probabilities    
        predicted_class = Sepsismodel.predict(features)    
        predicted_proba = Sepsismodel.predict_proba(features)

        # Display prediction results    
        st.write(f"**Predicted Class:** {predicted_class}")    
        st.write(f"**Prediction Probabilities:** {predicted_proba}")

        # Generate advice based on prediction results    
        probability = predicted_proba[predicted_class] * 100

        if predicted_class == 1:        
            advice = (            
                f"According to our model, patient has a high risk of sepsis. "            
                f"The model predicts that patient's probability of sepsis is {probability:.1f}%. "            
                # "While this is just an estimate, it suggests that you may be at significant risk. "           
                # "I recommend that you consult a cardiologist as soon as possible for further evaluation and "            
                # "to ensure you receive an accurate diagnosis and necessary treatment."        
                )    
        else:        
            advice = (            
                f"According to our model, patient has a low risk of heart disease. "            
                f"The model predicts that patient's probability of sepsis is {probability:.1f}%. "            
                # "However, maintaining a healthy lifestyle is still very important. "            
                # "I recommend regular check-ups to monitor your heart health, "            
                # "and to seek medical advice promptly if you experience any symptoms."        
                )
        st.write(advice)

        # Calculate SHAP values and display force plot    
        explainer = shap.TreeExplainer(Sepsismodel)    
        shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
        shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)    
        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
        st.image("shap_force_plot.png")

# Choose target "Death" 
else:
    #
    feature_names = ['resp_median', 'sbp_min', 'agp_avg', 'hr_max', 'icustay_seq', 'DOP', 'agp_min', 'lactate_mean', 'Age', 'wbc', 'resp_max', 'bun', 'hr_min', 'creatinine', 'sbp_var', 'ventilation', 'pco2', 'totalco2', 'weight', 'gcs']
    resp_rate_median_value = st.number_input("", min_value=1, max_value=120, value=0.50)
    #
    sbp_min_value = st.number_input("",value=0.50)
    #
    agp_avg = st.number_input("",value=0.50)
    #
    hr_max_value = st.number_input("",value=0.50)
    #
    icustay_seq = st.number_input("",value=0.50)
    #
    Disease_Onset_Period = st.number_input("",value=0.50)
    #
    agp_min = st.number_input("",value=0.50)
    #
    lactate_mean = st.number_input("",value=0.50)
    #
    Age = st.number_input("",value=0.50)
    #
    wbc = st.number_input("",value=0.50)
    # 
    resp_rate_max_value = st.number_input("",value=0.50)
    #
    bun = st.number_input("",value=0.50)
    #
    hr_min_value = st.number_input("",value=0.50)
    #
    creatinine_baseline = st.number_input("",value=0.50)
    #
    sbp_variance_value = st.number_input("",value=0.50)
    # 
    ventilation = st.number_input("",value=0.50)
    # 
    pco2= st.number_input("",value=0.50)
    # 
    totalco2 = st.number_input("",value=0.50)
    # 
    weight = st.number_input("",value=0.50)
    # 
    gcs = st.number_input("",value=0.50)

    feature_values = [resp_rate_median_value,sbp_min_value,agp_avg,hr_max_value,icustay_seq,Disease_Onset_Period,agp_min,lactate_mean,Age,wbc,resp_rate_max_value,bun,hr_min_value,creatinine_baseline,sbp_variance_value,ventilation,pco2,totalco2,weight,gcs]
    features = np.array([feature_values])    

    if st.button("Predict"):    
        # Predict class and probabilities    
        predicted_class = Deathmodel.predict(features)[0]    
        predicted_proba = Deathmodel.predict_proba(features)[0]

        # Display prediction results    
        st.write(f"**Predicted Class:** {predicted_class}")    
        st.write(f"**Prediction Probabilities:** {predicted_proba}")

        # Generate advice based on prediction results    
        probability = predicted_proba[predicted_class] * 100

        if predicted_class == 1:        
            advice = (            
                f"According to our model, patient has a high risk of mortality. "            
                f"The model predicts that patient's probability of mortality is {probability:.1f}%. "            
                # "While this is just an estimate, it suggests that you may be at significant risk. "           
                # "I recommend that you consult a cardiologist as soon as possible for further evaluation and "            
                # "to ensure you receive an accurate diagnosis and necessary treatment."        
                )    
        else:        
            advice = (            
                f"According to our model, you have a low risk of mortality. "            
                f"The model predicts that patient's probability of not mortality is {probability:.1f}%. "            
                # "However, maintaining a healthy lifestyle is still very important. "            
                # "I recommend regular check-ups to monitor your heart health, "            
                # "and to seek medical advice promptly if you experience any symptoms."        
                )
        st.write(advice)
        K = 100
        X_train=pd.read_csv('COVID_Filled_cut_missing_row.csv')
        X_train_kmeans = shap.kmeans(X_train, K)

        explainer = shap.KernelExplainer(Deathmodel.predict_proba , X_train_kmeans)
        shap_values = explainer.shap_values(pd.DataFrame([feature_values],columns=feature_names))
        shap.force_plot(explainer.expected_value[0], shap_values[0,:,0], pd.DataFrame([feature_values],columns=feature_names), matplotlib=True)   
        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
        st.image("shap_force_plot.png")
