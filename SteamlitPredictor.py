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
    # Add unique key for each number_input
    agp_min = st.number_input("AGP Min", value=0.50, key="agp_min")
    bun = st.number_input("BUN", value=0.50, key="bun")
    hospstay_seq = st.number_input("Hospital Stay Sequence", value=0.50, key="hospstay_seq")
    sbp_min_value = st.number_input("SBP Min", value=0.50, key="sbp_min_value")
    po2 = st.number_input("PO2", value=0.50, key="po2")
    resp_rate_avg_value = st.number_input("Resp Rate Avg", value=0.50, key="resp_rate_avg_value")
    Disease_Onset_Period = st.number_input("Disease Onset Period", value=0.50, key="Disease_Onset_Period")
    hr_max_value = st.number_input("HR Max", value=0.50, key="hr_max_value")
    agp_max = st.number_input("AGP Max", value=0.50, key="agp_max")
    los_icu = st.number_input("LOS ICU", value=0.50, key="los_icu")
    resp_rate_max_value = st.number_input("Resp Rate Max", value=0.50, key="resp_rate_max_value")
    ast_max = st.number_input("AST Max", value=0.50, key="ast_max")
    ph = st.number_input("PH", value=0.50, key="ph")
    creatinine = st.number_input("Creatinine", value=0.50, key="creatinine")
    inr_avg = st.number_input("INR Avg", value=0.50, key="inr_avg")
    ventilation = st.number_input("Ventilation", value=0.50, key="ventilation")
    pco2 = st.number_input("PCO2", value=0.50, key="pco2")
    Gender = st.selectbox("Sex (0=Female, 1=Male):", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)', key="Gender")
    gcs = st.number_input("GCS", value=0.50, key="gcs")
    resp_rate_variance_value = st.number_input("Resp Rate Variance", value=0.50, key="resp_rate_variance_value")

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
    feature_names1 = ['resp_median', 'sbp_min', 'agp_avg', 'hr_max', 'icustay_seq', 'DOP', 'agp_min', 'lactate_mean', 'Age', 'wbc', 'resp_max', 'bun', 'hr_min', 'creatinine', 'sbp_var', 'ventilation', 'pco2', 'totalco2', 'weight', 'gcs']
    resp_rate_median_value1 = st.number_input("",value=0.50,key='resp_rate_median_value1')
    #
    sbp_min_value1 = st.number_input("",value=0.50,key='')
    #
    agp_avg1 = st.number_input("",value=0.50,key='agp_avg1')
    #
    hr_max_value1 = st.number_input("",value=0.50,key='hr_max_value1')
    #
    icustay_seq1 = st.number_input("",value=0.50,key='icustay_seq1')
    #
    Disease_Onset_Period1 = st.number_input("",value=0.50,key='Disease_Onset_Period1')
    #
    agp_min1 = st.number_input("",value=0.50,key='agp_min1')
    #
    lactate_mean1 = st.number_input("",value=0.50,key='lactate_mean1')
    #
    Age1 = st.number_input("",value=0.50,key='Age1')
    #
    wbc1 = st.number_input("",value=0.50,key='wbc1')
    # 
    resp_rate_max_value1 = st.number_input("",value=0.50,key='resp_rate_max_value1')
    #
    bun1 = st.number_input("",value=0.50,key='bun1')
    #
    hr_min_value1 = st.number_input("",value=0.50,key='hr_min_value1')
    #
    creatinine_baseline1 = st.number_input("",value=0.50,key='creatinine_baseline1')
    #
    sbp_variance_value1 = st.number_input("",value=0.50,key='sbp_variance_value1')
    # 
    ventilation1 = st.number_input("",value=0.50,key='ventilation1')
    # 
    pco21 = st.number_input("",value=0.50,key='pco21')
    # 
    totalco21 = st.number_input("",value=0.50,key='totalco21')
    # 
    weight1 = st.number_input("",value=0.50,key='weight1')
    # 
    gcs1 = st.number_input("",value=0.50,key='gcs1')

    feature_values1 = [resp_rate_median_value1,sbp_min_value1,agp_avg1,hr_max_value1,icustay_seq1,Disease_Onset_Period1,agp_min1,lactate_mean1,Age1,wbc1,resp_rate_max_value1,bun1,hr_min_value1,creatinine_baseline1,sbp_variance_value1,ventilation1,pco21,totalco21,weight1,gcs1]
    features1 = np.array([feature_values1])    

    if st.button("Predict"):    
        # Predict class and probabilities    
        predicted_class = Deathmodel.predict(features1)[0]    
        predicted_proba = Deathmodel.predict_proba(features1)[0]

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

        explainer1 = shap.KernelExplainer(Deathmodel.predict_proba , X_train_kmeans)
        shap_values1 = explainer1.shap_values(pd.DataFrame([feature_values1],columns=feature_names1))
        shap.force_plot(explainer1.expected_value[0], shap_values1[0,:,0], pd.DataFrame([feature_values1],columns=feature_names1), matplotlib=True)   
        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
        st.image("shap_force_plot.png")
