##########################################################################################分割线
##########################################################################################可段代码可能需要单独运行
#streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


df=pd.read_csv('COVID_Filled_cut_missing_row.csv')
Sepsis_top_feature=list(pd.read_csv("top_20_features_Sepsis.csv").feature)
XS = df[Sepsis_top_feature]
df1=df[df['Is_Sepsis_Yes1No0']==1]
Death_top_feature=list(pd.read_csv("top_20_features_death1.csv").feature)
XD = df1[Death_top_feature]

scalerS = StandardScaler()
X_scaledS = scalerS.fit_transform(XS)
scalerD = StandardScaler()
X_scaledD = scalerD.fit_transform(XD)



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
Choice=st.selectbox("Predict the target",options=[0,1],format_func=lambda x: 'Sepsis' if x == 0 else 'mortality')
if Choice==0:
    feature_names = ['agp_min', 'bun', 'hospstay_seq', 'sbp_min0', 'po2', 'resp_avg', 'DOP', 'hr_max', 'agp_max', 'los_icu', 'resp_max', 'ast_max', 'ph', 'creatinine', 'inr_avg', 'ventilation', 'pco2', 'Gender', 'gcs', 'resp_var']
    #
    # Add unique key for each number_input
    agp_min = st.number_input("AGP Min: Anion gap(mEq/L)", value=8.27, min_value=-10.00, max_value=33.00, key="agp_min")
    bun = st.number_input("BUN: The lastest recorded blood urea nitrogen(mg/dL)", value=27.17, min_value=1.00, max_value=180.00, key="bun")
    hospstay_seq = st.number_input("Hospital Stay Sequence(No unit)", value=1.59, min_value=1.00, max_value=20.00, key="hospstay_seq")
    sbp_min_value = st.number_input("SBP Min: systolic blood pressure(mmHg)", value=84.51, min_value=6.00, max_value=161.00, key="sbp_min_value")
    po2 = st.number_input("PO2: Lastest recorded (mmHg)", value=126.43, min_value=15.00, max_value=536.00, key="po2")
    resp_rate_avg_value = st.number_input("Resp Rate Avg: respiratory rate(insp/min)", value=19.75, min_value=10.16, max_value=41.00, key="resp_rate_avg_value")
    Disease_Onset_Period = st.number_input("Disease Onset Period(DOP): Hours between hospital admit and ICU admit(hours)", value=15.18, min_value=0.00, max_value=186.00, key="Disease_Onset_Period")
    hr_max_value = st.number_input("HR Max: Heart rate (bpm)", value=113.25, min_value=54.00, max_value=282.00, key="hr_max_value")
    agp_max = st.number_input("AGP Max: Anion gap(mEq/L)", value=16.83, min_value=5.00, max_value=49.00, key="agp_max")
    los_icu = st.number_input("LOS ICU: ICU length of stay (hours)", value=5.13, min_value=0.01, max_value=136.03, key="los_icu")
    resp_rate_max_value = st.number_input("Resp Rate Max: respiratory rate(insp/min)", value=32.21, min_value=14.00, max_value=67.00, key="resp_rate_max_value")
    ast_max = st.number_input("AST Max: aspartate aminotransferase(IU/L)", value=258.56, min_value=1.24, max_value=16000.00, key="ast_max")
    ph = st.number_input("PH: (No unit)", value=7.35, min_value=6.95, max_value=7.70, key="ph")
    creatinine = st.number_input("Creatinine: Lastest record(mg/dL)", value=1.52, min_value=0.20, max_value=21.50, key="creatinine")
    inr_avg = st.number_input("INR Avg: international normalized ratio(No unit)", value=1.56, min_value=0.90, max_value=500000.05, key="inr_avg")
    ventilation = st.selectbox("ventilation(Yes or No)", options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key="ventilation")
    pco2 = st.number_input("PCO2: Lastest recorded (mmHg)", value=44.41, min_value=0.00, max_value=104.00, key="pco2")
    Gender = st.selectbox("Sex (0=Female, 1=Male):", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)', key="Gender")
    gcs = st.number_input("GCS: Glasgow Coma Scale(No unit)", value=14.22, min_value=3.00, max_value=15.00, key="gcs")
    resp_rate_variance_value = st.number_input("Resp Rate Variance: respiratory rate(No unit)", value=20.56, min_value=0.00, max_value=119.17, key="resp_rate_variance_value")
    Ddata = round(pd.read_csv("CovidDataDiscribe.csv",index_col=0),2)
    feature_values = [agp_min, bun, hospstay_seq, sbp_min_value, po2, resp_rate_avg_value, Disease_Onset_Period, hr_max_value, agp_max, los_icu, resp_rate_max_value, ast_max, ph, creatinine, inr_avg, ventilation, pco2, Gender, gcs, resp_rate_variance_value]
    NormData = scalerS.transform([feature_values])
    features = np.array(NormData)
    if st.button("Predict"):    
        # Predict class and probabilities    
        predicted_class = Sepsismodel.predict(features)[0]   
        predicted_proba = Sepsismodel.predict_proba(features)[0]

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
        string1="please wait for the generation of the SHAP force plot"
        st.write(string1)
        explainer = shap.TreeExplainer(Sepsismodel)    
        shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
        shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)    
        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
        st.image("shap_force_plot.png")

# Choose target "Death" 
else:
    #
    feature_names1 = ['res_med', 'sbp_min', 'agp_avg', 'hr_max', 'icu_seq', 'DOP', 'agp_min', 'lac_mea', 'Age', 'wbc', 'res_max', 'bun', 'hr_min', 'cre', 'sbp_var', 'ven', 'pco2', 'tco2', 'wei', 'gcs']
    resp_rate_median_value1 = st.number_input("resp rate median value:respiratory rate(insp/min)", value=26.0, min_value=11.0, max_value=43.5, key='resp_rate_median_value1')
    sbp_min_value1 = st.number_input("sbp min value: systolic blood pressure(mmHg)", value=46.0, min_value=6.0, max_value=149.0, key='sbp_min_value1')
    agp_avg1 = st.number_input("agp avg1:Anion gap(mEq/L)", value=20.00, min_value=4.0, max_value=29.0, key='agp_avg1')
    hr_max_value1 = st.number_input("hr max value1:Heart rate (bpm)", value=166.0, min_value=66.0, max_value=282.0, key='hr_max_value1')
    icustay_seq1 = st.number_input("icustay seq:ICU sequence (No unit):", value=1.0, min_value=1.0, max_value=5.0, key='icustay_seq1')
    Disease_Onset_Period1 = st.number_input("Disease Onset Period(DOP):Hours between hospital admit and ICU admit(hours)", value=9.0, min_value=0.0, max_value=186.0, key='Disease_Onset_Period1')
    agp_min1 = st.number_input("agp min:Anion gap(mEq/L)", value=16.0, min_value=-10.0, max_value=33.0, key='agp_min1')
    lactate_mean1 = st.number_input("lactate mean (mg)", value=2.4, min_value=0.3, max_value=27.0, key='lactate_mean1')
    Age1 = st.number_input("Age: Lastest record(year)", value=93.0, min_value=18.0, max_value=102.0, key='Age1')
    wbc1 = st.number_input("wbc: Lastest recorded white blood count(K/uL)", value=10.5, min_value=0.1, max_value=92.2, key='wbc1')
    resp_rate_max_value1 = st.number_input("resp rate max value:respiratory rate(insp/min)", value=37.0, min_value=15.0, max_value=536.0, key='resp_rate_max_value1')
    bun1 = st.number_input("bun: The lastest recorded blood urea nitrogen(mg/dL)", value=88.0, min_value=3.0, max_value=180.0, key='bun1')
    hr_min_value1 = st.number_input("HR(Heart Rate) min value (bpm)", value=92.0, min_value=7.0, max_value=149.0, key='hr_min_value1')
    creatinine_baseline1 = st.number_input("creatinine baseline(mg/dL):", value=1.5, min_value=0.1, max_value=16.2, key='creatinine_baseline1')
    sbp_variance_value1 = st.number_input("sbp variance value: systolic blood pressure(mmHg)", value=668.5, min_value=0.5, max_value=2550.7, key='sbp_variance_value1')
    ventilation1 = st.selectbox("ventilation(Yes or No)", options=[0.0, 1.0], format_func=lambda x: 'No' if x == 0.0 else 'Yes', key="ventilation1")
    pco21 = st.number_input("pco2:Lastest recorded (mmHg)", value=38.0, min_value=0.0, max_value=104.0, key='pco21')
    totalco21 = st.number_input("totalco2:The lastest recorded total of CO2(mEq/L)", value=26.0, min_value=4.0, max_value=51.0, key='totalco21')
    weight1 = st.number_input("weight:The lastest recorded weight(kg)", value=96.1, min_value=55.5, max_value=190.0, key='weight1')
    gcs1 = st.number_input("gcs: Glasgow Coma Scale(No unit)", value=13.0, min_value=3.0, max_value=15.0, key='gcs1')

    #Ddata1 = round(pd.read_csv("CovidDataDeathDiscribe.csv",index_col=0),2)
    feature_values1 = [resp_rate_median_value1,sbp_min_value1,agp_avg1,hr_max_value1,icustay_seq1,Disease_Onset_Period1,agp_min1,lactate_mean1,Age1,wbc1,resp_rate_max_value1,bun1,hr_min_value1,creatinine_baseline1,sbp_variance_value1,ventilation1,pco21,totalco21,weight1,gcs1]
 
    features1_df = pd.DataFrame([feature_values1], columns=XD.columns)                                              # 将 features1 转换成 DataFrame，确保列名一致
    
                                                                                    # 3. 使用已训练的 scaler 对 features1 进行标准化
    NormData1 = scalerD.transform(features1_df)  
                    #NormData1 = scalerD.transform([feature_values1])c
    
    normdata1_list = NormData1.flatten().tolist()
    NormData2 = [round(i,1) for i in normdata1_list]
    #NormData2=[0.51, -0.55, 0.48, -0.58, -0.3, 0.26, 0.32, -0.06, 0.71, 0.31, -0.56, 0.35, -0.55, -0.03, 0.32, 1.15, 0.42, -0.37, 0.51, 0.4]
    
    features1 = NormData2 


    if st.button("Predict"):    
        # Predict class and probabilities    
        predicted_class = Deathmodel.predict([features1])[0]    
        predicted_proba = Deathmodel.predict_proba([features1])[0]

        # Display prediction results    

        st.write(f"**Predicted Class:** {predicted_class}")    
        st.write(f"**Prediction Probabilities:** {[predicted_proba[1],predicted_proba[0]]}")

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
        Death_top_feature=list(pd.read_csv("top_20_features_death1.csv").feature)
        X_train1=X_train[Death_top_feature]
        X_train_kmeans = shap.kmeans(X_train1, K)
        string1="please wait for the generation of the SHAP force plot"
        st.write(string1)
        explainer1 = shap.KernelExplainer(Deathmodel.predict_proba , X_train_kmeans)
        shap_values1 = explainer1.shap_values(pd.DataFrame([features1],columns=feature_names1))
        shap.force_plot(explainer1.expected_value[0], shap_values1[0,:,0], pd.DataFrame([features1],columns=feature_names1), matplotlib=True)   
        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
        st.image("shap_force_plot.png")
        string1=''
