import pickle
import time
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

diabetesModel = pickle.load(open(
    'diabetes_model.sav', 'rb'))
heartModel = pickle.load(open(
    'heart_model.sav', 'rb'))
parkinsonsModel = pickle.load(open(
    'parkionson_model.sav', 'rb'))


with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinson Disease Prediction'],

                           default_index=0,

                           icons=['activity', 'clipboard2-heart', 'person'])

if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction Using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input(
            'Number of Pregnancies', placeholder='Enter here')
        SkinThickness = st.text_input(
            'SkinThickness', placeholder='Enter here')
        DiabetesPedigreeFunction = st.text_input(
            'DiabetesPedigreeFunction', placeholder='Enter here')
    with col2:
        glucose = st.text_input('Glucose level', placeholder='Enter here')
        Insulin = st.text_input('Insulin level', placeholder='Enter here')
        Age = st.text_input('Age', placeholder='Enter here')
    with col3:
        BloodPressure = st.text_input(
            'BloodPressure', placeholder='Enter here')
        BMI = st.text_input('BMI', placeholder='Enter here')

    diagonsis = ''

    if st.button('Diabetes test Result'):
        with st.spinner('Wait...'):
            time.sleep(0.9)

        diagonsis = diabetesModel.predict(
            [Pregnancies, glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        if (diagonsis[0] == 0):
            print('The person is not diabetic')
        else:
            print('The person is diabetic')

    st.success(diagonsis)


if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction Using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age', placeholder='Enter here')
        trestbps = st.text_input(
            'Resting blood pressure(Trestbps)', placeholder='Enter here')
        restecg = st.selectbox('Resting Electrocardiographic(Restecg)', [
                               '0', '1', '2'], index=None)
        oldpeak = st.text_input(
            'ST Depression Induced By Exercise(Oldpeak)', placeholder='Enter here')
        thal = st.selectbox('Thal', ['1', '2', '3'], index=None)
    with col2:
        sex = st.selectbox('Male(1) and Female(0)', ['0', '1'], index=None)
        chol = st.text_input(
            'Serum cholestoral in mg/dl(chol)', placeholder='Enter here')
        thalach = st.text_input(
            'Maximum Heart Rate(Thalach)', placeholder='Enter here')
        slope = st.text_input(
            'Slope Of The Peak Exersice In ST Segement', placeholder='Enter here')
    with col3:
        cp = st.selectbox('Chest Pain Type(CP)', [
                          '0', '1', '2', '3'], index=None)
        fbs = st.selectbox('Fasting Blood Sugar(FBS)', ['0', '1'], index=None)
        exang = st.text_input(
            'Exercise Induced Angina(Exang)', placeholder='Enter here')
        ca = st.selectbox('Major Vessel Coloured By Flourocsopy', [
                          '0', '1', '2', '3'], index=None)
    cardio = ''

    if st.button('Heart Disease Test Result'):
        with st.spinner('Wait...'):
            time.sleep(0.9)

        # to much problem
        input_data = np.array([float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs),
                               float(restecg), float(thalach), float(exang), float(oldpeak), float(slope),
                               float(ca), float(thal)]).reshape(1, -1)

        cardio = heartModel.predict(input_data)
        # print(cardio)

        if (cardio[0] == 0):
            st.success('No Heart Disease')
        else:
            st.success('Heart Disease')


if selected == 'Parkinson Disease Prediction':
    st.title('Parkinson Disease Prediction Using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        MDVPFo = st.text_input('MDVP-Fo(Hz)', placeholder='Enter here')
        MDVPJitterPer = st.text_input(
            'MDVP-Jitter(%)', placeholder='Enter here')
        MDVPPPQ = st.text_input('MDVP-PPQ', placeholder='Enter here')
        MDVPShimmerdB = st.text_input(
            'MDVP-Shimmer(dB)', placeholder='Enter here')
        MDVPAPQ = st.text_input('MDVP-APQ', placeholder='Enter here')
        HNR = st.text_input('HNR', placeholder='Enter here')
        spread1 = st.text_input('spread1', placeholder='Enter here')
        PPE = st.text_input(
            'PPE - Three nonlinear measures of fundamental frequency variation', placeholder='Enter here')
    with col2:
        MDVPFhi = st.text_input('MDVP-Fhi(Hz)', placeholder='Enter here')
        MDVPJitterAbs = st.text_input(
            'MDVP-Jitter(Abs)', placeholder='Enter here')
        JitterDDPSeveral = st.text_input(
            'Jitter-DDP - Several', placeholder='Enter here')
        ShimmerAPQ3 = st.text_input('Shimmer-APQ3', placeholder='Enter here')
        ShimmerDDA = st.text_input('Shimmer-DDA', placeholder='Enter here')
        RPDE = st.text_input('RPDE', placeholder='Enter here')
        spread2 = st.text_input('spread2', placeholder='Enter here')
    with col3:
        MDVPflo = st.text_input('MDVP-Flo(Hz)', placeholder='Enter here')
        MDVPRAP = st.text_input('MDVP-RAP', placeholder='Enter here')
        MDVPShimmer = st.text_input('MDVP-Shimmer', placeholder='Enter here')
        ShimmerAPQ5 = st.text_input('Shimmer-APQ5', placeholder='Enter here')
        NHR = st.text_input('NHR', placeholder='Enter here')
        DFA = st.text_input(
            'DFA - Signal fractal scaling exponent', placeholder='Enter here')
        D2 = st.text_input('D2', placeholder='Enter here')
    parkinson = ''

    if st.button('Parkinson Disease Test Result'):
        with st.spinner('Wait...'):
            time.sleep(0.9)

        # to much problem
        input_data = np.array([float(MDVPFo), float(MDVPFhi), float(MDVPflo), float(MDVPJitterPer), float(MDVPJitterAbs), float(MDVPRAP),
                               float(MDVPPPQ), float(JitterDDPSeveral), float(
                                   MDVPShimmer), float(MDVPShimmerdB), float(ShimmerAPQ3),
                               float(ShimmerAPQ5), float(MDVPAPQ), float(ShimmerDDA), float(NHR), float(HNR), float(RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)]).reshape(1, -1)

        parkinson = parkinsonsModel.predict(input_data)
        print(parkinson[0])

        if (parkinson[0] == 1):
            st.success('Person Does Have Parkinson Disease')
        else:
            st.success('Person Does Not Have Parkinson Disease')
