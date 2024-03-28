import streamlit as st
import numpy as np
import joblib

# モデルのロード
model_1m = joblib.load('toeic_prediction_model_1m.joblib')
model_3m = joblib.load('toeic_prediction_model_3m.joblib')
model_6m = joblib.load('toeic_prediction_model_16m.joblib')
model_12m = joblib.load('toeic_prediction_model_20m.joblib')

# タイトルと説明
st.title('TOEIC Score Prediction App')
st.write('This app predicts the TOEIC score at various future points based on your study hours and initial score.')

# ユーザー入力
study_hours = st.number_input('Enter your daily study hours:', min_value=0.0, max_value=24.0, value=5.0)
initial_score = st.number_input('Enter your initial TOEIC score:', min_value=0, max_value=990, value=350)

# 予測ボタン
if st.button('Predict'):
    input_data = np.array([[study_hours, initial_score]])
    
    # 各時点での予測実行
    predicted_scores = [
        model_1m.predict(input_data)[0],
        model_3m.predict(input_data)[0],
        model_6m.predict(input_data)[0],
        model_12m.predict(input_data)[0]
    ]

    # 予測結果を5点刻みに丸め、最大値を990点に制限
    rounded_scores = [min(990, round(score / 5) * 5) for score in predicted_scores]

    # 結果の表示
    st.write(f'Predicted TOEIC score after 1 month: {rounded_scores[0]:.2f}')
    st.write(f'Predicted TOEIC score after 3 months: {rounded_scores[1]:.2f}')
    st.write(f'Predicted TOEIC score after 6 months: {rounded_scores[2]:.2f}')
    st.write(f'Predicted TOEIC score after 12 months: {rounded_scores[3]:.2f}')
