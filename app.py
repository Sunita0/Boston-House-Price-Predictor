import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("random_forest_model.pkl")

# Page config
st.set_page_config(page_title="Boston House Price Predictor", page_icon="🏡")

# Title and intro
st.title("🏡 Boston House Price Predictor")
st.markdown(
    """
    Welcome! This app predicts the estimated **house price in Boston** based on key factors like crime rate, number of rooms, school quality, and more.

    Fill in the property details below, and click **"Predict House Price"** to get an estimate. 💰
    """
)

st.markdown("---")
st.header("📋 Property Details")

# Input fields with explanations
crim = st.number_input("🏙️ Crime Rate (CRIM)", min_value=0.0, value=0.1, format="%.4f",
                       help="Per capita crime rate by town")
indus = st.number_input("🏭 Industrial Proportion (INDUS)", min_value=0.0, value=7.0,
                        help="Proportion of non-retail business acres per town")
nox = st.number_input("🌫️ Air Pollution Level (NOX)", min_value=0.0, max_value=1.0, value=0.5, format="%.3f",
                      help="Nitric oxide concentration (parts per 10 million)")
rm = st.number_input("🛏️ Average Rooms (RM)", min_value=1.0, max_value=10.0, value=6.0, format="%.2f",
                     help="Average number of rooms per dwelling")
age = st.number_input("🏚️ Older Buildings (%) (AGE)", min_value=0.0, max_value=100.0, value=60.0,
                      help="Proportion of units built before 1940")
dis = st.number_input("🛣️ Distance to Employment (DIS)", min_value=1.0, value=4.0, format="%.2f",
                      help="Distance to 5 Boston employment centres")
ptratio = st.number_input("👩‍🏫 Pupil-Teacher Ratio (PTRATIO)", min_value=10.0, max_value=25.0, value=18.0,
                          help="Student-teacher ratio by town")
lstat = st.number_input("📉 Lower Income Population (%) (LSTAT)", min_value=1.0, value=12.0, format="%.2f",
                        help="Percentage of population considered lower status")

# Transform inputs (as done during training)
crim_log = np.log1p(crim)
dis_sqrt = np.sqrt(dis)
lstat_log = np.log1p(lstat)

# Input array for prediction (8 features)
input_data = np.array([[crim_log, indus, nox, rm, age, dis_sqrt, ptratio, lstat_log]])

# Prediction button
st.markdown("---")
if st.button("🔍 Predict House Price"):
    prediction_sqrt = model.predict(input_data)[0]
    predicted_price = prediction_sqrt ** 2  # reverse sqrt

    st.markdown(
        f"""
        ## 💰 Estimated Price: **${predicted_price * 1000:,.2f}**
        > _(in US dollars)_
        """
    )
    st.success("Prediction complete! Feel free to adjust inputs and try again.")

# Footer
st.markdown("---")
st.caption("📊 Model: Random Forest Regressor trained on Boston Housing Dataset")

