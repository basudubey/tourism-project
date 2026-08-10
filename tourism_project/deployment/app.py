"""
Streamlit App - Wellness Tourism Package Purchase Predictor
--------------------------------------------------------------
Loads the model committed by the pipeline (sits next to this file),
collects customer details from the user into a DataFrame, and predicts
whether the customer is likely to purchase the Wellness Tourism Package.
"""

import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Must match the binning used in model_building/prep.py so AgeGroup is
# computed identically at training time and at inference time
AGE_BINS = [17, 25, 35, 45, 55, 100]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56+"]

st.title("Wellness Tourism Package - Purchase Predictor")
st.write(
    "Enter the customer's details below to predict whether they are "
    "likely to purchase the newly introduced Wellness Tourism Package."
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    occupation = st.selectbox(
        "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    # Note: 'Single' also covers customers who might describe themselves as
    # 'Unmarried' -- the two are merged into one category during training
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input("Monthly Income", min_value=0, value=20000, step=500)

with col2:
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, value=15)
    product_pitched = st.selectbox(
        "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )
    num_persons = st.number_input("Number of Persons Visiting", min_value=1, value=2)
    num_children = st.number_input("Number of Children Visiting", min_value=0, value=0)
    num_followups = st.number_input("Number of Followups", min_value=0, value=3)
    num_trips = st.number_input("Number of Trips (avg/year)", min_value=0, value=2)
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    passport = st.selectbox("Has Passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a Car?", ["Yes", "No"])

if st.button("Predict"):
    age_group = pd.cut([age], bins=AGE_BINS, labels=AGE_LABELS)[0]

    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_persons,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
        "AgeGroup": age_group,
    }])

    # Match the 0.45 decision threshold used during training, which favors
    # recall on the minority (purchase) class
    proba = model.predict_proba(input_data)[0][1]
    prediction = int(proba >= 0.45)

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"Likely to **PURCHASE** the package (confidence: {proba:.1%})")
    else:
        st.warning(f"Unlikely to purchase the package (purchase probability: {proba:.1%})")
