import streamlit as st
import shap
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load your pre-trained model and data
# Replace with your actual model and data loading
# best_xgb_model = xgb.Booster()  # Load your trained model here
# X_balanced = pd.read_csv('X_balanced.csv')  # Assuming you have it saved
# data = pd.read_csv('data.csv')  # Data with pitcher names in the same order as X_balanced

# To load the model later
with open('best_xgb_model.pkl', 'rb') as file:
    best_xgb_model = pickle.load(file)

X_balanced = pd.read_csv('data/X_balanced.csv', index_col=0)
data = pd.read_csv('data/data_balanced.csv')


# Create SHAP explainer
explainer = shap.Explainer(best_xgb_model, X_balanced)

# Streamlit app
st.title("SHAP Analysis for Pitchers")

# Dropdown for selecting a pitcher
# Sort pitcher names alphabetically
pitcher_names = sorted(data['pitcher'].unique())  # Sorting the unique pitcher names

# Dropdown for selecting a pitcher
selected_pitcher = st.selectbox("Choose a Pitcher:", pitcher_names)

# Filter data for the selected pitcher
pitcher_indices = data[data['pitcher'] == selected_pitcher].index
pitcher_data = X_balanced.iloc[pitcher_indices]

# SHAP Summary Plot
st.subheader(f"SHAP Summary Plot for {selected_pitcher}")
shap_values = explainer(pitcher_data)
fig_summary, ax = plt.subplots()
shap.summary_plot(shap_values, pitcher_data, show=False)
st.pyplot(fig_summary)

# Dropdown to select a specific pitch instance
pitch_indices = list(pitcher_indices)
selected_index = st.selectbox("Select a Pitch Instance:", pitch_indices)

# SHAP Waterfall Plot for a single observation
st.subheader(f"SHAP Waterfall Plot for Pitch Instance {selected_index}")
single_instance = X_balanced.iloc[[selected_index]]
shap_values_single = explainer(single_instance)
fig_waterfall, ax = plt.subplots()
shap.waterfall_plot(shap_values_single[0], show=False)
st.pyplot(fig_waterfall)