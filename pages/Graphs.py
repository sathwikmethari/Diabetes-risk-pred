import pandas as pd
import streamlit as st
data=pd.read_csv("datasets/diabetes_data_upload.csv")
cat_columns=data.select_dtypes(include='object').columns
feature_selected = st.selectbox("Select a feature to plot:", cat_columns)

# Plot the selected feature against the target class
if feature_selected:
    st.image(f"plots/{feature_selected}_countplot.png")