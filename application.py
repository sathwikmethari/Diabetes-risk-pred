import pandas as pd
import pickle
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
st.set_page_config(
    layout="wide"    
)

data=pd.read_csv("datasets/diabetes_data_upload.csv")
x=data.drop(['class'],axis=1)
columns=[col for col in x.columns]
Input_dict={}
for column in columns:
     Input_dict[column]=0


@st.cache_data
def load_models():
    Transformer=pickle.load(open('models/Transformer.pkl','rb'))
    Encoder=pickle.load(open('models/Encoder.pkl','rb'))
    Classifier=pickle.load(open('models/Classifier.pkl','rb'))
    return Transformer, Encoder, Classifier
Transformer, Encoder, Classifier = load_models()

st.title("Early Stage Diabetes risk prediction")

st.sidebar.title("FAQ")
st.sidebar.link_button("What is Polyuria?", "https://en.wikipedia.org/wiki/Polyuria")
st.sidebar.link_button("What is Polydipsia?", "https://en.wikipedia.org/wiki/Polydipsia")
st.sidebar.link_button("What is Polyphagia?", "https://en.wikipedia.org/wiki/Polyphagia")

st.sidebar.link_button("What is Genital thrush?", "https://www.healthdirect.gov.au/genital-thrush-in-males#what-is")
st.sidebar.link_button("What is Partial Paresis?", "https://www.healthline.com/health/paresis#definition")
st.sidebar.link_button("What is Alopecia?", "https://en.wikipedia.org/wiki/Alopecia_areata")


with st.form("my_form"):
    left, middle, right = st.columns(3, gap="large", vertical_alignment="top")
    Age=left.number_input("Enter your age", min_value=10, step=1, key="Age")
    middle.radio("Select your gender", ["Male", "Female"], key="Gender")
    right.radio("Do you have Polyuria?", ["Yes", "No"], key="Polyuria")

    left2, middle2, right2 = st.columns(3, gap="large", vertical_alignment="top")
    left2.radio("Do you have Polydipsia?", ["Yes", "No"], key="Polydipsia")
    middle2.radio("Was there any sudden weightloss?", ["Yes", "No"], key="sudden weight loss")
    right2.radio("Do you have weakness?", ["Yes", "No"], key="weakness")

    left3, middle3, right3 = st.columns(3, gap="large", vertical_alignment="top")
    left3.radio("Do you have Polyphagia?", ["Yes", "No"], key="Polyphagia")
    middle3.radio("Do you have Genital thrush?", ["Yes", "No"], key="Genital thrush")
    right3.radio("Do you have visual blurring?", ["Yes", "No"], key="visual blurring")

    left4, middle4, right4 = st.columns(3, gap="large", vertical_alignment="top")
    left4.radio("Do you feel like Itching?", ["Yes", "No"], key="Itching")
    middle4.radio("Do you have Irritability?", ["Yes", "No"], key="Irritability")
    right4.radio("Do you have delayed healing?", ["Yes", "No"], key="delayed healing")

    left5, middle5, right5 = st.columns(3, gap="large", vertical_alignment="top")
    left5.radio("Do you have Partial Paresis?", ["Yes", "No"], key="partial paresis")
    middle5.radio("Do you have muscle stiffness?", ["Yes", "No"], key="muscle stiffness")
    right5.radio("Do you have Alopecia?", ["Yes", "No"], key="Alopecia")

    left6=st.columns(1)
    left6=st.radio("Do you suffer from Obesity?", ["Yes", "No"], key="Obesity")
    #Inputdata=[Age]
    submitted = st.form_submit_button("Submit")  

container = st.container(border=True)
container.write("RESULT :")

if submitted:
        for column in columns:
             Input_dict[column]=st.session_state[column]
        Input_df=pd.DataFrame(Input_dict, index=[0])
        prediction=Encoder.inverse_transform(Classifier.predict(Transformer.transform(Input_df)))
        if prediction=='Positive':
             container.write("You have a very HIGH risk of getting Diabetes")
        else:
             container.write("You have a no risk of getting Diabetes, for now")