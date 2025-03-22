import pandas as pd
import torch
device='cuda' if torch.cuda.is_available() else 'cpu'
#print(device)
import pickle
import streamlit as st
from pytorchclassifier import ClassificationModel, encoding
from pytorchclassifier import ClassificationModel, encoding
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    layout="wide"    
)


@st.cache_data
def load_models():
    scaler=pickle.load(open('models/pytorch_scaler.pkl','rb'))
    columns_dict=pickle.load(open('models/columns_dict.pkl','rb'))
    #Its is not recommended to use pickle to load pytorch models, but for this simple model it is fine
    #try not to use weights_only=False.
    pytorch_model=torch.load('models/pytorch_model.pt',weights_only=False)
    return scaler, columns_dict, pytorch_model

scaler, columns_dict, pytorch_model = load_models()
#print("Models loaded successfully")

pytorch_model.to(device)
Input_dict={}
for column in columns_dict['columns']:
     Input_dict[column]=0
  
st.title("Diabetes risk prediction (PYTORCH)")

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

container = st.container()
container.write("RESULT :")

if submitted:
     for column in columns_dict['columns']:
          Input_dict[column]=st.session_state[column]
     Input_df=pd.DataFrame(Input_dict, index=[0])
     encoding(Input_df,columns_dict['cat_columns'])
     for column in columns_dict['num_columns']:
          Input_df[column]=scaler.transform(Input_df[[column]])
     Input_tensor=torch.from_numpy(Input_df.iloc[:,:].values).to(torch.float32).to(device)
     pytorch_model.eval()
     with torch.inference_mode():
          prediction_logits=pytorch_model(Input_tensor).squeeze()
          prediction=torch.round(torch.sigmoid(prediction_logits))
          prediction.cpu().detach().numpy()
     container.write(prediction)
     if prediction==1:
          container.write("You have a very HIGH risk of getting Diabetes")
     else:
          container.write("You have a no risk of getting Diabetes, for now")