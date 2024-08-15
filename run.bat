@echo off
pip install -r requirements.txt
streamlit run application.py
start "" http://localhost:8501