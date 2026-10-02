# ☀️ Solar Power Generation Prediction

ML model and Streamlit dashboard that predicts AC power output of two solar
plants from weather conditions and time of day.

## Setup
python -m venv venv
venv\Scripts\activate        (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python src/preprocess.py
python src/train_model.py
streamlit run app.py

## Dataset
Solar Power Generation Data (Kaggle) by Ani Rudra Sen Nandi.