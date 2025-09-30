# Amazon Delivery Time Predictor 🚚⏱️

[Live Demo](https://amazon-delivery-predictor-lcup6rbytbwjnq6tns59gk.streamlit.app/)  

Machine learning project to estimate Amazon order delivery times using order, agent, and product features. Built with Python, trained on regression models, and deployed as an interactive Streamlit app.

---

## Overview
- Predicts delivery time (hours) based on distance, agent details, order timing, and product category.  
- Models: **Linear Regression, Random Forest, XGBoost** (best R² ≈ 0.61).  
- Experiment tracking with **MLflow**.  
- Deployed on **Streamlit Cloud** for real-time interaction.  

---

## Tech Stack
- **Python**: Pandas, NumPy, scikit-learn, XGBoost  
- **Visualization**: Matplotlib, Seaborn  
- **Experiment Tracking**: MLflow  
- **Deployment**: Streamlit  

---

## Run Locally
```bash
git clone https://github.com/JunaidKamate/amazon-delivery-predictor.git
cd amazon-delivery-predictor
pip install -r requirements.txt
streamlit run streamlit_app.py
Open: http://localhost:8501

---

## Author
**Junaid Kamate**
