Hackathon Success Predictor AI

"Predicting hackathon outcomes by modeling the human dynamics behind high-performing teams."

Hackathon Success Predictor AI is a machine learning–powered analytics dashboard that forecasts team performance using behavioral telemetry rather than just technical skill. The system analyzes commit velocity, communication frequency, and sleep schedules to estimate a team’s probability of winning a hackathon.

Predictions and strategic insights are delivered in real time through an interactive Streamlit web application.

Features

Real-Time Prediction Engine
Instantly calculates win probability based on user-provided metrics.

6-Tier Classification System
Categorizes teams from Critical Failure to Elite / Dominating.

Explainable AI (XAI)
Uses SHAP (SHapley Additive Explanations) to visualize why a prediction was made
(e.g., score reduction due to sleep deprivation).

Strategic AI Assessment
Generates actionable feedback to improve team odds
(e.g., increase communication to more than 50 messages per hour).

Bottleneck Detection
Automatically identifies the primary constraint impacting performance
(Sleep vs. Velocity vs. Communication).

Tech Stack

Frontend: Streamlit (Python)

Machine Learning: XGBoost Classifier, Scikit-Learn

Explainability: SHAP

Data Processing: Pandas, NumPy

Persistence: Joblib

Installation
1. Clone the Repository
git clone https://github.com/manikanta7cheruku/Hackathon-Prediction.git
cd Hackathon-Prediction

2. Create a Virtual Environment (Recommended)

Windows

py -3.11 -m venv venv
venv\Scripts\activate


Mac / Linux

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the Application
streamlit run app.py

Usage

Launch the Streamlit application.

Input team metrics such as commit velocity, communication rate, and sleep hours.

View predicted win probability, classification tier, and SHAP-based explanations.

Apply strategic recommendations to improve predicted outcomes.

Notes

Designed for hackathon analytics and experimentation.

Python 3.11 is recommended for best compatibility.