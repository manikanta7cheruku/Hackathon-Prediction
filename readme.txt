# Hackathon Success Predictor AI

**A Predictive Analytics Dashboard to forecast Hackathon team performance using Multi-Factor Dynamics Analysis.**

## 📌 Project Overview
This project utilizes Machine Learning (**XGBoost**) to analyze behavioral telemetry data—such as **Commit Velocity**, **Communication Frequency**, and **Sleep Schedules**—to predict whether a team will win a Hackathon.

Unlike traditional models that look only at technical skills, this system evaluates the **"Human Dynamics"** that drive success, providing real-time strategic insights via a **Streamlit** web dashboard.

##Key Features
*   **Real-time Prediction Engine:** Instantly calculates Win Probability based on user inputs.
*   **6-Tier Classification System:** Categorizes teams from "Critical Failure" to "Elite / Dominating".
*   **Explainable AI (XAI):** Uses **SHAP (SHapley Additive exPlanations)** to visualize exactly *why* a prediction was made (e.g., "Score lowered due to Sleep Deprivation").
*   **Strategic AI Assessment:** Provides dynamic, actionable feedback to improve team odds (e.g., "Increase Communication to >50 msgs/hr").
*   **Bottleneck Detection:** Automatically identifies the primary constraint (Sleep vs. Velocity vs. Communication).

## 🛠️ ech Stack
*   **Frontend:** Streamlit (Python)
*   **Machine Learning:** XGBoost Classifier, Scikit-Learn
*   **Explainability:** SHAP
*   **Data Processing:** Pandas, NumPy
*   **Persistence:** Joblib

## How to Run Locally (For Teammates)

If you clone this repository, you must install the dependencies to run the app.

### 1. Clone the Repo
```bash
git clone git clone https://github.com/manikanta7cheruku/Hackathon-Prediction.git
cd Hackathon-Prediction


Create a Virtual Environment (Recommended)
This keeps your computer clean.

For Windows:
Bash -
- py -3.11 -m venv venv
- venv\Scripts\activate

For Mac/Linux:
Bash - 
- python3 -m venv venv
- source venv/bin/activate

3. Install Dependencies
This installs Streamlit, Pandas, XGBoost, and other required libraries.
Bash
- pip install -r requirements.txt
4. Run the App
Bash

- streamlit run app.py