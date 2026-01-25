# Hackathon Success Predictor AI

**A Predictive Analytics Dashboard to forecast Hackathon team performance using Multi-Factor Dynamics Analysis.**

## Project Overview

This project utilizes Machine Learning (**XGBoost**) to analyze behavioral telemetry data—such as **Commit Velocity**, **Communication Frequency**, and **Sleep Schedules**—to predict whether a team will win a Hackathon.

Unlike traditional models that look only at technical skills, this system evaluates the **"Human Dynamics"** that drive success. The solution processes these inputs to provide real-time strategic insights via an interactive **Streamlit** web dashboard.

## Key Capabilities

*   **Real-time Prediction Engine:** Instantly calculates Win Probability based on specific team behavioral metrics.
*   **6-Tier Classification System:** Categorizes teams into distinct performance brackets, ranging from "Critical Failure" to "Elite / Dominating."
*   **Explainable AI (XAI):** Integrated **SHAP (SHapley Additive exPlanations)** values to visualize exactly *why* a prediction was made (e.g., scoring impacted negatively by Sleep Deprivation).
*   **Strategic AI Assessment:** Provides dynamic, actionable feedback to improve team odds (e.g., "Increase Communication to >50 msgs/hr").
*   **Bottleneck Detection:** Automatically identifies the primary constraint limiting team performance (Sleep vs. Velocity vs. Communication).

## Technology Stack

*   **Frontend Interface:** Streamlit
*   **Machine Learning:** XGBoost Classifier, Scikit-Learn
*   **Model Interpretability:** SHAP
*   **Data Processing:** Pandas, NumPy
*   **Model Persistence:** Joblib

## Installation & Setup

Follow these steps to set up the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/manikanta7cheruku/Hackathon-Prediction.git
cd Hackathon-Prediction
2. Create a Virtual Environment
It is recommended to run this project in a virtual environment to manage dependencies.

For Windows:

Bash

py -3.11 -m venv venv
.\venv\Scripts\activate
For Mac/Linux:

Bash

python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Install Streamlit, Pandas, XGBoost, and other required libraries.

Bash

pip install -r requirements.txt
4. Run the Application
Launch the dashboard on your local server.

Bash

streamlit run app.py
Usage
Once the app is running, open the local URL provided in the terminal (usually http://localhost:8501).
Input the team parameters (Commit Velocity, Sleep Hours, Communication Frequency).
Click Predict.
Review the Win Probability, Team Classification, and AI-generated strategic advice.
License
Distributed under the MIT License.