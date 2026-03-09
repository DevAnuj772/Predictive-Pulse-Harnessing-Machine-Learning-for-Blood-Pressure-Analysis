🩺 Predictive Pulse: Blood Pressure & Hypertension Analysis
Predictive Pulse is an end-to-end Machine Learning application designed to analyze patient demographic and clinical data to predict hypertension stages. This project was developed as part of the Skill Wallet group project to demonstrate full-stack AI/ML integration using Python and Flask.

👥 Developer Team

Anuj Tiwari

Anuj Patel

Archit Raj

Anurag Soni

📊 Project Overview
The system processes 1,825 patient records to classify hypertension into four distinct stages: Normal, Stage-1, Stage-2, and Hypertensive Crisis.

Key Features:
Clinical Data Processing: Handles gender, age groups, medical history, and clinical symptoms like shortness of breath and nosebleeds.

Strategic Model Selection: Utilizes Logistic Regression (95.2% accuracy) over complex models like Random Forest to prevent overfitting and ensure better generalization in clinical settings.

Reliability: Achieved 100% Crisis Recall, ensuring that high-risk hypertensive crisis cases are never missed.

🛠️ Tech Stack & Requirements
To run this project, ensure the following environment is configured:

Platform: Anaconda Navigator & Visual Studio Code

Backend: Python 3.x, Flask

ML Libraries: * numpy & pandas for data manipulation

scikit-learn for model building and scaling

matplotlib & seaborn for visual analysis

pickle-mixin for model serialization

🚀 Workflow
Data Preparation: Handled 477 duplicate records and applied Label Encoding to categorical features.

Feature Engineering: Applied MinMaxScaler to ordinal features for optimal gradient descent in the Logistic Regression model.

Model Serialization: The trained model is saved as logreg_model.pkl for persistent, deployment-ready performance.

Web Integration: A Flask-based interface allows users to enter patient parameters and receive instant clinical interpretations and recommendations.
