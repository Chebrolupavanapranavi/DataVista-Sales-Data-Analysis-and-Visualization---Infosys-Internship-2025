import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap
import pickle
from io import BytesIO
import base64

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Function to generate a downloadable report
def generate_report(prediction_data, visuals_html, suggestions):
    report = f"""
    <html>
        <body style="font-family:Arial; padding:20px;">
            <h1 style="color:#1f77b4;">Depression Risk Prediction Report</h1>
            <h2>Prediction Results</h2>
            <p><b>Depression Risk Score:</b> {prediction_data['score']:.2f}</p>
            <p><b>Risk Category:</b> {prediction_data['category']}</p>
            <h2>Top Contributing Factors</h2>
            <ul>
                {''.join([f'<li>{factor}</li>' for factor in prediction_data['factors']])}
            </ul>
            <h2>Visual Comparison: You vs Recommended</h2>
            {visuals_html}
            <h2>Personalized Mental Wellbeing Suggestions</h2>
            <ul>
                {''.join([f'<li>{suggestion}</li>' for suggestion in suggestions])}
            </ul>
        </body>
    </html>
    """
    return report

# Convert HTML report to base64 string for download
def download_report(report_html):
    b64 = base64.b64encode(report_html.encode('utf-8')).decode('utf-8')
    href = f'<a href="data:text/html;base64,{b64}" download="Depression_Risk_Report.html">📄 Download Report</a>'
    st.markdown(href, unsafe_allow_html=True)

# Prediction logic and UI
def predict():
    st.title("🔮 Predict Depression Risk")
    st.write("Fill out the form below to predict your depression risk and receive personalized recommendations.")

    # Input fields
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.selectbox("Age", list(range(15, 61)))
    academic_pressure = st.selectbox("Academic Pressure (1-5)", [1, 2, 3, 4, 5])
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
    study_satisfaction = st.selectbox("Study Satisfaction (1-5)", [1, 2, 3, 4, 5])
    sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
    dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
    suicidal_thoughts = st.selectbox("Ever had Suicidal Thoughts?", ["Yes", "No"])
    study_hours = st.selectbox("Study Hours per Day", list(range(0, 13)))
    financial_stress = st.selectbox("Financial Stress (1-5)", [1, 2, 3, 4, 5])
    family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

    # Convert categorical inputs
    gender_num = 1 if gender == "Male" else 0
    sleep_map = {"Less than 5 hours": 0, "5-6 hours": 1, "7-8 hours": 2, "More than 8 hours": 3}
    sleep_duration_num = sleep_map[sleep_duration]
    diet_map = {"Healthy": 0, "Moderate": 1, "Unhealthy": 2}
    dietary_habits_num = diet_map[dietary_habits]
    suicidal_thoughts_num = 1 if suicidal_thoughts == "Yes" else 0
    family_history_num = 1 if family_history == "Yes" else 0

    input_data = np.array([[gender_num, age, academic_pressure, cgpa, study_satisfaction,
                            sleep_duration_num, dietary_habits_num, suicidal_thoughts_num,
                            study_hours, financial_stress, family_history_num]])

    if st.button("Predict Depression Risk"):
        # Prediction
        prob = model.predict_proba(input_data)[0][1]
        category = "Low" if prob < 0.4 else "Medium" if prob < 0.7 else "High"

        st.subheader("🎯 Depression Risk Results")
        st.write(f"**Depression Risk Score:** {prob:.2f}")
        st.write(f"**Risk Category:** {category}")

        # SHAP Explanation
        st.subheader("🔍 Top Contributing Factors")
        shap.initjs()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(pd.DataFrame(input_data, columns=[
            'Gender', 'Age', 'Academic_Pressure', 'CGPA', 'Study_Satisfaction',
            'Sleep_Duration', 'Dietary_Habits', 'Ever_had_suicidal_thoughts',
            'Study_Hours', 'Financial_Stress', 'Family_History_of_Mental_Illness'
        ]))

        fig, ax = plt.subplots()
        shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                             base_values=explainer.expected_value,
                                             data=input_data[0],
                                             feature_names=[
                                                'Gender', 'Age', 'Academic_Pressure', 'CGPA', 'Study_Satisfaction',
                                                'Sleep_Duration', 'Dietary_Habits', 'Ever_had_suicidal_thoughts',
                                                'Study_Hours', 'Financial_Stress', 'Family_History_of_Mental_Illness'
                                             ]),
                            max_display=5, show=False)
        st.pyplot(fig)

        # Suggestions
        suggestions = []
        if academic_pressure > 3:
            suggestions.append("Try relaxation techniques and consider time management coaching.")
        if sleep_duration_num == 0:
            suggestions.append("Aim for 7-8 hours of sleep daily.")
        if dietary_habits == "Unhealthy":
            suggestions.append("Adopt a balanced and nutritious diet.")
        if suicidal_thoughts_num == 1:
            suggestions.append("Please reach out to a mental health professional immediately.")
        if financial_stress > 3:
            suggestions.append("Seek financial counseling or explore scholarship opportunities.")
        if study_satisfaction < 3:
            suggestions.append("Consider reflecting on your study methods or seek academic counseling.")
        if study_hours > 8:
            suggestions.append("Avoid overstudying; balance your day with rest and social activities.")

        st.subheader("💡 Personalized Mental Wellbeing Suggestions")
        for s in suggestions:
            st.markdown(f"- {s}")

        # Input vs Recommended Visuals
        st.subheader("📊 Visual Comparison: You vs Recommended Values")
        visuals_html = ""

        input_vs_recommended = {
            "Academic Pressure": (academic_pressure, 2),
            "Study Satisfaction": (study_satisfaction, 4),
            "Sleep Duration (0–3)": (sleep_duration_num, 2),
            "Study Hours": (study_hours, 5),
            "Financial Stress": (financial_stress, 2)
        }

        for feature, (user_val, rec_val) in input_vs_recommended.items():
            fig, ax = plt.subplots()
            sns.barplot(x=["You", "Recommended"], y=[user_val, rec_val], palette="pastel", ax=ax)
            ax.set_title(feature)
            img = BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            visuals_html += f"<h4>{feature}</h4><img src='data:image/png;base64,{base64.b64encode(img.read()).decode()}'/><br>"
            st.pyplot(fig)
            plt.close(fig)

        # Prepare report
        prediction_data = {
            'score': prob,
            'category': category,
            'factors': [f"{col}: {shap_values[0][i]:.2f}" for i, col in enumerate([
                'Gender', 'Age', 'Academic_Pressure', 'CGPA', 'Study_Satisfaction',
                'Sleep_Duration', 'Dietary_Habits', 'Ever_had_suicidal_thoughts',
                'Study_Hours', 'Financial_Stress', 'Family_History_of_Mental_Illness'
            ])]
        }

        report_html = generate_report(prediction_data, visuals_html, suggestions)
        download_report(report_html)

# Main App
def main():
    st.sidebar.title("🧠 Depression Risk App")
    app_mode = st.sidebar.selectbox("Choose a Section", ["Home", "Prediction", "About"])

    if app_mode == "Home":
        st.title("Welcome to the Depression Risk Prediction App")
        st.markdown("""
            This application uses a machine learning model to assess your risk of depression and provides suggestions based on your inputs.
        """)
    elif app_mode == "Prediction":
        predict()
    elif app_mode == "About":
        st.title("About This App")
        st.markdown("""
        - 📌 Built with: Streamlit, SHAP, XGBoost  
        - 🎯 Purpose: Provide mental health awareness and support  
        - 👨‍🔬 Author: Anitha, Vyshnavi , Pranavi  
        - 🔒 Disclaimer: This app is for educational purposes and not a substitute for professional medical advice.
        """)

if __name__ == "__main__":
    main()
