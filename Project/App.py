import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap
import pickle
from io import BytesIO
import base64
import time
from fpdf import FPDF
from datetime import datetime
import os

# Load model
model = pickle.load(open("model.pkl", "rb"))

def save_prediction_data(input_data, prediction_score, category, factors):
    """Save prediction data to Excel file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create prediction data dictionary
    prediction_data = {
        'Timestamp': timestamp,
        'Gender': 'Male' if input_data[0][0] == 1 else 'Female',
        'Age': int(input_data[0][1]),
        'Academic_Pressure': int(input_data[0][2]),
        'CGPA': float(input_data[0][3]),
        'Study_Satisfaction': int(input_data[0][4]),
        'Sleep_Duration': ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'][int(input_data[0][5])],
        'Dietary_Habits': ['Healthy', 'Moderate', 'Unhealthy'][int(input_data[0][6])],
        'Suicidal_Thoughts': 'Yes' if input_data[0][7] == 1 else 'No',
        'Study_Hours': int(input_data[0][8]),
        'Financial_Stress': int(input_data[0][9]),
        'Family_History': 'Yes' if input_data[0][10] == 1 else 'No',
        'Prediction_Score': prediction_score,
        'Risk_Category': category,
        'Top_Factors': str(factors[:3])  # Save top 3 factors
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([prediction_data])
    
    # Save to Excel file
    file_path = 'prediction_history.xlsx'
    if os.path.exists(file_path):
        try:
            # Read existing Excel file
            existing_df = pd.read_excel(file_path, sheet_name='Predictions')
            # Concatenate with new data
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            # Save back to Excel
            updated_df.to_excel(file_path, sheet_name='Predictions', index=False)
        except Exception as e:
            # If there's any error reading the file, create a new one
            df.to_excel(file_path, sheet_name='Predictions', index=False)
    else:
        # If file doesn't exist, create new file
        df.to_excel(file_path, sheet_name='Predictions', index=False)

def save_feedback_data(rating, suggestions):
    """Save feedback data to Excel file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create feedback data dictionary
    feedback_data = {
        'Timestamp': timestamp,
        'Rating': rating,
        'Suggestions': suggestions
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([feedback_data])
    
    # Save to Excel file
    file_path = 'feedback_history.xlsx'
    if os.path.exists(file_path):
        try:
            # Read existing Excel file
            existing_df = pd.read_excel(file_path, sheet_name='Feedback')
            # Concatenate with new data
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            # Save back to Excel
            updated_df.to_excel(file_path, sheet_name='Feedback', index=False)
        except Exception as e:
            # If there's any error reading the file, create a new one
            df.to_excel(file_path, sheet_name='Feedback', index=False)
    else:
        # If file doesn't exist, create new file
        df.to_excel(file_path, sheet_name='Feedback', index=False)

def create_mood_tracking_worksheet():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Mood Tracking Worksheet", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    
    # Add table headers
    pdf.set_font("Arial", "B", 10)
    pdf.cell(30, 10, "Date", 1)
    pdf.cell(30, 10, "Mood (1-5)", 1)
    pdf.cell(30, 10, "Sleep Hours", 1)
    pdf.cell(30, 10, "Stress Level", 1)
    pdf.cell(30, 10, "Notes", 1)
    pdf.ln()
    
    # Add empty rows for tracking
    pdf.set_font("Arial", "", 10)
    for _ in range(7):
        pdf.cell(30, 10, "", 1)
        pdf.cell(30, 10, "", 1)
        pdf.cell(30, 10, "", 1)
        pdf.cell(30, 10, "", 1)
        pdf.cell(30, 10, "", 1)
        pdf.ln()
    
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Mood Scale:", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, "1 - Very Unhappy, 2 - Unhappy, 3 - Neutral, 4 - Happy, 5 - Very Happy", ln=True)
    
    return bytes(pdf.output(dest="S"))

def create_anxiety_management_worksheet():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Anxiety Management Worksheet", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    
    # Anxiety Triggers Section
    pdf.cell(0, 10, "1. Identify Your Anxiety Triggers", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 10, "Common Triggers:", ln=True)
    pdf.set_font("Arial", "", 10)
    triggers = [
        "Academic pressure",
        "Social situations",
        "Financial concerns",
        "Family issues",
        "Health concerns",
        "Other:"
    ]
    for trigger in triggers:
        pdf.cell(0, 10, f"[ ] {trigger}", ln=True)
    
    # Coping Strategies Section
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "2. Coping Strategies", ln=True)
    pdf.set_font("Arial", "", 10)
    strategies = [
        "Deep breathing exercises",
        "Meditation",
        "Physical exercise",
        "Talking to someone",
        "Writing in a journal",
        "Taking a break",
        "Other:"
    ]
    for strategy in strategies:
        pdf.cell(0, 10, f"[ ] {strategy}", ln=True)
    
    return bytes(pdf.output(dest="S"))

def create_self_care_guide():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Self-Care Guide", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    
    # Daily Wellness Practices
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "1. Daily Wellness Practices", ln=True)
    pdf.set_font("Arial", "", 10)
    practices = [
        "- Get 7-8 hours of sleep",
        "- Eat balanced meals",
        "- Stay hydrated",
        "- Exercise regularly",
        "- Take breaks when needed"
    ]
    for practice in practices:
        pdf.cell(0, 10, practice, ln=True)
    
    # Mindfulness Exercises
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "2. Mindfulness Exercises", ln=True)
    pdf.set_font("Arial", "", 10)
    exercises = [
        "- 5-minute breathing meditation",
        "- Body scan relaxation",
        "- Mindful walking",
        "- Gratitude journaling",
        "- Progressive muscle relaxation"
    ]
    for exercise in exercises:
        pdf.cell(0, 10, exercise, ln=True)
    
    # Healthy Lifestyle Tips
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "3. Healthy Lifestyle Tips", ln=True)
    pdf.set_font("Arial", "", 10)
    tips = [
        "- Maintain a regular sleep schedule",
        "- Practice time management",
        "- Set boundaries",
        "- Stay connected with others",
        "- Seek help when needed"
    ]
    for tip in tips:
        pdf.cell(0, 10, tip, ln=True)
    
    return bytes(pdf.output(dest="S"))

# Function to generate downloadable report
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
            <h2>Visual Comparison: You vs Maximum Recommended</h2>
            {visuals_html}
            <h2>Personalized Mental Wellbeing Suggestions</h2>
            <ul>
                {''.join([f'<li>{suggestion}</li>' for suggestion in suggestions])}
            </ul>
            <h2>💚 Thank you for using the Depression Risk Prediction App</h2>
            <p>We hope this tool helps you take a step towards better mental well-being. Remember, seeking support is always a sign of strength. Take care of yourself!</p>
        </body>
    </html>
    """
    return report

# Convert HTML report to base64 string for download
def download_report_button(report_html):
    b64 = base64.b64encode(report_html.encode('utf-8')).decode('utf-8')
    href = f'<a href="data:text/html;base64,{b64}" download="Depression_Risk_Report.html"><button style="background-color:#4CAF50; color:white; padding:10px; border:none; border-radius:5px;">📄 Download Report</button></a>'
    st.markdown(href, unsafe_allow_html=True)

# Prediction logic and UI
def predict():
    st.title("🔮 Predict Your Depression Risk")
    st.markdown("Fill out the form below to predict your risk score and get personalized wellbeing suggestions.")

    with st.form(key="input_form"):
        # Input fields
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.selectbox("Age", list(range(15, 61)))
        academic_pressure = st.selectbox("Academic Pressure (1-5)", [1, 2, 3, 4, 5], help="Higher = More stress")
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
        study_satisfaction = st.selectbox("Study Satisfaction (1-5)", [1, 2, 3, 4, 5])
        sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
        dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
        suicidal_thoughts = st.selectbox("Ever had Suicidal Thoughts?", ["Yes", "No"])
        study_hours = st.selectbox("Study Hours per Day", list(range(0, 13)))
        financial_stress = st.selectbox("Financial Stress (1-5)", [1, 2, 3, 4, 5])
        family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

        submit_button = st.form_submit_button(label="Predict Depression Risk 🚀")

    if submit_button:
        with st.spinner('Predicting... Please wait ⏳'):
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

            prob = model.predict_proba(input_data)[0][1]
            category = "Low" if prob < 0.4 else "Medium" if prob < 0.7 else "High"

            time.sleep(1)

        st.success(f"✅ Depression Risk Score: **{prob:.2f}**")
        st.info(f"📈 Risk Category: **{category}**")

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
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_data[0],
            feature_names=[
                'Gender', 'Age', 'Academic_Pressure', 'CGPA', 'Study_Satisfaction',
                'Sleep_Duration', 'Dietary_Habits', 'Ever_had_suicidal_thoughts',
                'Study_Hours', 'Financial_Stress', 'Family_History_of_Mental_Illness'
            ]),
            max_display=5, show=False)
        st.pyplot(fig)

        # Save prediction data
        save_prediction_data(input_data, prob, category, [f"{col}: {shap_values[0][i]:.2f}" for i, col in enumerate([
            'Gender', 'Age', 'Academic_Pressure', 'CGPA', 'Study_Satisfaction',
            'Sleep_Duration', 'Dietary_Habits', 'Ever_had_suicidal_thoughts',
            'Study_Hours', 'Financial_Stress', 'Family_History_of_Mental_Illness'
        ])])

        # Suggestions
        st.subheader("💡 Personalized Suggestions")
        suggestions = []
        if academic_pressure > 3:
            suggestions.append("💼 Try relaxation techniques like meditation or yoga. Time management coaching could also help reduce stress and improve productivity.")
        if sleep_duration_num == 0:
            suggestions.append("😴 Prioritize getting at least 7-8 hours of sleep every night. Establishing a bedtime routine and limiting screen time can improve sleep quality.")
        if dietary_habits == "Unhealthy":
            suggestions.append("🥗 Consider adopting a balanced and nutritious diet. Include more fruits, vegetables, and proteins. Avoid excessive caffeine and sugar.")
        if suicidal_thoughts_num == 1:
            suggestions.append("🚨 It's crucial to reach out to a mental health professional immediately. There are helplines and counseling services available.")
        if financial_stress > 3:
            suggestions.append("💰 Seek financial counseling or explore scholarships, financial aid, or part-time work options to alleviate some of the stress.")
        if study_satisfaction < 3:
            suggestions.append("📚 Explore different study techniques that match your learning style. You might benefit from active learning strategies or study groups.")
        if study_hours > 8:
            suggestions.append("⏳ Overstudying can lead to burnout. Incorporate regular breaks, stay hydrated, and maintain a balanced schedule.")
        if family_history_num == 1:
            suggestions.append("👨‍👩‍👧‍👦 If there's a family history of mental illness, it's essential to be proactive about your mental health. Regular check-ins with a professional can help.")
        if cgpa < 6.0:
            suggestions.append("🎯 Consider seeking academic counseling or tutoring to help with your studies. Understanding the root cause of the academic struggle can help improve performance.")
        if study_satisfaction == 1:
            suggestions.append("😟 If you're extremely dissatisfied with your studies, it might be time to reassess your course choices. Speak with a mentor to explore options.")
        if gender == "Female" and age < 25:
            suggestions.append("👩‍🎓 As a young woman balancing studies and personal growth, consider mentoring programs or peer support groups that could provide guidance and connection.")
        if financial_stress == 5:
            suggestions.append("💳 High financial stress can be overwhelming. Consider speaking to a financial advisor to make a plan and look for ways to reduce financial burdens.")
        if sleep_duration_num == 3:
            suggestions.append("🌙 Sleep quality is essential. Although you're getting more than 8 hours, make sure the sleep is restful. Limit screen exposure an hour before bedtime.")
        if suicidal_thoughts_num == 0 and sleep_duration_num < 2:
            suggestions.append("🧠 While you don't experience suicidal thoughts, your sleep patterns suggest a need for attention. Improving your sleep might enhance your overall mental wellbeing.")
        if study_satisfaction > 4:
            suggestions.append("🌟 You're likely on the right track! Keep maintaining a balanced approach to your studies and ensure you're also taking care of your mental health.")
        if dietary_habits == "Moderate" and sleep_duration_num == 1:
            suggestions.append("🍽️ Your diet and sleep habits could be better. Aim for more balanced meals and aim to improve your sleep duration for better overall health.")
        if study_hours == 0:
            suggestions.append("🛑 It seems you might not be studying enough. Find a quiet and comfortable study environment to motivate yourself and increase productivity.")
        if academic_pressure == 5:
            suggestions.append("🔥 High academic pressure can be draining. Consider seeking support from a counselor or joining study groups to manage stress.")
        if sleep_duration_num == 2:
            suggestions.append("🛏️ Try incorporating more relaxation techniques into your routine, such as deep breathing exercises, to improve sleep quality.")
        if dietary_habits == "Healthy":
            suggestions.append("🥗 Great job maintaining healthy eating habits! Keep up the good work, and continue nourishing your body and mind.")
        if suicidal_thoughts_num == 0 and study_satisfaction > 3:
            suggestions.append("🌈 You're managing well. Keep maintaining a healthy work-life balance and stay connected with supportive people.")
        if financial_stress == 1:
            suggestions.append("💸 Financial stress is low, but it's still important to stay mindful of budgeting. Consider keeping track of your expenses to ensure long-term stability.")
        if family_history_num == 0:
            suggestions.append("👍 Having no family history of mental illness is a positive sign. Continue focusing on your mental health through self-care and professional check-ins.")
        if study_hours == 12:
            suggestions.append("⏰ Long study hours might lead to burnout. Take regular breaks and try using techniques like the Pomodoro method to balance studying and rest.")
        if academic_pressure == 1:
            suggestions.append("🌱 Low academic pressure is great! Use this time to explore hobbies, relax, and focus on your overall wellbeing.")
        if study_satisfaction == 2:
            suggestions.append("😓 Low study satisfaction indicates you may need a change. Try discussing your concerns with a mentor or counselor for guidance.")
        if cgpa > 8.5:
            suggestions.append("🎉 You're doing excellent academically! Keep up the hard work while ensuring you prioritize your mental and physical health.")
        if sleep_duration_num == 1 and study_hours < 4:
            suggestions.append("🕓 It seems you're not getting enough study time and sleep. Balance your schedule to improve both.")
        if family_history_num == 0 and suicidal_thoughts_num == 1:
            suggestions.append("⚠️ Even without a family history, suicidal thoughts are a serious matter. Seek help immediately from a mental health professional.")

        for s in suggestions:
            st.markdown(f"- {s}")

        # Input vs Recommended Visuals
        st.subheader("📊 Visual Comparison")
        visuals_html = ""
        input_vs_recommended = {
            "Sleep Duration (0–3)": (sleep_duration_num, 2),
            "Study Hours": (study_hours, 5),
        }

        for feature, (user_val, rec_val) in input_vs_recommended.items():
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.barplot(x=["You", "Maximum Recommended"], y=[user_val, rec_val], palette="pastel", ax=ax)
            ax.set_title(feature,fontsize=10)
            img = BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            img_base64 = base64.b64encode(img.read()).decode()
            visuals_html += f"<h4>{feature}</h4><img src='data:image/png;base64,{img_base64}'/><br>"

            # Display the plot using Streamlit with a specified width (e.g., 500px)
            st.image(f"data:image/png;base64,{img_base64}", width=500) 
            plt.close(fig)

        # Report
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
        st.divider()
        st.subheader("⬇️ Download Your Report")
        download_report_button(report_html)
        st.divider()
        st.markdown("<h3 style='text-align: center; color: #4CAF50;'>💚 Thank you for using the Depression Risk Prediction App. We hope this tool helps you take a step towards better mental well-being. Remember, seeking support is always a sign of strength. 💪</h3>", unsafe_allow_html=True)

def feedback_form():
    st.title("📝 We'd Love Your Feedback!")
    
    # Rating
    rating = st.slider("Rate Your Experience (1 - Poor to 5 - Excellent)", 1, 5, 3)
    
    # Suggestions
    suggestions = st.text_area("Any suggestions to improve the app?")
    
    # Submit Button
    submit_button = st.button("Submit Feedback")
    
    if submit_button:
        # Save feedback data
        save_feedback_data(rating, suggestions)
        
        st.success("Thank you for your feedback!")
        
        # Thank you message
        st.write(f"Your rating: {rating}")
        st.write(f"Your suggestions: {suggestions if suggestions else 'No suggestions provided.'}")

def interactive_dashboard():
    st.title("📊 Interactive Analytics Dashboard")
    
    # Load the dataset for analysis
    try:
        df = pd.read_csv("student_depression_dataset.csv")
        
        # Get actual column names from the dataset
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if not numeric_columns:
            st.error("No numeric columns found in the dataset. Please check the dataset structure.")
            return
            
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📈 Risk Factor Trends", "🔄 Correlation Analysis", "📊 Population Comparison"])
        
        with tab1:
            st.subheader("Risk Factor Distribution")
            
            # Select factor to analyze
            factor = st.selectbox(
                "Select Factor to Analyze",
                numeric_columns
            )
            
            # Create distribution plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x=factor, bins=20, kde=True)
            plt.title(f"Distribution of {factor}")
            st.pyplot(fig)
            
            # Show statistics
            st.write("### Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{df[factor].mean():.2f}")
            with col2:
                st.metric("Median", f"{df[factor].median():.2f}")
            with col3:
                st.metric("Standard Deviation", f"{df[factor].std():.2f}")
        
        with tab2:
            st.subheader("Correlation Analysis")
            
            # Select factors for correlation
            factors = st.multiselect(
                "Select Factors to Compare",
                numeric_columns,
                default=numeric_columns[:2] if len(numeric_columns) >= 2 else numeric_columns
            )
            
            if len(factors) >= 2:
                # Create correlation heatmap
                correlation = df[factors].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
                plt.title("Correlation Heatmap")
                st.pyplot(fig)
                # Create scatter plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df, x=factors[0], y=factors[1])
                plt.title(f"{factors[0]} vs {factors[1]}")
                st.pyplot(fig)
        
        with tab3:
            st.subheader("Population Comparison")
            
            # Select metric for comparison
            metric = st.selectbox(
                "Select Metric",
                numeric_columns
            )
            
            # Create box plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df, y=metric)
            plt.title(f"Population Distribution of {metric}")
            st.pyplot(fig)
            
            # Show percentile information
            st.write("### Percentile Information")
            percentiles = [25, 50, 75, 90]
            values = [df[metric].quantile(p/100) for p in percentiles]
            
            for p, v in zip(percentiles, values):
                st.write(f"{p}th Percentile: {v:.2f}")
    
    except FileNotFoundError:
        st.error("Dataset file 'student_depression_dataset.csv' not found. Please ensure the file is in the correct location.")
    except Exception as e:
        st.error("An error occurred while loading or processing the dataset.")
        st.write("Error details:", str(e))
        st.write("Please check if the dataset is properly formatted and contains numeric columns.")

def resource_center():
    st.title("📚 Mental Health Resource Center")
    
    # Create tabs for different resource categories
    tab1, tab2, tab3 = st.tabs(["📖 Educational Content", "🆘 Professional Help", "📥 Downloadable Resources"])
    
    with tab1:
        st.header("Educational Content")
        
        # Understanding Depression
        with st.expander("Understanding Depression", expanded=True):
            st.markdown("""
            ### What is Depression?
            Depression is a common and serious medical illness that negatively affects how you feel, the way you think and how you act. 
            It can lead to a variety of emotional and physical problems and can decrease your ability to function at work and at home.
            
            #### Common Symptoms:
            - Persistent sad, anxious, or "empty" mood
            - Feelings of hopelessness or pessimism
            - Irritability
            - Feelings of guilt, worthlessness, or helplessness
            - Loss of interest or pleasure in hobbies and activities
            - Decreased energy or fatigue
            - Moving or talking more slowly
            - Difficulty sleeping, early-morning awakening, or oversleeping
            """)
        
        # Coping Strategies
        with st.expander("Coping Strategies", expanded=True):
            st.markdown("""
            ### Effective Coping Strategies
            
            #### 1. Lifestyle Changes
            - Maintain a regular sleep schedule
            - Exercise regularly
            - Eat a balanced diet
            - Practice mindfulness or meditation
            
            #### 2. Social Support
            - Stay connected with friends and family
            - Join support groups
            - Seek professional help when needed
            
            #### 3. Academic Balance
            - Set realistic goals
            - Break tasks into smaller steps
            - Take regular breaks
            - Practice time management
            """)
        
        # Stress Management
        with st.expander("Stress Management", expanded=True):
            st.markdown("""
            ### Managing Academic Stress
            
            #### 1. Time Management
            - Create a study schedule
            - Set priorities
            - Avoid procrastination
            
            #### 2. Study Techniques
            - Use active learning methods
            - Take regular breaks
            - Create a conducive study environment
            
            #### 3. Self-Care
            - Practice relaxation techniques
            - Maintain work-life balance
            - Get adequate sleep
            """)
    
    with tab2:
        st.header("Professional Help Resources")
        
        # Emergency Contacts
        st.subheader("🚨 Emergency Contacts")
        st.markdown("""
        ### Immediate Help (24/7)
        - **National Suicide Prevention Lifeline**: 988
        - **Crisis Text Line**: Text HOME to 741741
        - **Emergency Services**: 911
        
        ### Mental Health Helplines
        - **SAMHSA's National Helpline**: 1-800-662-4357
        - **National Alliance on Mental Illness (NAMI)**: 1-800-950-6264
        """)
        
        # Professional Resources
        st.subheader("👨‍⚕️ Professional Resources")
        st.markdown("""
        ### Finding Professional Help
        1. **University Counseling Services**
           - Contact your university's counseling center
           - Many offer free or low-cost services
        
        2. **Online Therapy Platforms**
           - BetterHelp
           - Talkspace
           - 7 Cups
        
        3. **Local Mental Health Clinics**
           - Search for certified mental health professionals in your area
           - Check with your insurance provider for covered services
        """)
        
        # Support Groups
        st.subheader("👥 Support Groups")
        st.markdown("""
        ### Available Support Groups
        1. **NAMI Support Groups**
           - Free support groups for individuals and families
           - Available both online and in-person
        
        2. **Depression and Bipolar Support Alliance (DBSA)**
           - Peer-led support groups
           - Online and in-person meetings available
        
        3. **University Support Groups**
           - Check with your university's counseling center
           - Often free for students
        """)
    
    with tab3:
        st.header("Downloadable Resources")
        
        # Create columns for different resource types
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Worksheets")
            st.markdown("""
            ### Self-Help Worksheets
            1. **Mood Tracking Worksheet**
               - Track daily moods and triggers
               - Identify patterns and coping strategies
            
            2. **Anxiety Management Worksheet**
               - Identify anxiety triggers
               - Practice coping techniques
            
            3. **Goal Setting Worksheet**
               - Set realistic academic goals
               - Create action plans
            """)
            
            # Add download buttons for worksheets with actual PDF content
            mood_tracking_pdf = create_mood_tracking_worksheet()
            st.download_button(
                label="📥 Download Mood Tracking Worksheet",
                data=mood_tracking_pdf,
                file_name="mood_tracking_worksheet.pdf",
                mime="application/pdf"
            )
            
            anxiety_worksheet_pdf = create_anxiety_management_worksheet()
            st.download_button(
                label="📥 Download Anxiety Management Worksheet",
                data=anxiety_worksheet_pdf,
                file_name="anxiety_management_worksheet.pdf",
                mime="application/pdf"
            )
        
        with col2:
            st.subheader("📚 Guides")
            st.markdown("""
            ### Educational Guides
            1. **Understanding Depression Guide**
               - Symptoms and causes
               - Treatment options
               - Self-help strategies
            
            2. **Academic Stress Management Guide**
               - Time management tips
               - Study techniques
               - Balance strategies
            
            3. **Self-Care Guide**
               - Daily wellness practices
               - Mindfulness exercises
               - Healthy lifestyle tips
            """)
            
            # Add download button for self-care guide with actual PDF content
            self_care_pdf = create_self_care_guide()
            st.download_button(
                label="📥 Download Self-Care Guide",
                data=self_care_pdf,
                file_name="self_care_guide.pdf",
                mime="application/pdf"
            )

# Main App
def main():
    st.set_page_config(page_title="Depression Risk Prediction", page_icon="🧠", layout="wide")
    st.sidebar.title("🧠 Depression Risk App")
    st.sidebar.image("c:\\Users\\pavan\\OneDrive\\Documents\\Project\\Project\\Img1.jpg", caption="Mental Health Awareness", use_container_width=True)

    app_mode = st.sidebar.radio("Navigation", ["🏠 Home", "ℹ️ About", "🔮 Prediction", "📊 Analytics", "📚 Resources", "📋 Feedback"])
    if app_mode == "🏠 Home":
        st.markdown(
        "<h1 style='text-align: center;'>💡Welcome to the Depression Risk Prediction App</h1>",
        unsafe_allow_html=True,
    )
        col1, col2 = st.columns([2, 2])  # Adjust the ratio to control column width

# Left column for the image
        with col2:
            st.image("c:\\Users\\pavan\\OneDrive\\Documents\\Project\\Project\\Img2.jpg", width=500)

# Right column for the text
        with col1:
            st.write("💬Turning silence into strength—where healing begins and hope thrives. Let's create spaces free from depression, where every mind can breathe, grow, and shine.🌱✨")
            st.markdown("### Features:")
            st.write("1. **Accurate Depression Risk Prediction**:  Predict the risk of depression based on various factors such as academic pressure, sleep patterns, and mental health history.")
            st.write("2. **Personalized Recommendations**:  Get tailored suggestions to improve mental health based on individual risk factors.")
            st.write("3. **Interactive Visualization**:  Visualize key factors contributing to your depression risk using advanced data science techniques.")
            st.write("4. **User-Friendly Interface**:  Easy-to-use design with clear input options for a seamless user experience.")
            st.write("5. **Comprehensive Data Insights**:  Access detailed insights into your mental health based on the analysis of your inputs.")


    elif app_mode == "🔮 Prediction":
        predict()
    
    elif app_mode == "📊 Analytics":
        interactive_dashboard()
    
    elif app_mode == "📚 Resources":
        resource_center()
    
    elif app_mode == "📋 Feedback":
        feedback_form()

    elif app_mode == "ℹ️ About":
        st.header("About This Application")
        st.write("""This application uses machine learning techniques to predict the likelihood of depression based on various 
                    physiological and psychological factors, providing users with valuable insights into their mental health. 
                    The goal is to offer a tool that helps individuals identify early signs of depression, promoting better 
                    mental health awareness and early intervention. By providing personalized recommendations, this model aims 
                    to improve well-being and empower users to take proactive steps toward better mental health.""")
        
        st.markdown("### Technologies Used:")
        st.write("- **XGBoost**: To build an optimized machine learning model for predicting depression risk with high accuracy.")
        st.write("- **Streamlit**: For creating an interactive and user-friendly web application that allows users to easily input their data and view results.")
        st.write("- **SHAP**: For generating model explainability, helping users understand the key factors influencing their depression risk score.")
        
        st.markdown("### Challenges Addressed:")
        st.write("- **Early Intervention**: Helping individuals identify early signs of depression, enabling timely support and treatment.")
        st.write("- **Personalized Recommendations**: Offering tailored suggestions based on each user's unique input to promote mental well-being.")
        st.write("- **Data Interpretation**: Providing transparency and understanding of how different factors contribute to the depression risk score using SHAP values.")
        
        
        st.markdown("### Acknowledgments:")
        st.write("""This project is inspired by the growing need for accessible mental health tools and the desire to promote better 
                    awareness of depression. Thanks to open-source contributions, research papers, and datasets that made this application possible. 
                    Special thanks to the mental health community for their dedication to improving lives and breaking the stigma around mental health issues.""")
        
        st.subheader("📞 Contact Us")
        st.write("""
    If you have any questions, concerns, or suggestions, feel free to reach out to us:
    - **Email**: metalhealthsupport@example.com
    - **Phone**: +91 2345678901
    - **Website**: [www.metalhealthsupport.com](http://www.example.com)
    """)
    
        
 
if __name__ == "__main__":
    main()
