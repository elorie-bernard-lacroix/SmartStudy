import gradio as gr
import pandas as pd
import joblib

from modeling.optimizer import optimize_study_habits
from modeling.knn_matching import get_similar_students
from modeling.gpt_utils import generate_recommendations_gpt4
from config import MODELS_DIR, PROCESSED_DATA_DIR

reg = joblib.load(MODELS_DIR / "tabpfn.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")

def demo_app(age, gender, parental_education, study_time, absences, tutoring, parental_support,
             extracurricular, sports, music, volunteering, target_gpa):

    current_habits = {
        'Age': age,
        'Gender': gender,
        'ParentalEducation': parental_education,
        'StudyTimeWeekly': study_time,
        'Absences': absences,
        'Tutoring': tutoring,
        'ParentalSupport': parental_support,
        'Extracurricular': extracurricular,
        'Sports': sports,
        'Music': music,
        'Volunteering': volunteering
    }

    user_fixed = {
        'Age': age,
        'Gender': gender,
        'ParentalEducation': parental_education
    }

    # optimize study habits
    optimized_values, _ = optimize_study_habits(user_fixed, reg, scaler, target_gpa)

    optimized_habits = {
        'StudyTimeWeekly': optimized_values['StudyTimeWeekly'],
        'Absences': optimized_values['Absences'],
        'Tutoring': optimized_values['Tutoring'],
        'ParentalSupport': optimized_values['ParentalSupport'],
        'Extracurricular': optimized_values['Extracurricular'],
        'Sports': optimized_values['Sports'],
        'Music': optimized_values['Music'],
        'Volunteering': optimized_values['Volunteering']
    }

    summary = generate_recommendations_gpt4(current_habits, {**user_fixed, **optimized_habits}, target_gpa)

    # Find similar students
    data = pd.read_csv(PROCESSED_DATA_DIR / "processed_data.csv")
    similar_students = get_similar_students(data, age, gender, parental_education, target_gpa)

    example_table = similar_students[[
        'GPA', 'StudyTimeWeekly', 'Absences', 'Extracurricular', 'Sports', 'Music', 'Volunteering', 'Tutoring'
    ]]

    result_table = pd.DataFrame([optimized_habits])
    return result_table, example_table, summary

#  the gradio interface
def app_ui():
    with gr.Blocks() as app:
        gr.Markdown("### \U0001F9E0 Fill in your current habits and target GPA:")

        with gr.Row():
            age = gr.Number(label="\U0001F9D2 Age", value=15)
            gender = gr.Radio([0, 1], label="♂️ Gender (0=Male, 1=Female)", value=0)
            parental_education = gr.Dropdown(
                choices=[0, 1, 2, 3, 4],
                label="🎓 Parental Education (0=None, 1=High School, 2=College, 3=Bachelor's, 4=Higher)",
                value=1
            )

        with gr.Row():
            study_time = gr.Number(label="📚 Study Time Weekly (hrs)", value=0.0)
            absences = gr.Number(label="🚫 Absences (0–30)", value=10)
            tutoring = gr.Radio([0, 1], label="🎓 Tutoring (0=No, 1=Yes)", value=0)
            parental_support = gr.Slider(0, 4, step=1,
                label="👨‍👩‍👧 Parental Support (0=None to 4=Very High)", value=0)

        with gr.Row():
            extracurricular = gr.Radio([0, 1], label="🎭 Extracurricular", value=0)
            sports = gr.Radio([0, 1], label="🏀 Sports", value=0)
            music = gr.Radio([0, 1], label="🎵 Music", value=0)
            volunteering = gr.Radio([0, 1], label="💪 Volunteering", value=0)

        target_gpa = gr.Number(label="🎯 Target GPA", value=0.0)

        with gr.Row():
            submit = gr.Button("🚀 Get Personalized Plan")

        output1 = gr.Dataframe(label="Optimized Study Habits")
        output2 = gr.Dataframe(label="Similar Students (KNN)")
        output3 = gr.Textbox(label="GPT Summary", lines=8)

        submit.click(fn=demo_app,
                     inputs=[age, gender, parental_education,
                             study_time, absences, tutoring, parental_support,
                             extracurricular, sports, music, volunteering, target_gpa],
                     outputs=[output1, output2, output3])
    return app

# landing Page
with gr.Blocks() as landing:
    gr.Markdown("""
    <center>
    <h1>🎓 <span style='color:#4A90E2'>Your Personalized GPA Booster:</span> SmartStudy</h1>
    <h3>By <i>Study Architects</i></h3>
    <p style="max-width: 700px; font-size: 17px;">
    A smart tool that helps students reach their academic goals by recommending better study habits,
    backed by real student data and explained using GPT-4.
    </p>
    </center>
    """)

    start_btn = gr.Button("Get Started")
    app_container = gr.Column(visible=False)

    start_btn.click(fn=lambda: gr.update(visible=True), outputs=app_container)

    with app_container:
        app_ui()

landing.launch()