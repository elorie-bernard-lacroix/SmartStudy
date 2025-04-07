import openai

openai.api_key = "your key"#replace with actual key

def generate_recommendations_gpt4(current_habits, optimized_habits, desired_grade):
    prompt = f"""
You are an academic coach. A student wants to raise their GPA to {desired_grade}.

Here are their current habits:
Study time: {current_habits['StudyTimeWeekly']} hrs/week
Absences: {current_habits['Absences']}
Tutoring: {current_habits['Tutoring']}
Parental Support: {current_habits['ParentalSupport']}
Extracurriculars: Sports={current_habits['Sports']}, Music={current_habits['Music']}, Volunteering={current_habits['Volunteering']}

And here’s what they’re willing to change:
Study time: {optimized_habits['StudyTimeWeekly']} hrs/week
Absences: {optimized_habits['Absences']}
Tutoring: {optimized_habits['Tutoring']}
Parental Support: {optimized_habits['ParentalSupport']}
Extracurriculars: Sports={optimized_habits['Sports']}, Music={optimized_habits['Music']}, Volunteering={optimized_habits['Volunteering']}

Please explain how these changes will help the student reach their goal. Use clear, encouraging language. Keep it under 200 words and write as bullet points.
"""
    response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600
        )

    response_text = response.choices[0].message.content

    return response_text
