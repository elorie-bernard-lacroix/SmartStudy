import pandas as pd
from sklearn.neighbors import NearestNeighbors

def get_similar_students(data, age, gender, parental_education, desired_grade):
    excluded_columns = ['StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport',
                        'Extracurricular', 'Sports', 'Music', 'Volunteering']
    
# Create the neighborhood DataFrame by dropping specific columns
    neighborhood = data.drop(columns=excluded_columns)

   
    user_query = {
        'Age': age,
        'Gender': gender,
        'ParentalEducation': parental_education,
        'GPA': desired_grade
    }
    user_query = pd.DataFrame(user_query, index=[0])

# apply weights
    weights = {
        'Age': 1.0,
        'Gender': 2.0,
        'ParentalEducation': 1.0,
        'GPA': 100.0
    }

    weighted_neighborhood = neighborhood.copy()
    weighted_user_query = user_query.copy()

    for feature, weight in weights.items():
        weighted_neighborhood[feature] *= weight
        weighted_user_query[feature] *= weight

    knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    knn.fit(weighted_neighborhood)
    _, indices = knn.kneighbors(weighted_user_query)

    # return top 5 similar students from the original data
    return data.iloc[indices.flatten()][[
        'GPA', 'StudyTimeWeekly', 'Absences', 'Extracurricular',
        'Sports', 'Music', 'Volunteering', 'Tutoring'
    ]]
