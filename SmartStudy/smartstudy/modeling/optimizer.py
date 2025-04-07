import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import joblib
from sklearn.preprocessing import StandardScaler

def get_search_space():
    return [
        Real(0.0, 20.0, name='StudyTimeWeekly'),
        Integer(0, 29, name='Absences'),
        Categorical([0, 1], name='Tutoring'),
        Integer(0, 4, name='ParentalSupport'),
        Categorical([0, 1], name='Extracurricular'),
        Categorical([0, 1], name='Sports'),
        Categorical([0, 1], name='Music'),
        Categorical([0, 1], name='Volunteering')
    ]


def optimize_study_habits(fixed_user_data, reg, scaler, desired_grade):
    
    # define search space for optimization
    space = get_search_space()

    @use_named_args(space)
    def objective(**params):
        user_data = fixed_user_data.copy()
        user_data.update(params)
        df = pd.DataFrame(user_data, index=[0])
        input_vec = scaler.transform(df)
        pred = reg.predict([input_vec[0]])[0]
        return abs(desired_grade - pred)

    result = gp_minimize(objective, space, n_calls=50, random_state=0)
    optimized = dict(zip([dim.name for dim in space], result.x))
    fixed_user_data.update(optimized)
    return fixed_user_data, result.fun
