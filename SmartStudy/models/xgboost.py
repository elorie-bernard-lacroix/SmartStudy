import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV

def xgboost_train(data, 
                  labels, 
                  best_params={'colsample_bytree': 0.8,
                                'gamma': 0,
                                'learning_rate': 0.01,
                                'max_depth': 3,
                                'n_estimators': 300,
                                'subsample': 1.0} ):
    
    '''
    Inputs:
    -- data 
    -- labels
    -- best_params (optional): found using xgboost_tuning(), otherwise specified for you
    Returns: best_params
    '''
    xgb_model = xgb.XGBClassifier(objective='multi:softmax',
                                    num_class=len(np.unique(labels)),
                                    random_state=42,
                                    **best_params)
    xgb_model.fit(data, labels)

    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(np.unique(labels)), random_state=42)
    xgb_model.fit(data, labels)

    return xgb_model

def xgboost_tuning(xgb_model, 
                   data, 
                   labels, 
                   param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 4, 5],
                        'learning_rate': [0.1, 0.01, 0.001],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0],
                        'gamma': [0, 0.1, 0.2]}):
    '''
    Inputs:
    -- xgb_model
    -- data
    -- labels
    -- param_grid (optional)
    Returns: best_params
    -- 
    '''
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(data, labels)

    return grid_search.best_params_