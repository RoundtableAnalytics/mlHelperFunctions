from xgboost import XGBRegressor
from rtml.los_pipeline import LosPipeline
from sklearn.metrics import explained_variance_score, mean_squared_error

class LosModel:
    '''
    TODO 
    N.B. This is little more than an XGBoost Wrapper right now, will build out hyperparameter optimization as a next step.
    '''
    def __init__(self):
        pass

    def fullTrain(self, pipe:LosPipeline, outcome:str='LOS', params:dict=None, verbose:bool=False):
        train, test = pipe.getTrainingDatasets(trainIndex=[x for x in range(pipe.cvFolds)], verbose=verbose)
        features = train.columns[1:].to_list()
        if not params:
            self.reg = XGBRegressor(n_estimators=700, max_depth=4, eta=.3, gamma=3, colsample_bytree=.6, alpha=.5, objective='reg:tweedie')
        else:
            self.reg = XGBRegressor(**params)
        self.reg.fit(train[features], train[outcome])
        test_preds = self.reg.predict(test[features])
        print("Explained Variance: ", explained_variance_score(test['LOS'], test_preds))
        print("MSE: ", mean_squared_error(test['LOS'], test_preds))
        
