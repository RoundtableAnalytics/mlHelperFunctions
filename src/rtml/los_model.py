
from xgboost import XGBRegressor
from rtml.los_pipeline import LosPipeline
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, mean_squared_error
import numpy as np
from pandas import DataFrame
import pickle
import shap

class LosModel:
    '''
    TODO 
    N.B. This is little more than an XGBoost Wrapper right now, will build out hyperparameter optimization as a next step.
    '''
    def __init__(self, cacheDir:str=None, verbose:bool=False):
        if cacheDir[-1] != "/":
            cacheDir = cacheDir + "/"
        self.cacheDir=cacheDir
        self.verbose=verbose
        
    def fullTrain(self, pipe:LosPipeline, outcome:str='LOS', params:dict=None, cache=False, shapThresh:int=80):
        '''
        TODO
        '''
        # Validate Inputs
        if cache:
            if self.cacheDir is None:
                raise Exception("'cache' parameter set to true, but 'cacheDir' (path where files will be written) parameter not set.  Please set this parameter a unique path or set cache to False")
            elif self.verbose:
                print(f"Storing results to {self.cacheDir}")

        # Get Data
        train, test = pipe.getTrainingDatasets(trainIndex=[x for x in range(pipe.cvFolds)], verbose=self.verbose)
        self.features = train.columns[1:].to_list()
        self.outcome=outcome
        if cache:
            if self.verbose:
                print(f"Storing train/test to:\n\t{self.cacheDir}train.csv\n\t{self.cacheDir}test.csv")
            train.to_csv(path_or_buf=self.cacheDir+"train.csv")
            test.to_csv(path_or_buf=self.cacheDir+"test.csv")

        # Create Model
        if not params:
            self.reg = XGBRegressor(n_estimators=700, max_depth=4, eta=.3, gamma=3, colsample_bytree=.6, alpha=.5, objective='reg:squarederror')
        else:
            self.reg = XGBRegressor(**params)
        self.reg.fit(train[self.features], train[self.outcome])
        trainPreds = self.reg.predict(train[self.features])
        testPreds = self.reg.predict(test[self.features])


        # Capture Performance
        if self.verbose:
            print("Explained Variance: ", explained_variance_score(test['LOS'], testPreds))
            print("MSE: ", mean_squared_error(test['LOS'], testPreds))

        # Caculate SHAP
        explainer = shap.TreeExplainer(self.reg)
        shapValues = explainer(test[self.features])
        predsAndFi = np.concatenate((testPreds.reshape([-1, 1]), shapValues.values), axis=1)
        predsAndFi = predsAndFi[predsAndFi[:, 0].argsort()[::-1]]
        topThresh = np.percentile(predsAndFi[:,0], 80)
        topFactors = predsAndFi[predsAndFi[:,0]>topThresh][:,1:]
        fig, ax = plt.subplots(figsize=(8,8))
        ax.boxplot(
            topFactors,
            vert=False,
            notch=True,
            patch_artist=True,
            labels=pipe.featureNames
        )

        # Bias/Fairness
        ## TODO

        # Cache Model
        if cache:
            file = open(self.cacheDir+"model.pkl", 'wb')
            pickle.dump(self.reg, file)
            file.close()

        return trainPreds, testPreds, fig, ax

    def getTrainFromCache(self) -> DataFrame:
        '''
        TODO
        '''


        
