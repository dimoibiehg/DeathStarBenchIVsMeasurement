from Learning.LearningApproach.SKLearn_Approach import SKLearnApproach
from sklearn.linear_model import LinearRegression
from Learning.LearningApproach.LearningApproachEnums import LearningApproachMethodName
import pandas as pd
from tqdm import tqdm
import numpy as np

class LRApproach(SKLearnApproach):
    """
    is_hierarchical: determine the learner follows hierarchical setting, i.e., local learners to predict IVs and an aggregator to get theses predictions
    as input and predict the objective, i.e., total response time. 

    is_separated: If True, local learners are individual, i.e., one separate learner for prediction of each IV in each microservice. Hence, 
    you will have n local learners for n IVs in total. Otherwise, using multioutput regressor for each local model. In this case, the number of local learners 
    is equal to the number of microservices (which is 7 in the original example).
    """
    def __init__(self, app_names = None, is_hierarchical=False, is_separated=False):
        super().__init__(LinearRegression, LearningApproachMethodName.LINEAR_REGRESSION, app_names=app_names, 
                         is_hierarchical=is_hierarchical, is_separated=is_separated)
        

    def train(self, X: pd.DataFrame, y: pd.DataFrame):
        SKLearnApproach.train(self, X, y)

    def validate(self, X, y):
        return(SKLearnApproach.validate(self, X,y))


        