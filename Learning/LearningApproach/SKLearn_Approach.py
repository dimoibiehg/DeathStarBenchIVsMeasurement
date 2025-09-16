import sys
from Learning.LearningApproach.LearningApproachInterface import LearningApproachInterface
from Learning.LearningApproach.LearningApproachEnums import  LearningApproachMethodName
import numpy as np
import copy
import pandas as pd

class SKLearnApproach(LearningApproachInterface):
    def __init__ (self, learning_type, learning_name: LearningApproachMethodName,
                  app_names = None,  is_hierarchical=False, is_separated=False):
        self.learning_name = learning_name
        self.learning_type = learning_type
        self.learning_model = learning_type()
        self.app_names = app_names
        self.is_hierarchical = is_hierarchical  
        self.is_separated = is_separated
        self.sub_models = []
    def train(self, X: pd.DataFrame , y: pd.DataFrame):
        if(not self.is_hierarchical):
            train_X = np.array(X)
            train_y = y.values.ravel()
            self.learning_model.fit(X,y)
        else:
            sub_models = []
            predicted_vals =  []
            total_features = []
            all_configs = [x for x in list(X.columns) if(x.endswith("_Opt"))]
            for app_name in self.app_names:
                app_owned_columns: list[str] = [x for x in list(X.columns) if(app_name in x)]
                app_owned_columns_input = [x for x in app_owned_columns if(x.endswith("_Opt"))]
                total_features.extend(app_owned_columns_input)
                app_owned_columns_obj = [x for x in app_owned_columns if(x.endswith("_IV"))]
                train_X = np.array(X[all_configs])
                
                if(self.is_separated):
                    sub_sub_models = []
                    sub_preds = []
                    for obj_col in app_owned_columns_obj:
                        train_y = X[obj_col].values.ravel()
                        lr_model = self.learning_type()
                        lr_model.fit(train_X, train_y)
                        sub_sub_models.append(lr_model)  
                        sub_preds.append(lr_model.predict(train_X))
                    
                    predicted_vals.append(np.array([list(l) for l in list(zip(*sub_preds))]))
                    sub_models.append(sub_sub_models)
                else:
                
                    train_y = X[app_owned_columns_obj].values
                    lr_model = self.learning_type()
                    lr_model.fit(train_X, train_y)
                    sub_models.append(lr_model)

                    predicted_vals.append(lr_model.predict(train_X))

            
            lr_super_model = self.learning_type()
            super_data_X = np.hstack([np.array(X[total_features]), np.hstack(predicted_vals)])
            super_data_y = y.values.ravel()
            lr_super_model.fit(super_data_X, super_data_y)

            self.sub_models = sub_models
            self.learning_model = lr_super_model

        

    def validate(self, X: pd.DataFrame , y: pd.DataFrame):

        test_X = copy.deepcopy(X)
        test_y = copy.deepcopy(y)

        if(self.is_hierarchical):
            predicted_vals =  []
            all_configs = [x for x in list(X.columns) if(x.endswith("_conf"))]
            for ind, app_name in enumerate(self.app_names):
                app_owned_columns: list[str] = [x for x in list(X.columns) if(x.startswith(app_name + "_"))]
                app_owned_columns_input = [x for x in app_owned_columns if(x.endswith("_conf"))]
                
                train_X = np.array(X[app_owned_columns_input])

                if(self.is_separated):
                    sub_preds = []
                    for lr_model in self.sub_models[ind]:
                        sub_preds.append(lr_model.predict(train_X))
                    predicted_vals.append(np.array([list(l) for l in list(zip(*sub_preds))]))
                else:
                    predicted_vals.append(self.sub_models[ind].predict(train_X))
            
            test_X = np.hstack(predicted_vals)
            test_y = y.values.ravel()
        
        else:
            test_X = np.array(X)
            test_y = y.values.ravel()

        eps = sys.float_info.epsilon
        pred = self.learning_model.predict(test_X)
        error = [np.arctan(np.abs((x - y - eps)/(y + eps))) for x,y in zip(pred, test_y)]
        maape = (sum(error)/len(error)) * 2 / np.pi
        return (maape, pred)
        
    
    
    
