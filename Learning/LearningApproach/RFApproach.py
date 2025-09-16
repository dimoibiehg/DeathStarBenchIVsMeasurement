from Learning.LearningApproach.SKLearn_Approach import SKLearnApproach
from sklearn.ensemble import RandomForestRegressor
from Learning.LearningApproach.LearningApproachEnums import LearningApproachMethodName
import random
from sklearn import metrics
from sklearn import model_selection
import numpy as np
from Learning.Learning_Utility import mape_scorer, maape_scorer
import keras_tuner
from functools import partial
import sys
import os
sys.path.append(f'/home/{os.getlogin()}/Documents/Projects/test/DSB_Sim/')  
from template_manager import domain
from Learning.Learning_Utility import extract_apps_name

class RFApproach(SKLearnApproach):
    def __init__(self, project_code, app_names = None, random_seed = 4321, is_hierarchical=False):
        super().__init__(RandomForestRegressor, LearningApproachMethodName.RANDOM_FOREST, app_names=app_names, is_hierarchical=is_hierarchical)
        self.random_seed = random_seed
        self.project_code = project_code
        self.app_names, _ = extract_apps_name()

    def train(self, X, y):
        search_space_model = {
            "random_state": {"_type": "choice", "_value": [self.random_seed]},
            "n_estimators": {"_type": "quniform", "_value": [100, 500, 50]},
            "max_depth": {"_type": "quniform", "_value": [5, 45, 5]},
            "max_features": {"_type": "choice", "_value": ['auto', 'sqrt']},
            "min_samples_leaf": {"_type": "quniform", "_value": [1,4,1]},
            "min_samples_split": {"_type": "quniform", "_value": [2,11,3]}
        }

        search_space_submodel = {
            "random_state": {"_type": "choice", "_value": [self.random_seed]},
            "n_estimators": {"_type": "quniform", "_value": [100, 200, 20]},
            "max_depth": {"_type": "choice", "_value": [None]},
            "max_features": {"_type": "choice", "_value": ['auto', 'sqrt']},
            "min_samples_leaf": {"_type": "quniform", "_value": [1,4,1]},
            "min_samples_split": {"_type": "quniform", "_value": [2,11,3]}
        }


        if(not self.is_hierarchical):
            train_X = np.array(X)
            train_y = y.values.ravel()
            cv_fold = 2
            tuner = keras_tuner.tuners.SklearnTuner(
                oracle=keras_tuner.oracles.BayesianOptimizationOracle(
                    objective=keras_tuner.Objective('score', 'min'),
                    max_trials=30),
                hypermodel= partial(self.build_model, features_num = train_X.shape[1]),
                scoring=metrics.make_scorer(maape_scorer),
                cv=model_selection.KFold(cv_fold, random_state=self.random_seed, shuffle=True),
                directory="./files/learning_cache/",
                project_name=f'RF_Tuning_{self.project_code}')

            tuner.search(train_X, train_y)
            best_model: RandomForestRegressor = tuner.get_best_models(num_models=1)[0]
            best_model.random_state = self.random_seed
            self.learning_model = best_model
            SKLearnApproach.train(self, X,y)
        else:
            sub_models = []
            predicted_vals =  []
            total_features = []
            true_IVs = []
            all_configs = [x for x in list(X.columns) if(x.endswith("_conf"))]
            cv_fold = 2
            ivs_cols = []
            for app_name in self.app_names:
                app_owned_columns: list[str] = [x for x in list(X.columns) if(x.startswith(app_name + "_"))]
                app_owned_columns_input = [x for x in app_owned_columns if(x.endswith("_conf"))]
                
                print(app_owned_columns)
                print(app_owned_columns_input)

                total_features.extend(app_owned_columns_input)
                app_owned_columns_obj = [x for x in app_owned_columns if(x.endswith("_iv") or x.endswith("_obj"))]
                print(app_owned_columns_obj)

                ivs_cols.extend(app_owned_columns_obj)

                train_X = np.array(X[app_owned_columns_input])
                
                train_y = X[app_owned_columns_obj].values
                tuner = keras_tuner.tuners.SklearnTuner(
                oracle=keras_tuner.oracles.BayesianOptimizationOracle(
                    objective=keras_tuner.Objective('score', 'min'),
                    max_trials=30),
                hypermodel= partial(self.build_sub_model, features_num = len(app_owned_columns_input)) ,
                scoring=metrics.make_scorer(maape_scorer),
                cv=model_selection.KFold(cv_fold, random_state=self.random_seed, shuffle=True),
                directory="./files/learning_cache/",
                project_name=f'RF_Tuning_{self.project_code}_modular_{app_name}')

                tuner.search(train_X, train_y)
                best_model: RandomForestRegressor = tuner.get_best_models(num_models=1)[0]
                best_model.random_state = self.random_seed
                lr_model = best_model
                sub_models.append(lr_model)

                predicted_vals.append(lr_model.predict(train_X))
                true_IVs.append(train_y)

            super_data_X = np.hstack(predicted_vals)
            tuner = keras_tuner.tuners.SklearnTuner(
                oracle=keras_tuner.oracles.BayesianOptimizationOracle(
                    objective=keras_tuner.Objective('score', 'min'),
                    max_trials=30),
                hypermodel= partial(self.build_model, features_num = super_data_X.shape[1]),
                scoring=metrics.make_scorer(maape_scorer),
                cv=model_selection.KFold(cv_fold, random_state=self.random_seed, shuffle=True),
                directory="./files/learning_cache/",
                project_name=f'RF_Tuning_{self.project_code}_modular_total')


            super_data_y = y.values.ravel()
            tuner.search(super_data_X, super_data_y)
            best_model: RandomForestRegressor = tuner.get_best_models(num_models=1)[0]

            lr_super_model = best_model
            lr_super_model.fit(super_data_X, super_data_y)

            self.sub_models = sub_models
            self.learning_model = lr_super_model

    def validate(self, X, y) -> float:
        return(SKLearnApproach.validate(self, X,y))

    def build_sub_model(self, hp: keras_tuner.HyperParameters, features_num):
        model = RandomForestRegressor(
            random_state= self.random_seed,
            n_estimators= 150,
            max_depth= hp.Int('max_depth', features_num//5, features_num+1, step = features_num//5) if (features_num > 5) else hp.Int('max_depth', 1, features_num, step = 1),
            max_features='sqrt',
            min_samples_leaf=hp.Int('min_samples_leaf', 1, 4),
            min_samples_split=hp.Int('min_samples_split', 2, 11, step=3),
            )
        return model

    def build_model(self, hp: keras_tuner.HyperParameters, features_num):
        model = RandomForestRegressor(
            random_state= self.random_seed,
            n_estimators=150,
            max_depth= hp.Int('max_depth', features_num//5, features_num+1, step = features_num//5),
            max_features="sqrt",
            min_samples_leaf=hp.Int('min_samples_leaf', 1, 4),
            min_samples_split=hp.Int('min_samples_split', 2, 11, step=3),
            )
        return model
