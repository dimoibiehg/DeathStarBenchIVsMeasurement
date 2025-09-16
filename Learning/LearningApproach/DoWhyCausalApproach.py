from Learning.LearningApproach.LearningApproachInterface import LearningApproachInterface
from Learning.LearningApproach.LearningApproachEnums import  LearningApproachTypeEnum
from Learning.LearningApproach.LearningApproachEnums import LearningApproachMethodName 
import dowhy.gcm as gcm
import networkx as nx
import numpy as np 
import pandas as pd
from tqdm import tqdm
from wrapt_timeout_decorator import *
from scipy.stats import uniform
from causallearn.utils.cit import CIT
from Learning.Learning_Utility import make_knowledge_graph_on_domain
import json

class DoWhyCausalApproach(LearningApproachInterface):
    def __init__ (self, project_code, p_value_threshold = 0.05, knowledge_type="Complete"):
        self.learning_type : LearningApproachTypeEnum = LearningApproachTypeEnum.DO_WHY_CAUSALML
        self.learning_model = None
        self.learning_name = LearningApproachMethodName.CAUSAL_STRUCTURE_MODEL
        self.knowledge_type = knowledge_type
        self.p_value_threshold = p_value_threshold
        self.project_code = project_code    

    def train(self, X,y):
        if(self.knowledge_type == "Complete"):
            causal_graph : nx.DiGraph = make_knowledge_graph_on_domain(X)
        else:
            causal_graph = self.causal_graph_discovery(X)

        self.learning_model = gcm.causal_models.StructuralCausalModel(causal_graph)
        columns = list(X.columns)
        training_size = X.shape[0]
        for col in columns:
            if("_conf" in col):
                self.learning_model.set_causal_mechanism(col, gcm.ScipyDistribution(uniform))
            elif("_iv" in col):
                app_name = col.split("__")[0]
                path = "./files/learning_cache/" + \
                        f'RF_Tuning_{self.project_code}_{training_size}_RF_Modular_modular_{app_name}' + \
                        '/trial_29/trial.json'
                values = None
                with open(path, 'r') as file:
                    data = json.load(file)
                    values = data["hyperparameters"]["values"]

                print(path)
                print(values)
                self.learning_model.set_causal_mechanism(col, gcm.AdditiveNoiseModel(
                    gcm.ml.create_random_forest_regressor(n_estimators=150, max_depth=values["max_depth"],
                                                                           max_features='sqrt', 
                                                                           min_samples_leaf=values["min_samples_leaf"], 
                                                                           min_samples_split=values["min_samples_split"], 
                                                                            random_state=4321),
                    noise_model= None #gcm.ScipyDistribution(norm)
                    ))
            elif("_obj" in col):
                if("total_" in col):
                    path = "./files/learning_cache/" + \
                            f'RF_Tuning_{self.project_code}_{training_size}_RF_Modular_modular_total' + \
                            '/trial_29/trial.json'
                    values = None
                    with open(path, 'r') as file:
                        data = json.load(file)
                        values = data["hyperparameters"]["values"]

                    print(path)
                    print(values)
                    self.learning_model.set_causal_mechanism(col, gcm.AdditiveNoiseModel(
                        gcm.ml.create_lasso_regressor(positive=True, alphas = [0.1], n_alphas=1),
                        noise_model= None))

        gcm.fit(self.learning_model, X)

    def validate(self, X, y):
        def make_func(y):
            return lambda x: y
        pred = []
        df_all_data = X #pd.concat([X, y], axis = 1)
        perf_cols = ["total__latency__obj"]
        for i in tqdm(range(len(df_all_data))):
            var_dict = {str(node): make_func(float(df_all_data[[str(node)]].values[i][0])) 
                        for node in list(X.columns) if(("_conf" in node))}
            pred.append(gcm.interventional_samples(self.learning_model, var_dict,
                                    num_samples_to_draw=1)[perf_cols].values[0][0])

        error = [np.arctan(np.abs((x-y)/y)) for x,y in zip(pred, y)]
        maape = (sum(error)/len(error)) * 2 / np.pi
        return (maape, pred)

    def causal_graph_discovery(self, dataframe: pd.DataFrame):
        cols = list(dataframe.columns)

        causal_graph : nx.DiGraph = make_knowledge_graph_on_domain(dataframe)

        causal_graph_copy : nx.DiGraph = make_knowledge_graph_on_domain(dataframe)
        all_edges = causal_graph_copy.edges(data=False)
        all_nodes = causal_graph_copy.nodes(data=False)

        for edge in all_edges:
            if(edge[1].endswith("_obj")):
                continue
            cit = CIT(dataframe.to_numpy(), method="kci", kernelZ='Gaussian',  approx=True, est_width='median')
            node_0_idx = -1
            node_1_idx = -1
            for i,x in enumerate(cols):
                if((node_0_idx > -1) and (node_1_idx > -1)):
                    break
                elif(x == edge[0]):
                    node_0_idx = i
                elif(x == edge[1]):
                    node_1_idx = i
            pVal = cit(node_0_idx, node_1_idx, None)
            if(pVal > self.p_value_threshold):
                causal_graph.remove_edge(*edge)
            else:
                if(("_iv" in edge[0]) and ("_iv" in edge[1])):
                    pVal_2 = cit(node_1_idx, node_0_idx, None)
                    if(pVal_2 < pVal):
                        causal_graph.remove_edge(*edge)
                        causal_graph.add_edge(edge[1], edge[0])

        for i, node in enumerate(list(all_nodes)):
            if(("_iv" in node) and  (causal_graph.in_degree(node) < 1)):
                p_vals = []
                opts = [(j, opt) for j, opt in enumerate(list(all_nodes)) if ("_conf" in opt)]
                for opt in opts:
                    p_vals.append(cit(opt[0], i, None))
                causal_graph.add_edge(opts[np.argsort(p_vals)[0]][1], node)
        return causal_graph
                