from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import chisq, fisherz, kci
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

import numpy as np

class CausalModel:
    def __init__(self, columns, app_names):
        print("initializing CausalModel class")      
        self.colmap={}
        for i in range(len(columns)):
            self.colmap[i] = columns[i]  
        self.app_names = app_names   
        
    def get_tabu_edges(self, columns, options, 
                       objectives):
        """This function is used to exclude edges which are not possible"""
        tabu_edges=[]
        # constraint on configuration options
        for opt in options:
            for cur_elem in columns:
                if cur_elem != opt:
                    tabu_edges.append((cur_elem, opt))

       
       # constraints on performance objetcives  
        for obj in objectives:
            for cur_elem in columns:
                if cur_elem != obj:
                    tabu_edges.append((obj, cur_elem))

        total_ivs = []
        for app_name in self.app_names:
            app_owned_columns: list[str] = [x for x in columns if(app_name in x)]
            app_owned_columns_input = [x for x in app_owned_columns if(x.endswith("_Conf"))]
            app_owned_columns_ivs = [x for x in app_owned_columns if(x.endswith("_IV"))]
            total_ivs.extend(app_owned_columns_ivs)
            if("SCM" in app_name):
                for iv in app_owned_columns_ivs:
                    tabu_edges.append(("SCM_App_Is_Fast_Conf", iv))
        
        for iv_1 in total_ivs:
            for iv_2 in total_ivs:
                if(iv_1 != iv_2):
                    tabu_edges.append((iv_1, iv_2))
        return tabu_edges
        
    def get_required_edges(self, columns):
        """This function is used to include edges which should be in the model (i.e., from domain knowledge)"""
        required_edges = []
        objs = [x for x in columns if(x.endswith("_Obj"))]
        for app_name in self.app_names:
            app_owned_columns: list[str] = [x for x in columns if(app_name in x)]
            app_owned_columns_input = [x for x in app_owned_columns if(x.endswith("_Conf"))]
            app_owned_columns_ivs = [x for x in app_owned_columns if(x.endswith("_IV"))]
            for conf in app_owned_columns_input:
                if("_SD" in conf):
                    for iv in app_owned_columns_ivs:
                        if("_CM_" not in iv):
                            required_edges.append((conf, iv))
                elif("_DC_" in conf):
                    for iv in app_owned_columns_ivs:
                        if("_CM_" in iv):
                            required_edges.append((conf, iv))
                elif("_DComp_" in conf):
                    for iv in app_owned_columns_ivs:
                        if("_CM_" not in iv):
                            required_edges.append((conf, iv))
                    if("DV" not in app_name):
                        for app_name_2 in self.app_names:
                            if("DV" in app_name_2):
                                app_owned_columns_2: list[str] = [x for x in columns if(app_name_2 in x)]
                                app_owned_columns_ivs_2 = [x for x in app_owned_columns_2 if(x.endswith("_IV"))]
                                for iv in app_owned_columns_ivs_2:
                                    if("_CM_" not in iv):
                                        required_edges.append((conf, iv))
                            
            if(("SCM" not in app_name) and ("DV" not in app_name)):
                for iv in app_owned_columns_ivs:
                    required_edges.append(("SCM_App_Is_Fast_Conf", iv))
            
            cm_iv = ""
            for iv in app_owned_columns_ivs:
                if("_CM_" in iv):
                    cm_iv = iv
                    break

            for iv in app_owned_columns_ivs:
                if("_CM_" not in iv):
                    for obj_item in objs:
                        required_edges.append((iv, obj_item))
                    required_edges.append((cm_iv, iv))
        
        # required_edges.append(("SCM_App_Is_Fast_Conf", "Total_RT_Obj"))
        return required_edges


    def learn_fci(self, df, tabu_edges, required_edges):
        """This function is used to learn model using FCI"""
        corr_method = fisherz
        G, edges = fci(df, corr_method, 0.05, verbose=False)  
        nodes = G.get_nodes()   
        bk = BackgroundKnowledge()      
        for ce in tabu_edges:
            f = list(self.colmap.keys())[list(self.colmap.values()).index(ce[0])]
            s = list(self.colmap.keys())[list(self.colmap.values()).index(ce[1])]     
            bk.add_forbidden_by_node(nodes[f], nodes[s])
        for ce in required_edges:
            f = list(self.colmap.keys())[list(self.colmap.values()).index(ce[0])]
            s = list(self.colmap.keys())[list(self.colmap.values()).index(ce[1])]     
            bk.add_required_by_node(nodes[f], nodes[s])
        try:
            G, edges = fci(df, corr_method, 0.05, verbose=False, background_knowledge=bk)
        except:
            G, edges = fci(df, corr_method, 0.05, verbose=False)
        fci_edges = []
        for edge in edges:
            fci_edges.append(str(edge))
        
        return fci_edges

    def resolve_edges(self, DAG, PAG, 
                      columns, tabu_edges,
                      objectives):
        """This function is used to resolve fci (PAG) edges"""
        bi_edge = "<->"
        directed_edge = "-->"
        undirected_edge = "o-o"
        trail_edge = "o->"
        #  entropy only contains directed edges.
        
        options = {}
        for opt in columns:
            options[opt]= {}
            options[opt][directed_edge]= []
            options[opt][bi_edge]= []
        # add DAG edges to current graph
        for edge in DAG:
            if edge[0] or edge[1] is None:
                options[edge[0]][directed_edge].append(edge[1])
        # replace trail and undirected edges with single edges using entropic policy
        for i in range (len(PAG)):
            if trail_edge in PAG[i]:
                PAG[i]=PAG[i].replace(trail_edge, directed_edge)
            elif undirected_edge in PAG[i]:
                    PAG[i]=PAG[i].replace(undirected_edge, directed_edge)
            else:
                continue
        # update causal graph edges
        
        for edge in PAG:
            cur = edge.split(" ")
            if cur[1]==directed_edge:
                
                node_one = self.colmap[int(cur[0].replace("X", ""))-1]               
                node_two = self.colmap[int(cur[2].replace("X", ""))-1]
                options[node_one][directed_edge].append(node_two)
            elif cur[1]==bi_edge:
                node_one = self.colmap[int(cur[0].replace("X", ""))-1]
                node_two = self.colmap[int(cur[2].replace("X", ""))-1]
                
                options[node_one][bi_edge].append(node_two)
            else: print ("[ERROR]: unexpected edges")
        # extract mixed graph edges 
        single_edges=[]
        double_edges=[]
        
        for i in options:
            
            options[i][directed_edge]=list(set(options[i][directed_edge]))
            options[i][bi_edge]=list(set(options[i][bi_edge]))
        for i in options:
            for m in options[i][directed_edge]:
                single_edges.append((i,m))
            for m in options[i][bi_edge]:
                double_edges.append((i,m))
        s_edges=list(set(single_edges)-set(tabu_edges))
        single_edges = []
        for e in s_edges: 
            if e[0]!=e[1]:
                single_edges.append(e)
        
        for i in range(int(len(s_edges)/2)):
             for obj in objectives:
                 if s_edges[i][0]!=s_edges[i][1]:
                     single_edges.append((s_edges[i][0],obj))
       
        double_edges=list(set(double_edges)-set(tabu_edges))
        return single_edges, double_edges