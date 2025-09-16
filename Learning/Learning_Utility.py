import sys
import numpy as np
import pandas as pd
from Learning.LearningApproach.LearningApproachInterface import LearningApproachInterface
import math
import plotly.graph_objects as go
from scipy.stats import spearmanr
from template_manager import domain
import networkx as nx

def mape_scorer(ground_truth, predictions):
    error = [np.abs((x-y)/y) for x,y in zip(predictions, ground_truth)]
    mape = (100*sum(error)/len(error))
    return mape

def maape_scorer(ground_truth, predictions):
    eps = sys.float_info.epsilon
    error = [np.arctan(np.abs((x - y - eps)/(y + eps))) for x,y in zip(predictions, ground_truth)]
    maape = (sum(error)/len(error)) * 2 / np.pi
    return maape

def generate_distance_based_config(length: int, rng: np.random.RandomState, distance: int):
    conf = [0] * length
    idx_range = list(range(length))
    rng.shuffle(idx_range)
    for i in range(distance):
        conf[idx_range[i]] = 1
    return conf


def validate_all_learning_approaches(train_X, train_y, test_X, test_y, learning_approaches: list[LearningApproachInterface]):

        validation_results_primary = []
        validation_results_primary_on_training = []
        preds = []
        preds_on_training = []
        for learning_approach in learning_approaches:
            learning_approach.train(train_X,train_y)
            (valid_primary_res, pred) = learning_approach.validate(test_X, test_y)
            validation_results_primary.append(valid_primary_res)
            preds.append(pred)

            print(f"{learning_approach.get_learning_name()}--primary:{valid_primary_res}")
            print("===================================================")

        return validation_results_primary, preds, validation_results_primary_on_training, preds_on_training

def compute_baselines(test_data: pd.DataFrame, new_seed_rts: list[float]) -> (float, float, float, float):
    test_mean = test_data.iloc[:, -1].mean()
    test_median = test_data.iloc[:, -1].median()
    ground_truth = list(test_data.iloc[:, -1].values)
    average_baseline =  maape_scorer(ground_truth, [test_mean] * len(ground_truth))
    median_baseline =  maape_scorer(ground_truth, [test_median] * len(ground_truth))
    noise_baseline =  maape_scorer(ground_truth, new_seed_rts)
    absolute_error_baseline = [np.abs(x-y) for x,y in zip(ground_truth, new_seed_rts)]
    return (average_baseline, median_baseline, noise_baseline, absolute_error_baseline)

def identify_jumps_in_performance(diff_sorted_rts, diff_cutoff=0.5, conitnuity_radius=100):
    cutoffs = [(i, x) for i, x in enumerate(diff_sorted_rts) if x > diff_cutoff]
    groups = []
    for c in cutoffs:
        if(len(groups) == 0):
            groups.append([c])
        else:
            index_of_last_memebr_of_last_detected_group = groups[-1][-1][0]
            if((c[0] - index_of_last_memebr_of_last_detected_group) < conitnuity_radius):
                groups[-1].append(c)
            else:
                groups.append([c])
    return groups

def kl_divergence(p, q):
    eps = sys.float_info.epsilon
    return sum(p[i] * math.log2((p[i] + eps)/(q[i] + eps)) for i in range(len(p)))

def compute_quality_measures(all_mapes_test_PR, 
                             all_corrs_test_PR, 
                             all_absolute_errors_test_PR, 
                             basline, all_rts):
    baseline_absolute_errors = basline[-1]
    klds = [[kl_divergence(baseline_absolute_errors, y) 
             if(len(y) == len(baseline_absolute_errors)) else 0.0  for y in x] 
            for x in all_absolute_errors_test_PR]
    eps = 1e-5
    klds_ratio = math.fabs(klds[2][1] / klds[2][0])
    median_noise_distance = basline[1] - basline[2]

    sorted_rts = sorted(all_rts)
    diff_step = 1
    diff_sorted_rts = [(sorted_rts[i] - sorted_rts[i-diff_step])/diff_step  for i in range(diff_step, len(sorted_rts))]
    jumps = identify_jumps_in_performance(diff_sorted_rts, diff_cutoff=0.5, conitnuity_radius=100/diff_step)
    n_jumps = len(jumps)

    maape_diff = math.fabs(all_mapes_test_PR[2][1] - all_mapes_test_PR[2][0])
    
    maape_ratio_PR_with_IVs =  math.fabs(( all_mapes_test_PR[2][1] - all_mapes_test_PR[5][1])/
                                         (eps + all_mapes_test_PR[0][1] - all_mapes_test_PR[2][1]))

    maape_ratio_PR = math.fabs((all_mapes_test_PR[2][0] - all_mapes_test_PR[5][0])/
                               (eps + all_mapes_test_PR[0][0] - all_mapes_test_PR[2][0]))

    corr_diff = math.fabs(all_corrs_test_PR[2][1] - all_corrs_test_PR[2][0])

    corr_ratio_PR_with_IVs = math.fabs((all_corrs_test_PR[2][1] - all_corrs_test_PR[5][1])/
                                       (eps + all_corrs_test_PR[0][1] - all_corrs_test_PR[2][1]))

    corr_ratio_PR = math.fabs((all_corrs_test_PR[2][0] - all_corrs_test_PR[5][0])/
                              (eps + all_corrs_test_PR[0][0] - all_corrs_test_PR[2][0]))
    
    return [klds_ratio, median_noise_distance, n_jumps,
            maape_diff, maape_ratio_PR_with_IVs, maape_ratio_PR, 
            corr_diff, corr_ratio_PR_with_IVs, corr_ratio_PR]

def compute_corrs(ground_truth, preds):
    corrs = []
    for i, pred in enumerate(preds):
        spearman_result = spearmanr(ground_truth, pred, nan_policy='omit')
        corrs.append(spearman_result.statistic)

    corrs = [0.0 if math.isnan(x) else x for x in corrs]
    return corrs

def compute_absolute_errors(ground_truth, preds):
    errors = []
    for i, pred in enumerate(preds):
        error = [np.abs(x-y) for x,y in zip(ground_truth, pred)]
        errors.append(error)
    return errors

def plot_RT_dist(fig: go.Figure, rts: list[float]):
    sorted_rts = sorted(rts)
    diff_step = 1
    diff_sorted_rts = [(sorted_rts[i] - sorted_rts[i-diff_step])/diff_step  for i in range(diff_step, len(sorted_rts))]

    fig.add_trace(go.Scatter(x = list(range(len(rts))), y = sorted_rts))
    return fig


def extract_apps_name() -> tuple[list[str],list[str]]:
    app_names = []
    srv_names = []
    for srv in domain.list_of_services:
        app_names.append(srv)
        srv_names.append(srv)
        if(domain.list_of_services[srv]["has_mongo"]):
            app_names.append(f"mongodb_{srv}")
            srv_names.append(srv)
        if(domain.list_of_services[srv]["has_memcache"]):
            srv_name = srv
            if("memcache_alias" in domain.list_of_services[srv]):
                srv_name = domain.list_of_services[srv]["memcache_alias"]
            app_names.append(f"memcached_{srv_name}")
            srv_names.append(srv)
    return app_names, srv_names


def make_knowledge_graph_on_domain(X):
    g = nx.DiGraph()
    obj_id = "total__latency__obj"
    app_names, srv_names = extract_apps_name()
    for i, app_name in enumerate(app_names):
        app_owned_columns: list[str] = [x for x in list(X.columns) if(x.startswith(app_name + "_"))]
        configs = [x for x in app_owned_columns if(x.endswith("_conf"))]
        ivs = [x for x in app_owned_columns if(x.endswith("_iv"))]
        for c in configs:
            for iv in ivs:
                g.add_edge(c, iv)
        
        for iv in ivs:
            g.add_edge(iv, obj_id)

        origin_srv_idx = -1
        for j in range(i-1, -1, -1):
            if(srv_names[i] == srv_names[j]):
                origin_srv_idx = j
                continue
            break

        if(origin_srv_idx > -1):
            new_app_owned_columns: list[str] = [x for x in list(X.columns) if(x.startswith(app_names[origin_srv_idx] + "_"))]
            new_configs = [x for x in new_app_owned_columns if(x.endswith("_conf"))]
            new_ivs = [x for x in new_app_owned_columns if(x.endswith("_iv"))]
        
            for c in new_configs:
                for iv in ivs:
                    g.add_edge(c, iv)

        
        if("search" in app_name):
            rate_owned_columns: list[str] = [x for x in list(X.columns) if(("rate" + "_") in x)]
            rate_configs = [x for x in rate_owned_columns if(x.endswith("_conf"))]

            geo_owned_columns: list[str] = [x for x in list(X.columns) if(("geo" + "_") in x)]
            geo_configs = [x for x in geo_owned_columns if(x.endswith("_conf"))]

            for c in rate_configs:
                for iv in ivs:
                    g.add_edge(c, iv)

            for c in geo_configs:
                for iv in ivs:
                    g.add_edge(c, iv)
        
        elif("front" in app_name):
            profile_owned_columns: list[str] = [x for x in list(X.columns) if("profile" in x)]
            profile_configs = [x for x in profile_owned_columns if(x.endswith("_conf"))]

            recom_owned_columns: list[str] = [x for x in list(X.columns) if("recom" in x)]
            recom_configs = [x for x in recom_owned_columns if(x.endswith("_conf"))]

            attract_owned_columns: list[str] = [x for x in list(X.columns) if("attrac" in x)]
            attract_configs = [x for x in attract_owned_columns if(x.endswith("_conf"))]

            reserv_owned_columns: list[str] = [x for x in list(X.columns) if("reserv" in x)]
            reserv_configs = [x for x in reserv_owned_columns if(x.endswith("_conf"))]

            review_owned_columns: list[str] = [x for x in list(X.columns) if("review" in x)]
            review_configs = [x for x in review_owned_columns if(x.endswith("_conf"))]

            search_owned_columns: list[str] = [x for x in list(X.columns) if("search" in x)]
            search_configs = [x for x in search_owned_columns if(x.endswith("_conf"))]

            user_owned_columns: list[str] = [x for x in list(X.columns) if("user" in x)]
            user_configs = [x for x in user_owned_columns if(x.endswith("_conf"))]
            
            for c in profile_configs:
                for iv in ivs:
                    g.add_edge(c, iv)

            for c in attract_configs:
                for iv in ivs:
                    g.add_edge(c, iv)

            for c in recom_configs:
                for iv in ivs:
                    g.add_edge(c, iv)

            for c in reserv_configs:
                for iv in ivs:
                    g.add_edge(c, iv)
            
            for c in review_configs:
                for iv in ivs:
                    g.add_edge(c, iv)
            
            for c in search_configs:
                for iv in ivs:
                    g.add_edge(c, iv)

            for c in user_configs:
                for iv in ivs:
                    g.add_edge(c, iv)

    return g
