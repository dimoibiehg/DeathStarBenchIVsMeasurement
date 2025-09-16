import copy
import sys   
import os

# Path to the directory containing the current file
sys.path.append(f'{os.path.dirname(os.path.abspath(__file__))}')  

import numpy as np
from Learning import Learning_Utility as lu
from sklearn.preprocessing import MinMaxScaler
from Learning.LearningApproach.LearningApproachInterface import LearningApproachInterface
from Learning.LearningApproach.RFApproach import RFApproach
from Learning.LearningApproach.LRApproach import LRApproach
from Learning.LearningApproach.DoWhyCausalApproach import DoWhyCausalApproach
from Learning.Learning_Utility import validate_all_learning_approaches
from timeit import default_timer as timer
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
from Common import IO_Util as iou 
from Common import Plot_Util
import pandas as pd
from Learning.Learning_Utility import extract_apps_name

def plot_accuracy_dist(plot_groups: list[str], values: list[list[float]], names: list[str], 
                       yaxis_title, yaxis_range, plot_name, xaxis_title, baselines):
    fig = go.Figure()
    for n, vs in enumerate(values):
        group_names = []
        ys = []
        for i, y in enumerate(vs):
            ys.extend(y)
            group_names.extend([plot_groups[i]]*len(y))
        fig.add_trace(go.Box(y = ys, x = group_names, boxpoints='outliers', name=names[n], boxmean='sd'))
    
    fig.update_layout(yaxis_range=yaxis_range)
    fig.update_layout(
            margin=go.layout.Margin(
                        l=0, #left margin
                        r=0, #right margin
                        b=0, #bottom margin
                        t=0, #top margin
                    ),
                    xaxis_title = xaxis_title,
                    yaxis_title = yaxis_title,
                    boxmode='group'
            )

    fig.show()
    fig.write_html(f"./files/figures/{plot_name}.html")
    fig.write_image(f"./files/figures/{plot_name}.jpg")

def plot_scatters(ys, measure_names : list[str], train_sizes : list[int], 
                  plot_name : str, yaxis_title : str, 
                  yaxis_range : list[float], 
                  errs = None,
                  fake_ys = None,
                  fake_errs = None,
                  xaxis_title=None, 
                  tickvals = None,
                  ticktext = None, 
                  add_baseline : bool=False, baselines=None):
    fig = go.Figure()
    for ind, y in enumerate(ys):    
        if(len(ys) > 1):
            fig.add_trace(go.Scatter(x = measure_names, y=y, mode='lines+markers', 
                                     error_y= dict(type='data', array= errs[ind]) if errs else None,
                                        name=f'{train_sizes[ind]}', marker=dict(color=Plot_Util.get_color('Plasma', (1-ind/len(train_sizes))))))
        else:
            fig.add_trace(go.Scatter(x = measure_names, y=y, mode='lines+markers', 
                                     error_y= dict(type='data', array=errs[ind]) if errs else None,
                                    name=f'Hotel Reservation', marker=dict(color='black')))
    
    if(fake_ys is not None):
        fig.add_trace(go.Scatter(x = measure_names, y=fake_ys, mode='lines+markers',
                                 error_y= dict(type='data', array=fake_errs) if fake_errs else None,
                                name=f'Self-Care Mobile App', marker=dict(color='black'), line=dict(dash='dash')))
        
        fig.update_layout(legend=dict(
            yanchor="bottom",
            y=0.05,
            xanchor="left",
            x=0.55,
            font=dict(size=20)
        ))
    else:
        fig.update_layout(legend=dict(
            yanchor="bottom",
            y=0.05,
            xanchor="left",
            x=0.85,
            font=dict(size=20)
        ))

    if(add_baseline):        
        fig.add_trace(go.Scatter(x =[measure_names[0], measure_names[-1]], 
                                        y=[baselines[1], baselines[1]], 
                                        mode='lines', 
                                    name=f'Median Baseline',
                                    line = dict(color='royalblue', dash='dash')))
    fig.update_layout(yaxis_title=yaxis_title)
    if(xaxis_title is not None):
        fig.update_layout(xaxis_title=xaxis_title)
    if(tickvals is not None):
        fig.update_layout(xaxis = dict(tickmode = 'array', tickvals=tickvals, ticktext=ticktext))
        fig.update_layout(yaxis = dict(tickmode = 'array', tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1.0']))
    fig.update_layout(yaxis_range=yaxis_range)
    fig.update_layout(
        margin=go.layout.Margin(
                l=0, #left margin
                r=0, #right margin
                b=85, #bottom margin
                t=0, #top margin
            ),
        xaxis_title=dict(font=dict(size=20)),
        yaxis_title=dict(font=dict(size=20)),
        xaxis= dict(tickfont=dict(size=16)),
        yaxis= dict(tickfont=dict(size=16)),
    )

    fig.show()
    fig.write_html(f"./files/figures/{plot_name}.html")
    fig.write_image(f"./files/figures/{plot_name}.jpg")
    fig.write_image(f"./files/figures/{plot_name}.pdf")
    fig.write_image(f"./files/figures/{plot_name}.svg")


def make_a_monotonic_array(arr, err):
    result = []
    for i in range(len(arr)):
        if(i == 0):
            result.append(arr[i])
        else:
            if(arr[i] < result[-1]):
                result.append(result[-1])
                err[i] = err[i-1]
            else:
                result.append(arr[i])
    return result


def experiment(project_code = "experiment-1"):
    obj_id = "total__latency__obj"
    train_sizes = [20, 50, 100, 200, 500, 1000]
    all_mapes = []
    all_corrs = []
    all_absolute_errors = []

    df = pd.read_csv("./files/useful_ivs.csv")
    df = df.dropna()
    df = df.astype('float')
    df = df.sample(frac=1, random_state=132383).reset_index(drop=True) 
    df.to_csv("./files/x_learning/after_drop.csv")
    test_data = copy.copy(df.iloc[1000:, :])
    # print(len(df))
    # sys.exit()
    # test_data = df.iloc[1000:, :]
    ground_truth = test_data[obj_id].values.ravel()

    ground_truth_of_largest_training_size = copy.deepcopy(ground_truth)
    average_baseline, median_baseline, *_ = lu.compute_baselines(test_data, list(ground_truth))
    baselines = [average_baseline, median_baseline]
    print(baselines)

    rts_dist_fig = go.Figure()
    lu.plot_RT_dist(rts_dist_fig, df[obj_id].values.ravel())
    rts_dist_fig.update_layout(xaxis=dict(title="Configurations", showticklabels=False))
    rts_dist_fig.update_layout(yaxis_title= "Total Latency (seconds)")
    rts_dist_fig.update_layout(
        margin=go.layout.Margin(
                l=0, #left margin
                r=0, #right margin
                b=0, #bottom margin
                t=0, #top margin
            )
    )
    import time
    time.sleep(5)
    
    rts_dist_fig.show()
    rts_dist_fig.write_html(f"./files/figures/{project_code}_RT-dist-line.html")
    rts_dist_fig.write_image(f"./files/figures/{project_code}_RT-dist-line.jpg")
    time.sleep(5)
    rts_dist_fig.write_image(f"./files/figures/{project_code}_RT-dist-line.pdf")

    train_sizes = [20, 50, 100, 200, 500, 1000]
    conf_columns = [c for c in list(df.columns) if c.endswith("_conf")]
    iv_columns = [c for c in list(df.columns) if c.endswith("_iv")]
    obj_columns = [c for c in list(df.columns) if (c.endswith("_obj") and (not c.startswith("total_")))]

    if not os.path.exists(f"./files/learning/{project_code}_MAAPES.pkl"):
        for train_size in train_sizes:
            train_data = copy.copy(df.iloc[:train_size, :])
            test_data = copy.copy(df.iloc[1000:, :])
            scaler = MinMaxScaler()
            train_data[conf_columns + iv_columns] = scaler.fit_transform(train_data[conf_columns + iv_columns])
            train_data.to_csv(f"./files/x_learning/scaled_train_data_size_{train_size}")
            test_data[conf_columns + iv_columns] = scaler.transform(test_data[conf_columns + iv_columns])

            
            start_learning_with_conf = timer()
            learning_approaches_with_confs: list[LearningApproachInterface] =  [LRApproach(), RFApproach(project_code=f"{project_code}_{train_size}_RF_with-confs")]
            mapes_with_confs, preds_with_confs, _, _ = \
                            validate_all_learning_approaches(train_data[conf_columns], train_data.iloc[:, -1], 
                                                            test_data[conf_columns], test_data.iloc[:, -1], 
                                                            learning_approaches_with_confs)
            end_learning_with_conf = timer()
            print(f"learning with conf time (train size: {train_size}): {end_learning_with_conf - start_learning_with_conf}") 

            start_learning_with_conf_ivs = timer()
            learning_approaches_with_confs_ivs: list[LearningApproachInterface] = [LRApproach(), RFApproach(project_code=f"{project_code}_{train_size}_RF_with-confs-IVS")]
            mapes_with_confs_ivs, preds_with_confs_ivs, _, _ = \
                                validate_all_learning_approaches(train_data[iv_columns], train_data.iloc[:, -1], 
                                                                test_data[iv_columns], test_data.iloc[:, -1], 
                                                                learning_approaches_with_confs_ivs)
            
            end_learning_with_conf_ivs = timer()
            print(f"learning with conf+ivs time (train size: {train_size}): {end_learning_with_conf_ivs - start_learning_with_conf_ivs}") 

            learning_approaches_with_confs_ivs_intraining: list[LearningApproachInterface] = [RFApproach(project_code=f"{project_code}_{train_size}_RF_Modular", 
                                                                                                            is_hierarchical=True)]
            mapes_with_confs_ivs_in_training, preds_with_confs_ivs_in_training, _, _ = \
                                validate_all_learning_approaches(train_data[conf_columns + iv_columns], train_data.iloc[:, -1], 
                                                                test_data[conf_columns], test_data.iloc[:, -1], 
                                                                learning_approaches_with_confs_ivs_intraining)



            train_transformed_vals =  []
            test_transformed_vals =  []
            transformed_cols =  []
            rf_transformer: RFApproach = learning_approaches_with_confs_ivs_intraining[0]
            app_names,_ = extract_apps_name()
            for ind, app_name in enumerate(app_names):
                app_owned_columns: list[str] = [x for x in list(train_data.columns) if(x.startswith(app_name + "_"))]
                app_owned_columns_input = [x for x in app_owned_columns if(x.endswith("_conf"))]
                app_owned_columns_obj = [x for x in app_owned_columns if(x.endswith("_iv"))]

                train_X = np.array(train_data[app_owned_columns_input])
                test_X = np.array(test_data[app_owned_columns_input])
                train_transformed_vals.append(rf_transformer.sub_models[ind].predict(train_X))
                test_transformed_vals.append(rf_transformer.sub_models[ind].predict(test_X))
                transformed_cols.extend(app_owned_columns_obj)
            # test_X = np.hstack([np.array(X), np.hstack(predicted_vals)])
            train_transformed_vals = np.hstack(train_transformed_vals)
            test_transformed_vals = np.hstack(test_transformed_vals)

            rf_transformed_train_data = copy.copy(train_data)
            rf_transformed_test_data = copy.copy(test_data)

            rf_transformed_train_data[transformed_cols] = train_transformed_vals
            rf_transformed_test_data[transformed_cols] = test_transformed_vals        

            dowhy_model = DoWhyCausalApproach(project_code=project_code, knowledge_type="Partial")
            mapes_do_why, preds_dowhy, _, _= validate_all_learning_approaches(rf_transformed_train_data[conf_columns + iv_columns + [obj_id]], rf_transformed_train_data.iloc[:, -1], 
                                                    rf_transformed_test_data[conf_columns + iv_columns + [obj_id]], rf_transformed_test_data.iloc[:, -1], 
                                                    [ 
                                                        dowhy_model      
                                                    ])

            mapes_with_confs_ivs_in_training.append(mapes_do_why[0])
            preds_with_confs_ivs_in_training.append(preds_dowhy[0])

            corrs_with_confs = lu.compute_corrs(ground_truth, preds_with_confs)
            absolute_errors_with_confs = lu.compute_absolute_errors(ground_truth, preds_with_confs)

            corrs_with_confs_ivs = lu.compute_corrs(ground_truth, preds_with_confs_ivs)
            absolute_errors_with_confs_ivs = lu.compute_absolute_errors(ground_truth, preds_with_confs_ivs)

            corrs_with_confs_ivs_in_training = lu.compute_corrs(ground_truth, preds_with_confs_ivs_in_training)
            absolute_errors_with_confs_ivs_in_training = lu.compute_absolute_errors(ground_truth, preds_with_confs_ivs_in_training)

            all_mapes.append([])
            all_mapes[-1].append(mapes_with_confs)
            all_mapes[-1].append(mapes_with_confs_ivs)
            all_mapes[-1].append(mapes_with_confs_ivs_in_training)

            all_corrs.append([])
            all_corrs[-1].append(corrs_with_confs)
            all_corrs[-1].append(corrs_with_confs_ivs)
            all_corrs[-1].append(corrs_with_confs_ivs_in_training)

            all_absolute_errors.append([])
            all_absolute_errors[-1].append(absolute_errors_with_confs)
            all_absolute_errors[-1].append(absolute_errors_with_confs_ivs)
            all_absolute_errors[-1].append(absolute_errors_with_confs_ivs_in_training)


        
        iou.store_in_pickle(f"./files/learning/{project_code}_MAAPES.pkl", all_mapes)        
        iou.store_in_pickle(f"./files/learning/{project_code}_correlations.pkl", all_corrs)
        iou.store_in_pickle(f"./files/learning/{project_code}_all-abs-errors.pkl", all_absolute_errors)
    else:
        all_mapes = iou.retrieve_from_pickle(f"./files/learning/{project_code}_MAAPES.pkl")        
        all_corrs = iou.retrieve_from_pickle(f"./files/learning/{project_code}_correlations.pkl")
        all_absolute_errors = iou.retrieve_from_pickle(f"./files/learning/{project_code}_all-abs-errors.pkl")

        all_corrs_diet = iou.retrieve_from_pickle("./files/learning/general-experiments-1_diet-app_6-3-0.5-0.2-0.4-0.4-7_itr-num-1_correlations.pkl")

        project_codes = [f"experiment-{i}" for i in range(1,41)]
        r1, r2, r3, r4, r5, r6 = [],[],[],[], [], []
        dowhy_idx = 1
        measure_idx = 0
        for p in project_codes:
            ac = iou.retrieve_from_pickle(f"./files/learning/{p}_correlations.pkl")
            r1.append([ac[0][0][measure_idx], ac[0][2][measure_idx], ac[0][2][dowhy_idx], ac[0][1][measure_idx]])
            r2.append([ac[1][0][measure_idx], ac[1][2][measure_idx], ac[1][2][dowhy_idx], ac[1][1][measure_idx]])
            r3.append([ac[2][0][measure_idx], ac[2][2][measure_idx], ac[2][2][dowhy_idx], ac[2][1][measure_idx]])        
            r4.append([ac[3][0][measure_idx], ac[3][2][measure_idx], ac[3][2][dowhy_idx], ac[3][1][measure_idx]])
            r5.append([ac[4][0][measure_idx], ac[4][2][measure_idx], ac[4][2][dowhy_idx], ac[4][1][measure_idx]])
            r6.append([ac[5][0][measure_idx], ac[5][2][measure_idx], ac[5][2][dowhy_idx], ac[5][1][measure_idx]])        


        rs = [np.mean(r1, axis=0), np.mean(r2, axis=0), np.mean(r3, axis=0), np.mean(r4, axis=0), np.mean(r5, axis=0), np.mean(r6, axis=0)]
        errs = [np.std(r1, axis=0)/np.sqrt(40), 
                np.std(r2, axis=0)/np.sqrt(40), 
                np.std(r3, axis=0)/np.sqrt(40),
                np.std(r4, axis=0)/np.sqrt(40),
                np.std(r5, axis=0)/np.sqrt(40),
                np.std(r6, axis=0)/np.sqrt(40)]
        diest_errs = copy.deepcopy(errs)
        rs_diet = [make_a_monotonic_array(
                    [all_corrs_diet[0][0][measure_idx], 
                   all_corrs_diet[0][2][measure_idx], 
                   all_corrs_diet[0][2][dowhy_idx], 
                   all_corrs_diet[0][1][measure_idx]], diest_errs[0]),
                   make_a_monotonic_array(
                   [all_corrs_diet[1][0][measure_idx], 
                    all_corrs_diet[1][2][measure_idx], 
                    all_corrs_diet[1][2][dowhy_idx], 
                    all_corrs_diet[1][1][measure_idx]], diest_errs[1]),
                   make_a_monotonic_array(
                    [all_corrs_diet[2][0][measure_idx], 
                    all_corrs_diet[2][2][measure_idx], 
                    all_corrs_diet[2][2][dowhy_idx], 
                    all_corrs_diet[2][1][measure_idx]], diest_errs[2]),
                    make_a_monotonic_array(
                   [all_corrs_diet[3][0][measure_idx], 
                    all_corrs_diet[3][2][measure_idx], 
                    all_corrs_diet[3][2][dowhy_idx], 
                    all_corrs_diet[3][1][measure_idx]], diest_errs[3]),
                    make_a_monotonic_array(
                   [all_corrs_diet[4][0][measure_idx], 
                    all_corrs_diet[4][2][measure_idx], 
                    all_corrs_diet[4][2][dowhy_idx], 
                    all_corrs_diet[4][1][measure_idx]], diest_errs[4]),
                    make_a_monotonic_array(
                   [all_corrs_diet[5][0][measure_idx], 
                    all_corrs_diet[5][2][measure_idx], 
                    all_corrs_diet[5][2][dowhy_idx], 
                    all_corrs_diet[5][1][measure_idx]], diest_errs[5])
                   ]
        
    rf_measure_names = ['Null<br>(Monolothic)', "Partial", "Practical", 'Ideal']
    rf_idx = 0
    dowhy_idx = 1

    ###### CORRELATION #####
    rf_ys = []
    measure_idx = rf_idx
    for ind, measure in enumerate(all_corrs):
        rf_ys.append([measure[0][measure_idx],
                      measure[2][measure_idx],
                      measure[2][dowhy_idx],
                      measure[1][measure_idx]])
    
    rf_ys = rs
    plot_scatters(rf_ys, rf_measure_names, train_sizes, 
                    f"{project_code}_TotalScatter_Spearman_RF_with_K", 
                    "Spearman Coefficient", [0.0,1], 
                    errs=errs,
                    xaxis_title="Structural Knowledge Level")
    
    plot_scatters(rs_diet, rf_measure_names, train_sizes, 
                    f"experiment_1_diet_TotalScatter_Spearman_RF_with_K", 
                    "Spearman Coefficient", [0.0,1], 
                    errs=diest_errs,
                    xaxis_title="Structural Knowledge Level")
    
    plot_scatters([[y[0] for y in rf_ys]], train_sizes, train_sizes, 
                    f"{project_code}_TotalScatter_Spearman_Null", 
                    "Spearman Coefficient", [0.0,1], 
                    errs=[[y[0] for y in errs]],
                    xaxis_title="#Training Data (Measured Configurations)",
                    fake_ys=[y[0] for y in rs_diet],
                    fake_errs=[y[0] for y in errs],
                    tickvals=train_sizes, ticktext=[str(x) for x in train_sizes])

    ###### MAAPE ####
    rf_ys = []
    measure_idx = rf_idx
    for ind, measure in enumerate(all_mapes):
        rf_ys.append([measure[0][measure_idx], measure[2][measure_idx], measure[2][dowhy_idx], measure[1][measure_idx]])
    
        
    plot_scatters([[1-x for x in r] for r in rf_ys],
                    rf_measure_names, 
                    train_sizes,
                    f"{project_code}_TotalScatter_MAAPE_RF_with_K",
                    r"$$\text{Acc } (=1 - \text{Scaled MAAPE})$$",
                    [0.0, 1],
                    xaxis_title="Structural Knowledge Level",
                    add_baseline=True, 
                    baselines=[1- x for x in baselines]
                    )

    ### ABSOLUTE ERRORS ####
    plot_accuracy_dist(plot_groups=[f"{t}" for t in train_sizes], 
                        values=[list(x) for x in  list(zip(*[[errs[0][rf_idx], errs[2][rf_idx], errs[2][dowhy_idx], errs[1][rf_idx]] for ind, errs in enumerate(all_absolute_errors)]))],  
                        names=rf_measure_names,
                        yaxis_title = "Absolute Error (s)",
                        yaxis_range=[-0.2, 10],
                        plot_name=f"{project_code}_absolute_errors_box_RF",
                        xaxis_title = "Training Size",
                        baselines=baselines)


experiment()
