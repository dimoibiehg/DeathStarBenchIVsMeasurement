import json
import os
import pandas as pd
from template_manager import domain 



useful_metrics_name = ["instructions", "mem-stores"]
obj_metric_name = "cpu-cycles"

path = "./files/outputs"
dir_list: list[str] = os.listdir(path)
dfs = []
units = []
list_of_untouched_units = ["M", "seconds", "C", "MiB", "Joules"]
for f_name in dir_list:
    if f_name.startswith("perf-result-"):
        id = int(f_name.split('-')[2])
        record = {}
        total_obj = 0
        with open(os.path.join(path, f_name), 'r') as f:
            data = json.load(f)
            record["file_idx"] = id

            configs = data["configs"]
            for x in configs:
                app_name, config_name, _ = x.split("__")
                corresponding_dict = {}
                if(app_name.startswith("mongodb")):
                    if(config_name.startswith("admin")):
                        corresponding_dict = domain.mongo_admin_options
                        config_name = config_name[len("admin_"):]
                    else:
                        corresponding_dict = domain.mongo_docker_options
                elif(app_name.startswith("memcached")):
                    corresponding_dict = domain.memcache_docker_options
                else:
                    corresponding_dict = domain.service_docker_options

                max_val = corresponding_dict[config_name]["max_val"]
                min_val = corresponding_dict[config_name]["min_val"]
                step = corresponding_dict[config_name]["step"]
                step_num = int((max_val-min_val)//step)
                idx = 0
                if('int' in str(type(min_val))):
                    idx = (int(configs[x]) - min_val)//step
                else:
                    idx = int((float(configs[x]) - min_val)//step)
        
                for i in range(step_num+1):
                    record[f"{app_name}__{config_name}-{i}__conf"] = 0
                
                record[f"{app_name}__{config_name}-{idx}__conf"] = 1

            found_error = False
            for key in data:
                if key == "configs":
                    pass
                else:
                    container_name : str = list(data[key].keys())[0]
                    has_mongo = False
                    has_mem = False
                    if("mongo" in container_name): has_mongo = True
                    elif("memcach" in container_name): has_mem = True

                    if(data[key][container_name]["extracted"] is None):
                        found_error = True
                        break
                    
                    idx = -1
                    
                    splitted_container_name = container_name.split("-")
                    for srv in domain.list_of_services:
                        found_srv = False
                        if(len(splitted_container_name) > 1):
                            key_container_name = "-".join(splitted_container_name[1:])
                            if(has_mem):
                                if("memcache_alias" in domain.list_of_services[srv]): 
                                    if(domain.list_of_services[srv]["memcache_alias"] in key_container_name):
                                        found_srv = True
                                else:
                                    if(srv in key_container_name):
                                        found_srv = True
                            if(srv in key_container_name):
                                found_srv = True
                        else:
                            if(srv == "reservation"):
                                if(container_name.count(srv) == 2):
                                    found_srv = True
                            elif(srv == "review"):
                                if(container_name.endswith("mmc")):
                                    found_srv = True
                                    has_mem = True
                                elif(srv in container_name):    
                                    found_srv = True
                            else:
                                if(srv in container_name):
                                    found_srv = True

                        if(found_srv):
                            for metric in data[key][container_name]["extracted"]:
                                if metric in useful_metrics_name + [obj_metric_name]:
                                    val = data[key][container_name]["extracted"][metric]["value"]
                                    u = data[key][container_name]["extracted"][metric]["unit"]
                                    metric_name = ""
                                    metric_scale = 1
                                    metric_type = ""
                                    if metric == obj_metric_name:
                                        metric_type = "obj"
                                        metric_scale = 4700 * 1e6 * 100
                                        total_obj += val/metric_scale
                                        metric = "latency"
                                    elif metric in useful_metrics_name:
                                        metric_type = "iv"
                                        if(u is not None):
                                            if(u == "ns"):
                                                metric_scale = 1e9
                                            elif(u not in list_of_untouched_units):
                                                raise Exception(f"unknow unit: {u}")
                                            
                                    if(has_mongo):
                                        assert domain.list_of_services[srv]["has_mongo"]
                                        metric_name = f"mongodb_{srv}__{metric}__{metric_type}"
                                    
                                    elif(has_mem):
                                        assert domain.list_of_services[srv]["has_memcache"]
                                        srv_name = srv
                                        if("memcache_alias" in domain.list_of_services[srv]): srv_name = domain.list_of_services[srv]["memcache_alias"]
                                        metric_name = f"memcached_{srv_name}__{metric}__{metric_type}"
                                    else:
                                        metric_name = f"{srv}__{metric}__{metric_type}"
                                    
                                    record[metric_name] = val/metric_scale

                            break
        record["total__latency__obj"] = total_obj

        if(not found_error):
            dfs.append(record)
            

data = pd.DataFrame(dfs)
data.to_csv("./files/useful_ivs.csv")



