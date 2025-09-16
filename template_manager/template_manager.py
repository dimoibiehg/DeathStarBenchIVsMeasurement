from template_manager import domain
import numpy as np
from jinja2.nativetypes import NativeEnvironment
# with open("dna.txt", "r") as file:
#     dna = file.read().replace("\n", "")

# print(dna)

class TemplateManager:
    def __init__(self, base_addr, rng_state = 1001):
        self.rng = np.random.RandomState(rng_state)
        self.col_names : list[str] = []
        self.base_addr = base_addr
        self.__get_col_names()
    
    def __generate_rand_val(self, min_val, max_val, step):
        step_num = int((max_val-min_val)//step)
        rand_val = min_val + self.rng.randint(0, step_num+1) * step
        return rand_val
    
    def __get_col_names(self) -> None:
        col_names = []
        for srv in domain.list_of_services:
            for option in domain.service_docker_options:
                col_names.append(f"{srv}__{option}__conf")
            
            if(domain.list_of_services[srv]["has_mongo"]):
                for option in domain.mongo_docker_options:
                    col_names.append(f"mongodb_{srv}__{option}__conf")
                    # template_names.append(col_names[-1])
                for option in domain.mongo_admin_options:
                    col_names.append(f"mongodb_{srv}__admin_{option}__conf")
                    # template_names.append(option)

            if(domain.list_of_services[srv]["has_memcache"]):
                for option in domain.memcache_docker_options:
                    srv_name = srv
                    if("memcache_alias" in domain.list_of_services[srv]): srv_name = domain.list_of_services[srv]["memcache_alias"]
                    col_names.append(f"memcached_{srv_name}__{option}__conf")
        self.col_names = col_names

    def generate_random_conf(self) -> list:
        conf = []

        for srv in domain.list_of_services:
            option_dicts = [domain.service_docker_options]
            if(domain.list_of_services[srv]["has_mongo"]):
                option_dicts.append(domain.mongo_docker_options)
                option_dicts.append(domain.mongo_admin_options)
            if(domain.list_of_services[srv]["has_memcache"]):
                option_dicts.append(domain.memcache_docker_options)
            for option_dict in option_dicts:        
                for option in option_dict:
                    min_val = option_dict[option]["min_val"]
                    max_val = option_dict[option]["max_val"]
                    step = option_dict[option]["step"]

                    rand_val = self.__generate_rand_val(min_val, max_val, step)
                    if(type(rand_val).__name__ == "float"):
                        conf.append(round(rand_val, 2))
                    else:
                        conf.append(rand_val)
        return conf
    def generate_distance_based_config(self):
        conf = []
        for srv in domain.list_of_services:
            option_dicts = [domain.service_docker_options]
            if(domain.list_of_services[srv]["has_mongo"]):
                option_dicts.append(domain.mongo_docker_options)
                option_dicts.append(domain.mongo_admin_options)
            if(domain.list_of_services[srv]["has_memcache"]):
                option_dicts.append(domain.memcache_docker_options)

               
            for option_dict in option_dicts:
                length = len(option_dict)
                idx_range = list(range(length))
                self.rng.shuffle(idx_range)
                distance = self.rng.randint(0, length+1)
                idxs =  [idx_range[i] for i in range(distance)]
                counter = 0 
                for option in option_dict:
                    min_val = option_dict[option]["min_val"]
                    max_val = option_dict[option]["max_val"]
                    step = option_dict[option]["step"]
                    if(counter in idxs):
                        rand_val = self.__generate_rand_val(min_val, max_val, step)
                    else:
                        rand_val = min_val
                    
                    if(type(rand_val).__name__ == "float"):
                        conf.append(round(rand_val, 2))
                    else:
                        conf.append(rand_val)
                    
                    counter += 1
        return conf
     
    def fill_template(self, file_addr, config) -> str:
        env = NativeEnvironment()
        tmpl_file = ""
        with open(file_addr, "r") as file:
            tmpl_file = file.read()
        tmpl = env.from_string(tmpl_file)
        docker_compose_instance = tmpl.render(config)
        return docker_compose_instance

    def write_instance(self, docker_compose_instance: str, file_name: str) -> bool:
        try: 
            with open(f"{self.base_addr}/{file_name}", "w") as f:
                f.write(docker_compose_instance)
            return True
        except Exception as e:
            print(e)
            return False

        