from template_manager.template_manager import TemplateManager
import numpy as np

tm = TemplateManager(base_addr="<path to the cloned DeathStarBench github repo>/DeathStarBench/hotelReservation")
col_names = tm.col_names
config = tm.generate_random_conf()

docker_compose_cols = {c:config[i] for i, c in enumerate(col_names) if (not ("admin" in c))}

mongo_admin_struct = {}
for i, c in enumerate(col_names):
    if("admin" in c):
        cs = c.split('__')
        srv_name = "_".join(cs[0].split("_")[1:])
        opt_name = "_".join(cs[1].split("_")[1:])
        if(srv_name in mongo_admin_struct):
            mongo_admin_struct[srv_name][opt_name] = config[i]
        else:
            mongo_admin_struct[srv_name] = {opt_name:config[i]}

for srv in mongo_admin_struct:
    instance = tm.fill_template(f"./files/templates/eviction_template.js", mongo_admin_struct[srv])
    tm.write_instance(instance, f"init_{srv}.d/eviction.js")


instance = tm.fill_template("./files/templates/docker-compose_template.yml", docker_compose_cols)
tm.write_instance(instance, "docker-compose.yml")
      



