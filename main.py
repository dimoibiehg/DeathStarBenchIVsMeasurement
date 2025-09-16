import subprocess
import multiprocessing 
import json
import re
import json
from template_manager.template_manager import TemplateManager

target_repository = "<path to the cloned DeathStarBench github repo>/DeathStarBench/hotelReservation"

def set_configuration(tm : TemplateManager, config):
    col_names = tm.col_names
    docker_compose_cols = {c:config[i] for i, c
                           in enumerate(col_names) if (not ("admin" in c))}

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
        instance = tm.fill_template(f"./files/templates/eviction_template.js",
                                    mongo_admin_struct[srv])
        tm.write_instance(instance, f"init_{srv}.d/eviction.js")


    instance = tm.fill_template("./files/templates/docker-compose_template.yml",
                                docker_compose_cols)
    tm.write_instance(instance, "docker-compose.yml")

def run_workload(target_repository=target_repository):
    result = subprocess.run("../wrk2/wrk -D exp -t 20 -c 50 -d 40 -L -s ./wrk2/scripts/hotel-reservation/mixed-workload_type_1.lua http://127.0.0.1:5000 -R 100",
                            shell=True, text=True, cwd=target_repository,
                            stdout=subprocess.PIPE)

def extract_docker_container_ids_and_names():
    try:
        result = subprocess.run("sudo docker ps --format \"{{.ID}} {{.Names}}\"", shell=True, text=True, cwd=target_repository, stdout=subprocess.PIPE)
        stats = result.stdout
         # Split the output by lines
        lines = stats.strip().split('\n')
        containers = [line.split(' ', 1) for line in lines]
        return containers

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running docker ps: {e}")
        return []

def install_perf_in_containers(container_id, container_name):
    try:
        subprocess.run(f"sudo docker exec -u 0 -it {container_id} apt-get update", shell=True, text=True, cwd=target_repository, stdout=subprocess.PIPE)
        if("mongo" in container_name):
            subprocess.run(f"sudo docker exec -u 0 -it {container_id} apt-get install -y linux-base linux-tools-generic",
                           shell=True, text=True, cwd=target_repository, stdout=subprocess.PIPE)
        
        else:
            subprocess.run(f"sudo docker exec -u 0 -it {container_id} dpkg --configure -a",
                           shell=True, text=True, cwd=target_repository, stdout=subprocess.PIPE)
            subprocess.run(f"sudo docker exec -u 0 -it {container_id} apt-get install -y linux-perf",
                           shell=True, text=True, cwd=target_repository, stdout=subprocess.PIPE)
        return True

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing perf in the container: {container_id}")
        return False


def run_perf(container_id, container_name,  return_dict, duration=30):

    event_names = ["branch-instructions", "branch-misses", "bus-cycles", "cache-misses", 
                   "cache-references", "cpu-cycles", "instructions", "mem-loads",
                   "mem-stores", "ref-cycles", "slots", "cstate_core/c6-residency/", 
                   "cstate_core/c7-residency/", "cstate_pkg/c10-residency/", "cstate_pkg/c2-residency/", 
                   "cstate_pkg/c3-residency/", "cstate_pkg/c6-residency/", "cstate_pkg/c7-residency/", 
                   "cstate_pkg/c8-residency/", "cstate_pkg/c9-residency/", "i915/actual-frequency/", "i915/bcs0-busy/",
                   "i915/bcs0-sema/", "i915/bcs0-wait/", "i915/interrupts/", "i915/rc6-residency/", "i915/rcs0-busy/", 
                   "i915/rcs0-sema/", "i915/rcs0-wait/", "i915/requested-frequency/", "i915/software-gt-awake-time/", "i915/vcs0-busy/",
                   "i915/vcs0-sema/", "i915/vcs0-wait/", "i915/vcs1-busy/", "i915/vcs1-sema/", "i915/vcs1-wait/",
                   "i915/vecs0-busy/", "i915/vecs0-sema/", "i915/vecs0-wait/", "msr/aperf/", "msr/cpu_thermal_margin/", 
                   "msr/mperf/", "msr/pperf/", "msr/smi/", "msr/tsc/", "power/energy-cores/", "power/energy-gpu/",
                   "power/energy-pkg/", "uncore_clock/clockticks/", "uncore_imc_free_running/data_read/", "uncore_imc_free_running/data_total/",
                   "uncore_imc_free_running/data_write/"]
    
    events = " -e ".join(event_names)
    if("mongo" in container_name):
        # [IMPORTANT] You should update the versoin "5.4.0-196-generic" 
        # based on the docker image kernel version of your host machine.
        result = subprocess.run(f"sudo docker exec -u 0 -it {container_id} /usr/lib/linux-tools/5.4.0-196-generic/perf stat -a -e {events} -- sleep {duration}",
                                shell=True, text=True, cwd=target_repository, stdout=subprocess.PIPE)
    else:
        result = subprocess.run(f"sudo docker exec -u 0 -it {container_id} perf stat -a -e {events} -- sleep {duration}",
                                shell=True, text=True, cwd=target_repository, stdout=subprocess.PIPE)
    stats = result.stdout
    if(len(stats) > 2):
        return_dict[container_id] = {container_name: {"raw": stats, "extracted": parse_perf_stat(stats)}}
    else:
        return_dict[container_id] = {container_name: {"raw" : result.stderr, "extracted": None}}


def parse_perf_stat(output):
    output = output.strip()
    lines = output.split('\n')
    parsed_data = {}
    
    # Regular expression to match the performance data lines with units
    pattern = re.compile(r'^\s*([\d,\.]+(?:\s+\w+)?)\s+([\w\-\/]+)\s+.*$')

    for line in lines:
        match = pattern.match(line)
        if match:
            value, event = match.groups()
            value = value.replace(',', '')
            if ' ' in value:
                number, unit = value.split()
                try:
                    number = float(number)
                except ValueError:
                    number = value  # in case the value is not a number
                parsed_data[event] = {"value": number, "unit": unit}
            else:
                try:
                    number = float(value)
                except ValueError:
                    number = value  # in case the value is not a number
                parsed_data[event] = {"value": number, "unit": None}

    # Extract the total time elapsed from the last line
    time_pattern = re.compile(r'(\d+\.\d+)\s+seconds\s+time\s+elapsed')
    for line in lines:
        match = time_pattern.search(line)
        if match:
            parsed_data['time_elapsed'] = {"value": float(match.group(1)), "unit": "seconds"}
            break

    return parsed_data


def collect_metrics() -> dict:
    subprocess.call("sudo docker compose down", shell=True, cwd=target_repository)
    subprocess.call("sudo docker compose build", shell=True, cwd=target_repository)
    subprocess.call("sudo docker compose up --build -d", shell=True, cwd=target_repository)

    containers = extract_docker_container_ids_and_names()
    # print(containers)
    for container in containers:
        install_perf_in_containers(container[0], container[1])
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    jobs = [multiprocessing.Process(target=run_workload)]
    jobs[0].start()


    for container in containers:
        p = multiprocessing.Process(target=run_perf, args=(container[0], container[1], return_dict))
        jobs.append(p)
        p.start()


    for proc in jobs:
        proc.join()
    
    return return_dict.copy()

if __name__ == "__main__":

    tm = TemplateManager(base_addr=target_repository)

    for i in range(3000):
        config = tm.generate_random_conf()
        set_configuration(tm, config)
        collected_metrics = collect_metrics()
        collected_metrics["configs"] = {c: config[i] for i, c in enumerate(tm.col_names)}
        with open(f"./files/outputs/perf-result-{i}-.json", 'w') as fp:
            json.dump(collected_metrics, fp, indent=2)
