# DeathStarBench Performance Measurement and Causal Learning

This project provides a comprehensive framework for measuring performance metrics of the [DeathStarBench](https://github.com/delimitrou/DeathStarBench) hotel reservation microservice application under different configurations, and applying machine learning techniques (including causal inference) to understand the relationships between configuration parameters and performance outcomes.

## Overview

The project consists of three main phases:
1. **Data Collection**: Automated measurement of performance metrics across randomized configurations
2. **Data Preparation**: Processing and cleaning of collected measurements for machine learning
3. **Learning Experiments**: Application of various ML approaches including causal inference to understand configuration-performance relationships

## Prerequisites

- Python 3.12
- Docker and Docker Compose
- DeathStarBench repository cloned locally
- `wrk2` load testing tool (provided in DeathStartBench project)
- Linux system with `perf` tools support
- Root/sudo access for Docker operations

## Project Structure
├── main.py                     # Data collection orchestrator
├── data_preparation.py         # Data preprocessing pipeline
├── learning_experiment.py      # ML experiments and analysis
├── template_manager/           # Configuration template management
├── Learning/                   # ML framework and approaches
│   ├── LearningApproach/      # Different learning algorithms
│   │   ├── LRApproach.py      # Linear Regression
│   │   ├── RFApproach.py      # Random Forest
│   │   └── DoWhyCausalApproach.py  # Causal inference
│   └── Learning_Utility.py    # ML utilities and validation
├── Common/                     # Shared utilities
├── files/                      # Data storage
│   ├── outputs/               # Raw measurement data
│   ├── templates/             # Configuration templates
│   ├── learning/              # ML results and models
│   └── figures/               # Generated plots and visualizations
└── README.md

## Installation and Setup

### 1. Clone DeathStarBench
```bash
git clone https://github.com/delimitrou/DeathStarBench.git
```

### 2. Install wrk2
In the root directory of the cloned DeathStarBench repository, run the followings:
```bash
cd wrk2
make
```

### 3. Install Python Dependencies
```bash
pip install pandas numpy scikit-learn plotly tqdm dowhy keras-tuner
```

### 4. Configure Project
Update the `target_repository` path in `main.py`:
```python
target_repository = "<path to the cloned DeathStarBench github repo>/DeathStarBench/hotelReservation"
```

### 5. Ensure Kernel Version Compatibility
Check the kernel version inside the MongoDB Docker container and update the perf command path in `main.py`:

```bash
# Check the kernel version inside the MongoDB container
sudo docker run --rm -it mongo:latest uname -r
```
Then update the kernel version in the run_perf function in main.py to match the Docker image's kernel:

```python
# Update "5.4.0-196-generic" to match the MongoDB Docker image's kernel version
result = subprocess.run(f"sudo docker exec -u 0 -it {container_id} /usr/lib/linux-tools/5.4.0-196-generic/perf stat...")
```

This is important because:
1. The MongoDB Docker image has its own kernel tools installed
2. The perf binary path inside the container must match the kernel version the image was built with
3. Different MongoDB image versions may have different underlying kernel versions


## Usage

### Phase 1: Data Collection
Run the main data collection script to gather performance measurements across 3000 random configurations:

```bash
python3 main.py
```

What this does:
- Generates random configurations using the template manager
- Deploys each configuration to the DeathStarBench hotel reservation system
- Runs workload tests using wrk2
- Collects detailed performance metrics using Linux perf tools
- Stores results in ./files/outputs/ as JSON files
- Expected runtime: Several hours to days depending on system performance

### Phase 2: Data Preparation
Process the collected raw data for machine learning:

```bash
python3 data_preparation.py
```

What this does:
- Reads all JSON files from ./files/outputs/
- Extracts useful metrics (instructions, mem-stores, cpu-cycles)
- Cleans and normalizes the data
- Outputs processed data to ./files/useful_ivs.csv

### Phase 3: Learning Experiments
Run machine learning experiments and causal analysis:

```bash
python3 learning_experiment.py
```

What this does:
- Loads prepared data from ./files/useful_ivs.csv
- Applies multiple learning approaches:
    - Linear Regression (LR)
    - Random Forest (RF)
    - Causal inference using DoWhy
- Generates performance comparison plots
- Evaluates different levels of structural knowledge
- Outputs results to ./files/learning/ and visualizations to ./files/figures/

## Key Features
### Multi-Level Learning Approaches
- **Null (Monolithic)**: Traditional ML without structural knowledge
- **Partial**: Incorporates some domain knowledge
- **Practical**: Uses hierarchical modeling with intermediate variables
- **Ideal**: Full causal modeling with complete structural knowledge

### Performance Metrics
- **MAAPE**: Mean Arctangent Absolute Percentage Error
- **Spearman Correlation**: Rank correlation between predictions and actual values
- **Absolute Error**: Direct error measurements

### Causal Inference
- Integration with DoWhy library for causal discovery and inference
- Structural causal models for understanding configuration impacts
- Interventional analysis capabilities


## Output Files
### Data Files
- `./files/useful_ivs.csv`: Processed dataset ready for ML
- `./files/learning/*_MAAPES.pkl`: Accuracy results across approaches
- `./files/learning/*_correlations.pkl`: Correlation analysis results
### Visualizations
- `./files/figures/*_RT-dist-line.*`: Response time distribution plots
- `./files/figures/*_TotalScatter_*.svg`: Performance comparison scatter plots
- `./files/figures/*_absolute_errors_box_*.jpg`: Error distribution box plots

## Configuration
### Measured Metrics
The system collects comprehensive performance counters including:
- CPU cycles, instructions, cache metrics
- Memory operations (loads/stores)
- Power consumption (cores, GPU, package)
- Hardware-specific counters (Intel i915, MSR events)

### Workload Configuration
- Load testing: 20 threads, 50 connections, 40-second duration
- Request rate: 100 RPS
- Mixed workload pattern for hotel reservation system

## Troubleshooting
### Common Issues
1. **Docker Permission Errors**
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

2. **Perf Tool Installation Issues**
- Ensure kernel headers are installed
- Verify perf tools match kernel version
3. **Memory Issues with Large Datasets**
- Reduce the number of configurations in `main.py`
- Process data in smaller batches
4. **Missing Dependencies**
```bash
pip install --upgrade pandas numpy scikit-learn plotly tqdm dowhy
```
## Research Applications
This framework is designed for:
- Performance modeling of microservice applications
- Causal analysis of configuration impacts
- Automated performance tuning research
- Comparative analysis of ML approaches for systems performance
## Contributing
When extending the project:
1. Add new learning approaches in `Learning/LearningApproach/`
2. Implement the `LearningApproachInterface` for consistency
3. Update `LearningApproachEnums.py` for new approach types
4. Add corresponding evaluation metrics in `Learning_Utility.py`

## Citation
If you use this code in your research, please cite our paper:

```
@misc{gheibi2025hardnessstructuralknowledgeopportunity,
      title={Hardness, Structural Knowledge, and Opportunity: An Analytical Framework for Modular Performance Modeling}, 
      author={Omid Gheibi and Christian Kästner and Pooyan Jamshidi},
      year={2025},
      eprint={2509.11000},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2509.11000}, 
}
```

## License

This project builds upon the [DeathStarBench](https://github.com/delimitrou/DeathStarBench) suite. Please refer to the original DeathStarBench license for usage terms.

## Contact

For questions or feedback, please contact `omid.gheibi@gmail.com`, `kaestner@cs.cmu.edu`, `pjamshi@cse.sc.edu`.
