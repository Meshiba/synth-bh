# Outlier Detection Experiments

This folder contains code for the outlier detection experiments presented in the paper.
All experiments can be reproduced using the provided configuration files.

---

## Environment Setup

You can create a conda environment with all required dependencies using:

```bash
conda env create -f requirements.yml
conda activate SynthBH-od
```
---

## Running Experiments

Experiments are run using `run_exp.py` with a YAML configuration file or using `main.py` with command line arguments (run `python main.py --help` for additional information).

**Example (with config file):**
```bash
python run_exp.py \
    -c ./experiments/config_files/real_data_shuttle.yml \
    -s ./results/shuttle/
```
- `-c` / `--config_path`: Path to configuration YAML file (see `./experiments/config_files/`)
- `-s` / `--save_path`: Path to save results
- `-d` / `--dataset`: Dataset name (optional, if not specified in config)
- `-v` / `--dataset_ver`: Dataset version (optional)

**Execution Modes**
- **Default**: Runs are executed via SLURM, with automatic distribution of runs across jobs.

  *Note*: When using SLURM, temporary bash scripts are created in `/home/tmp/`.  
  This folder must exist before execution. The path is defined in `create_empty_bash.sh`.

- **Run locally:**
    - Add `--local` to the command line

- **Disable run distribution:**
    - Add `--no_distribute` to the command line, or
    - In the config file under `flag_parameters`, set:
    ```yaml
    no_distribute: true
    ```
**Configuration Files:**
Example configuration files are provided in `./experiments/config_files/`

---

## Plotting Results
After experiments complete, results are stored in automatically created run-specific folders inside the results/ directory.
Each run folder contains a `results.pkl` file used for plotting.

To plot results across multiple datasets together, use `plot_main.py` and provide all paths to the directories containing the `results.pkl` file.

**Example:**
```bash
python plot_main.py \
    --x dataset \
    --result_dir ./results/shuttle/run1/results/ \
                 ./results/creditcard/run2/results/ \
                 ./results/KDDCup99/run3/results/ \
    --plot_dir ./results/plots/
```
- `--x`: Column for the x-axis (e.g., dataset)
- `--result_dir`: One or more paths to results directories  (separated by spaces).
Each path must point to the `results/` subfolder created automatically by `run_exp.py` or `main.py`, which contains `results.pkl`.
- `--plot_dir`: Path to save generated plots

---

## Notes

- Default runs assume SLURM; use `--local` or `local: true` in the config for local runs.

- Ensure you always point `plot_main.py` to the correct `results/` subdirectory (not the higher-level folder).

- All datasets and experiment configurations used in the paper are included in `./experiments/config_files/`.