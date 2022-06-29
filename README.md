# Self-Supervised Time Series Representation Learning with Temporal-Instance Similarity Distillation
This repository contains the official implementation of the paper "Self-Supervised Time Series Representation Learning with Temporal-Instance Similarity Distillation" (ICML 2022 Pretraining Workshop).

## Requirements
We use `conda` to create a new environment for running experiments, called `atom-ssl-ts`, using `env.yaml`:
```bash
conda env create -f env.yaml
conda activate atom-ssl-ts
```
We used precommit hooks (`flake8` and `black`) to reformat the code but their use is optional. You don't need the hooks to reproduce the results of the paper. To use precommit hooks, simply run `pre-commit install` before commiting your code (that is based off of this project).

## Data
We used 5 different datasets to evaluate our results. Use below links to download these datasets.

* [128 UCR datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018)
* [30 UEA datasets](http://www.timeseriesclassification.com)
* [3 ETT datasets](https://github.com/zhouhaoyi/ETTDataset)
* [Electricity dataset](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)
* [KPI dataset](http://test-10056879.file.myqcloud.com/10056879/test/20180524_78431960010324KPI%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B%E5%86%B3%E8%B5%9B%E6%95%B0%E6%8D%AE%E9%9B%86.zip)

The electricity and KPI datasets need preprocessing. Simply run the preprocessing scripts under `datasets` to create their corresponding preprocessed versions. Use `--output` to indicate where you want to save the results to.

The data path to all datasets is specified as a constant under `PATH` in `constants.py`. Each dataset is stored in this path under its corresponding folder. Make sure to update this path, before running the code.

## Experiments

To reproduce the experiments, simply use the scripts that are provided in the root directory of the project under `scripts`. These scripts would use `slurm` to submit the jobs to the cluster. You need to specify the node name and the name of the job to run the script. Refer to `scripts/submit_job_forecasting.sh` for farther details on how to choose `dataset`, `run_name`, and `loader` to run forecasting experiments. The scripts run the pipeline with 5 different seeds so each script submits 5 jobs to the cluster. 

```bash
# UCR dataset:
bash scripts/submit_job_ucr.sh <node_name> <job_name>

# UEA dataset
bash scripts/submit_job_uea.sh <node_name> <job_name>

# KPI dataset:
bash scripts/submit_job_kpi.sh <node_name> <job_name>

# ETT and electricity dataset
bash scripts/submit_job_forecasting.sh <node_name> <job_name> <dataset> <run_name> <loader>

# Example:
bash scripts/submit_job_forecasting.sh sample_node sample_job ETTh2 forecast_univar forecast_csv_univar
```

### Hyper-parameters

The following table shows the set of hyper-parameters used in all of our experiments:

| Temperature | Queue Size | Alpha | Dropout | Learning Rate | Momentum |
| ------------|------------|-------|---------|---------------|----------|
| 0.07        | 128        | 0.5   | 0.1     | 0.001         | 0.999    |
