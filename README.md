# mlops-assignment1
**Student Roll No:** G24AI2004  
**GitHub:** https://github.com/viks-iitj/mlops-assignment1.git
			git@github.com:viks-iitj/mlops-assignment1.git 

## Overview
This repository contains a reproducible pipeline to train and evaluate two regression models
(Decision Tree and Kernel Ridge) on the Boston Housing dataset. 

## Conda environment (recommended)
 Create the conda environment from the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate mlops-env
   ```
   
## Run training scripts
After activating the environment, run:
```bash
python train.py      # trains DecisionTreeRegressor and saves artifacts
python train2.py     # trains KernelRidge and saves artifacts
```

## What is saved
- `artifacts/dtree_model.joblib` - Decision Tree model
- `artifacts/kernelridge_model.joblib` - Kernel Ridge model
- `artifacts/scaler.joblib` - StandardScaler used for preprocessing

## CI (GitHub Actions)
A workflow is included under `.github/workflows/mlops.yml`. It creates a conda environment and
runs both training scripts when changes are pushed to branches `dtree`, `kernelridge`, and `main`.

## Notes
- If the runner cannot reach the dataset URL due to network restrictions, download the CMU dataset
  from `http://lib.stat.cmu.edu/datasets/boston` and place it in the working directory; the loader
  will attempt to read from that URL by default.
