# Exercise: Regression (BERN02)

Author: Hafiza Asifa Naseer  
Date: 09/09/2025  

## Overview
This repository contains the solution to the exercise *Regression (BERN02) course*.  
The task was to implement a **local regression function (y, x, k, x0)** with tricube weights,that returns the predicted mean and the standard error of the expected value and choose the number of neighbors `k` by cross-validation, and apply it to the **air pollution dataset**. Predicting mortality from air pollution dataset. 

## Files
- `notebook.ipynb` – Main Jupyter notebook with all code, results, and plots.  
- `src/local_regression.py` – Python module with the local regression function.  
- `data/pollution_cleaneddata.csv` – Dataset used in the exercise.  
- `data/pollution_metadata.txt` – Description of dataset variables.  
- `requirements.txt` – Python dependencies needed to run the code.  
- `README.md` – This file.  

## Results
- The best number of neighbors was selected as **k = 13** using leave-one-out cross-validation (LOOCV).  
- Predictions for mortality (MORT) at given poverty levels (POOR):  
  - POOR = 10 → Predicted MORT = 898.56, SE = 16.74  
  - POOR = 18 → Predicted MORT = 955.96, SE = 15.11  
  - POOR = 25 → Predicted MORT = 1012.24, SE = 22.93  

The regression curve, CI band (±1 SE), and predictions are shown in the notebook output.  

## How to Run
1. Clone this repository.  https://github.com/asifakhannn/BERN02-Ex.Regression-Asifa.git 
2. Create a virtual environment and install dependencies:  
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   .venv\Scripts\activate     # Windows

   pip install -r requirements.txt
