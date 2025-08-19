# Classic GP vs. Vecchia GP in 1-D

## Overview
This script demonstrates a comparison between a **classic Gaussian Process (GP)** and the **Vecchia approximation** in **one dimension**.  
In 1-D, the Vecchia method is equivalent to the classic GP, so both models produce nearly identical results.

## What the Code Does
1. Loads input data from `data_for_vecchia1d.csv` (two columns: surrogate and target).
2. Introduces a missing gap in the target signal.
3. Uses surrounding data to train both the classic GP and the Vecchia GP.
4. Optimizes model hyperparameters with `fminunc`.
5. Predicts the missing region with both models.
6. Produces plots:
   - Observed vs. predicted data (classic vs. Vecchia).
   - Comparison of learned model parameters.

## Expected Outcome
- Both methods reconstruct the missing data in 1-D with almost identical predictions.
- Plots confirm the equivalence of the two approaches in this setting.

## Requirements
- MATLAB (with Optimization Toolbox).
- Data file: `data_for_vecchia1d.csv`.

## Usage
Place the script and data in the same folder, then run the script in MATLAB.  
The results will show side-by-side predictions and parameter comparisons.
