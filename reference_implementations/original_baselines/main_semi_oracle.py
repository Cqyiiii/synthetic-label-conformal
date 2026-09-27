import os
import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 

import torch
import torch.optim as optim

# Import custom modules 
from losses import *
from trainers import *
from models import *
from metrics import *
from utils import *
from methods import *
from data_utils import *

# Suppress warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")


def run_benchmark_suite():
    try:
        seed_everything(42)
    except:
        np.random.seed(42)
        
    print(f"Using Device: {DEVICE}")

    # --- 1. Dataset Selection ---
    dataset_loaders = {    
        "bike": load_bike,
        "diamond": load_diamonds,            
        "gas_turbine":load_gas_turbine,        
        "naval": load_naval,    
        "SGEMM": load_sgemm_product,                               
        "superconduct": load_superconductivity,                
        "Transcoding": load_transcoding, 
        "WEC": load_wec,                 
    }

    # --- 2. Method Selection ---
    methods_to_run = [
        "Split",
        "SPI",    
        "SemiCP",
        "RCP",
        "CQR", 
    ]

    alpha = 0.1
    n_seeds = 20
    calib_prop = 0.01
    train_prop = 0.02    
    
    # Output directory for ORACLE results
    output_dir = f"./results_semi_calib{calib_prop:.2f}_train{train_prop:.2f}_oracle"
    
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    
    for ds_name, loader in dataset_loaders.items():
        print(f"\n>>>>>> Running {ds_name} <<<<<<")
        try:
            X_full, y_full = loader()
            print(f"Data Shape: {X_full.shape}, {y_full.shape}")
        except Exception as e:
            print(f"Error loading {ds_name}: {e}")
            continue

        # Initialize results container dynamically
        results = {}
        for m in methods_to_run:
            results[m] = []        
            results[f"{m}-Syn"] = []
        
        # Container for RF Labeler Performance (Still logging to compare against Oracle)
        # rf_metrics_list = []

        # Check output dimension & reshape if 1D
        if y_full.ndim == 1:
            y_full = y_full.reshape(-1, 1)
        
        y_dim = y_full.shape[1] 
        
        total_start_time = time.time()

        for seed in range(n_seeds):
            seed_everything(42+seed)
            seed_start_time = time.time()
            print(f"Seed {seed}...", end="", flush=True)

            # --- A. Data Split (Semi-Supervised) ---
            data_dict = data_split_syn(X_full, y_full, n_cal_prop=calib_prop, n_train_prop=train_prop, seed=seed)
            # X_ext, Y_ext = data_dict["ext"]


            # --- C. Generate Synthetic Labels (ORACLE MODE) ---
            # Instead of using rf.predict(), we take the TRUE labels from the data dictionary.
            
            # 1. 'syn' (Large Unlabeled)
            Y_ext_syn = data_dict['syn'][1]
            
            # 2. 'cal' (Small Labeled)
            Y_cal_syn = data_dict['cal'][1]
            
            # 3. 'mean' (Training Set)
            Y_tr_syn = data_dict['mean'][1]
            
            # Ensure shapes are correct
            if y_dim == 1:
                Y_ext_syn = Y_ext_syn.reshape(-1, 1)
                Y_cal_syn = Y_cal_syn.reshape(-1, 1)
                Y_tr_syn  = Y_tr_syn.reshape(-1, 1)
            

            # --- D. Run Selected Methods ---

            # 0. SCP (Standard Split Conformal Prediction - Supervised Baseline)
            if "Split" in methods_to_run:
                try:
                    # Uses ONLY real labeled data:
                    # 'mean' split for Training the regressor
                    # 'cal' split for Calibration (Conformalization)
                    res_Split = run_split(
                        data_dict['mean'][0], data_dict['mean'][1], 
                        data_dict['cal'][0], data_dict['cal'][1], 
                        data_dict['test'][0], data_dict['test'][1], 
                        alpha=alpha
                    )
                    results['Split'].append(res_Split)
                except Exception as e:
                    print(f" Err(Split:{e})", end="")
            
            # 1. RCP Family
            if "RCP" in methods_to_run:
                try:
                    res_syn = run_rcp_syn(
                        data_dict['mean'][0], data_dict['mean'][1], 
                        data_dict['cal'][0], data_dict['cal'][1], 
                        data_dict['test'][0], data_dict['test'][1], 
                        data_dict['syn'][0], Y_ext_syn, Y_cal_syn,
                        alpha=alpha
                    )
                    results['RCP-Syn'].append(res_syn)
                except Exception as e:
                    print(f" Err(RCP-Syn:{e})", end="")

                try:
                    res_base = run_rcp(
                        data_dict['mean'][0], data_dict['mean'][1], 
                        data_dict['cal'][0], data_dict['cal'][1], 
                        data_dict['test'][0], data_dict['test'][1], 
                        alpha=alpha
                    )
                    results['RCP'].append(res_base)
                except Exception as e:
                    print(f" Err(RCP:{e})", end="")


            # 2. CQR Family
            if "CQR" in methods_to_run:
                try:
                    res_cqr_syn = run_cqr_syn(
                        data_dict['mean'][0], data_dict['mean'][1], 
                        data_dict['cal'][0], data_dict['cal'][1], 
                        data_dict['test'][0], data_dict['test'][1], 
                        data_dict['syn'][0], Y_ext_syn, Y_tr_syn, 
                        alpha=alpha
                    )
                    results['CQR-Syn'].append(res_cqr_syn)
                except Exception as e:
                    print(f" Err(CQR-Syn:{e})", end="")

                try:
                    res_cqr = run_cqr(
                        data_dict['mean'][0], data_dict['mean'][1], 
                        data_dict['cal'][0], data_dict['cal'][1], 
                        data_dict['test'][0], data_dict['test'][1], 
                        alpha=alpha, mode='pinball'
                    )
                    results['CQR'].append(res_cqr)
                except Exception as e:
                    print(f" Err(CQR:{e})", end="")


            # 5. SPI (Synthetic-Powered Inference)
            if "SPI" in methods_to_run:
                try:
                    # SPI uses Real Cal + Synthetic Cal (with Synthetic Labels)
                    res_spi = run_spi(
                        data_dict['mean'][0], data_dict['mean'][1], 
                        data_dict['cal'][0], data_dict['cal'][1], 
                        data_dict['test'][0], data_dict['test'][1], 
                        data_dict['syn'][0], Y_ext_syn, # Using the synthetic labels here
                        alpha=alpha, beta=0.4
                    )
                    results['SPI'].append(res_spi)
                except Exception as e:
                    print(f" Err(SPI:{e})", end="")

            # 6. SemiCP (Semi-Supervised CP)
            if "SemiCP" in methods_to_run:
                try:
                    res_semicp = run_semicp(
                        data_dict['mean'][0], data_dict['mean'][1], 
                        data_dict['cal'][0], data_dict['cal'][1], 
                        data_dict['test'][0], data_dict['test'][1], 
                        data_dict['syn'][0], Y_ext_syn, Y_cal_syn,  
                        alpha=alpha
                    )
                    results['SemiCP'].append(res_semicp)
                except Exception as e:
                    print(f" Err(SemiCP:{e})", end="")

            print(f" Done ({time.time() - seed_start_time:.2f}s)")


        # --- Summary & Reporting ---
        summary_rows = []
        metrics_to_report = ['Cov', 'Size', 'WSC', 'MSCE_10', 'MSCE_30', "L1-ERT", "L2-ERT"]

        for name, mets in results.items():
            if not mets: continue
            row = {'Method': name}
            for k in metrics_to_report:
                try:
                    vals = [m[k] for m in mets]
                    row[k] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
                except KeyError:
                    row[k] = "N/A"
            summary_rows.append(row)
        
        print("\nSummary:")
        df_res = pd.DataFrame(summary_rows)
        print(df_res)
        
        # Save aggregated results
        df_res.to_csv(f"{output_dir}/{ds_name}_results.csv")
        
        # Save raw results for all methods
        for method_name, res_list in results.items():
            if res_list:
                pd.DataFrame(res_list).to_csv(f"{output_dir}/{ds_name}_raw_{method_name.lower()}.csv")

if __name__ == "__main__":
    run_benchmark_suite()