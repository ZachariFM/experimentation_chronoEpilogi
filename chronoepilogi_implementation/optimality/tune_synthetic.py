import os
import sys
import pandas as pd
import numpy as np
import time

import func_timeout


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import tuning_single
import forecasters
import feature_selectors



results_directory = ".\\results\\"
gt_directory = "..\\data\\characteristics.csv"
data_path = "..\\data\\"
seed = 0
train_test_split = 0.8
first_phase_trials = None
second_phase_trials = None
# maximal time (s) of a study
# partially complete studies are counted as completed, unless one configuration takes more than 2 minutes to evaluate
study_time_limit = 2400


# list of synthetic data and files to evaluate
file_beg, file_end = 0,2
print(file_beg, file_end)
filelist = sorted(os.listdir("..\\data\\SynthBase\\"))[file_beg:file_end+1]
filelist = list(map(lambda x: ".".join(x.split(".")[:-1]), filelist))
mts_list = list(zip(["SynthBase"]*len(filelist), filelist))


# ground truth opening
gt_dataframe = pd.read_csv(gt_directory)

for mts_description in mts_list:
    # open data
    dataset_name = mts_description[0]
    mts_name = mts_description[1]

    mts_path = data_path + dataset_name + "\\" + mts_name+".csv"
    data = pd.read_csv(mts_path, compression="gzip")
    tuning_data = data.iloc[:int(len(data)*train_test_split)]
    tuning_data.columns = list(map(str, tuning_data.columns))  # column names as str

    # open gt 
    gt_df = gt_dataframe[gt_dataframe["dataset_name"]==dataset_name]
    gt_df = gt_df[gt_df["mts_name"]==mts_name]
    gt = gt_df["th.equivalence_classes"].values[0]
    gt = eval(gt)# put in list of list form
    gt = [list(map(str,c)) for c in gt]  # column names as str

    # set data-dependent parameters
    forecasting_horizon = 1
    lookback_window = int(gt_df["th.lags"].values[0])
    target = "0"
    variable_types = {"numerical":[x for x in tuning_data.columns], "categorical":[]}

    # set fs algo to try and set forecasters to try
    list_fs = ["NoSelection","T-MRMR", "CausalPairs","SingleChronoEpilogi", "GroupLasso"]
    
    if dataset_name !="SynthNonLin":
        list_forecasters = ["LinearDartsWrapper"]
    else:
        list_forecasters = ["LinearDartsWrapper", "XGBDartsWrapper"]

    # main loop
    t = [time.time()]
    for fs_name in list_fs:
        # set fs algo base config
        fs_constructor, fs_config = feature_selectors.build_base_configuration(fs_name, variable_types, lookback_window, seed)
        for forecaster_name in list_forecasters:
            print(mts_name, fs_name, forecaster_name)
            # set forecaster algo base config
            es_constructor, es_config = forecasters.build_base_configuration(forecaster_name,forecasting_horizon, lookback_window)
            # launch
            try:
                result = func_timeout.func_timeout(study_time_limit+1200,
                    tuning_single.tune_single_solution,
                    args=(tuning_data, 
                        dataset_name+","+mts_name,
                        fs_name, forecaster_name,
                        fs_constructor, es_constructor,
                        fs_config, es_config,
                        target),
                    kwargs={"causal_ground_truth":gt,
                        "nb_folds":3,
                        "seed":seed,
                        "minimal_training_fraction":0.7,
                        "first_phase_trials":first_phase_trials,
                        "second_phase_trials":second_phase_trials,
                        "time_limit":study_time_limit})
                
                if result is not None and isinstance(result, tuple) and len(result) == 2:
                    df_params, df_results = result
                    tuning_single.save_tuning_results(results_directory+dataset_name ,df_params, df_results)
                else:
                    print("Warning: tune_single_solution did not return expected results for", fs_name, forecaster_name)
            
            except func_timeout.FunctionTimedOut:
                print("Timeout on",fs_name, forecaster_name)
            t.append(time.time())
    print("time spent for the file:",np.diff(t))

