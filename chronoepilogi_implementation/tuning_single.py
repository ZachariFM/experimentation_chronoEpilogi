import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
import time
import os

import optuna
from sklearn.preprocessing import StandardScaler, PowerTransformer
from scipy.stats import skew

import metrics

def generator_cross_validation_folds(data: pd.DataFrame, 
                                      nb_folds=1,
                                      minimal_training_fraction=0.5,
                                      minimal_validation_fraction=0.1,
                                      minimal_training_length=0,
                                      minimal_validation_length=0,
                                      minimal_training_span: Optional[pd.Timedelta]=pd.Timedelta(0),  # unused
                                      minimal_validation_span: Optional[pd.Timedelta]=pd.Timedelta(0)):  # unused
    
    minimal_validation_length = max(int(np.ceil(len(data)*minimal_validation_fraction)), minimal_validation_length)
    training_length_start = max(int(np.ceil(len(data)*minimal_training_fraction)), minimal_training_length)

    actual_fold_length = (len(data) - training_length_start)/nb_folds
    assert actual_fold_length >= minimal_validation_length
    
    for fold_number in range(nb_folds):
        end_training_set = training_length_start + int(fold_number * actual_fold_length)
        end_validation_set = training_length_start + int((fold_number+1) * actual_fold_length)
        training_df = data.iloc[:end_training_set]
        validation_df = data.iloc[end_training_set:end_validation_set]

        yield fold_number, training_df, validation_df 




def flatten_dict(d:Dict[str, Any], name="")->Dict[str,Any]:
    if len(d)==0:return d
    flat = pd.json_normalize(d).to_dict(orient="records")[0]
    flat = {name+"."+str(key): value for key, value in flat.items()}
    return flat





def smart_transform_fit(df, feature_list, skew_threshold=0.5, verbose=False):
    df_transformed = df.copy()
    transformers = {}

    for feature in feature_list:
        data = df_transformed[[feature]].values

        if np.std(data) == 0:
            if verbose: print(f"[Skip Transform] {feature}: Feature is constant.")
            scaler = StandardScaler()
            scaler.fit(data)  
            transformers[feature] = {'type': 'constant', 'scaler': scaler}
            continue
        
        feature_skewness = skew(df_transformed[feature])

        if abs(feature_skewness) > skew_threshold:
            if verbose: print(f"[Transform] {feature}: Skewness {feature_skewness:.2f} -> Applying Yeo-Johnson")
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            try:
                pt.fit(data)
                data = pt.transform(data)  # need to transform before scaling
            except:
                if verbose: print(f"[Error during Yeo-Johnson] {feature}: Skewness {feature_skewness:.2f} -> Just scaling")
            transformers[feature] = {'type': 'yeo-johnson', 'transformer': pt}
        else:
            if verbose: print(f"[Skip Yeo-Johnson] {feature}: Skewness {feature_skewness:.2f} -> Just scaling")

        scaler = StandardScaler()
        scaler.fit(data)  
        transformers[feature] = transformers.get(feature, {})
        transformers[feature]['scaler'] = scaler

    return transformers

def smart_transform_apply(df, transformers):
    df_transformed = df.copy()

    for feature in transformers.keys():
        data = df_transformed[[feature]].values
        trans_info = transformers[feature]

        # Only apply Yeo-Johnson if it was fitted
        if trans_info.get('type') == 'yeo-johnson':
            pt = trans_info['transformer']
            data = pt.transform(data)

        # Always apply StandardScaler
        scaler = trans_info['scaler']
        data = scaler.transform(data)

        df_transformed[feature] = data.flatten()

    return df_transformed

def smart_transform_inverse(df, transformers):
    df_transformed = df.copy()
    
    for feature in transformers.keys():
        if feature in df_transformed.columns:
            data = df_transformed[[feature]].values
            trans_info = transformers[feature]
            # Only apply Yeo-Johnson if it was fitted
            if trans_info.get('type') == 'yeo-johnson':
                pt = trans_info['transformer']
                data = pt.inverse_transform(data)
            # Always apply StandardScaler
            scaler = trans_info['scaler']
            data = scaler.inverse_transform(data)

            df_transformed[feature] = data.flatten()

    return df_transformed


def create_time_related_features(time_index):
    """time_index should be a pandas.Index"""
    if isinstance(time_index, pd.DatetimeIndex):
        time_related_features = pd.DataFrame(index=time_index)
        timestamp_as_series = pd.Series(time_index,index=time_index)
        time_related_features['hour_sin'] = np.sin(2 * np.pi * timestamp_as_series.dt.hour / 24)
        time_related_features['hour_cos'] = np.cos(2 * np.pi * timestamp_as_series.dt.hour / 24)
        time_related_features['day_of_week_sin'] = np.sin(2 * np.pi * timestamp_as_series.dt.dayofweek / 7)
        time_related_features['day_of_week_cos'] = np.cos(2 * np.pi * timestamp_as_series.dt.dayofweek / 7)
        time_related_features['interval_sin'] = np.sin(2 * np.pi * timestamp_as_series.dt.minute // 15 / 4)
        time_related_features['interval_cos'] = np.cos(2 * np.pi * timestamp_as_series.dt.minute // 15 / 4)
    else:
        time_related_features = pd.DataFrame(index=time_index)

    return time_related_features



def cross_validation_single(es_constructor, es_config, target, model_name, metric_dict, dataset, fs_sets=None, **cv_generator_kwargs):
    records = []
    for fold_id, data_train, data_val in generator_cross_validation_folds(data=dataset, **cv_generator_kwargs):
        # select TS according to fs_sets
        if fs_sets is not None:
            if isinstance(fs_sets, list) and isinstance(fs_sets[0],list):
                data_train = data_train[fs_sets[fold_id]]
                data_val = data_val[fs_sets[fold_id]]
            elif isinstance(fs_sets, list):
                data_train = data_train[fs_sets]
                data_val = data_val[fs_sets]

        # generate time features
        data_train_time = create_time_related_features(data_train.index)
        data_val_time = create_time_related_features(data_val.index)

        # transform / scale
        transformers = smart_transform_fit(data_train, data_train.columns)
        data_train_transform = smart_transform_apply(data_train, transformers)
        data_val_transform = smart_transform_apply(data_val, transformers)
        # configure
        es_config = es_constructor.infer_data_shape_parameters(es_config, data_train_transform, data_train_time)
        estimator = es_constructor(es_config, target, model_name=model_name+str(fold_id))

        # train / predict
        t=time.time()
        estimator.fit(data_train_transform, data_train_time, data_val_transform, data_val_time)
        training_time = time.time()-t
        t=time.time()
        y_pred, y_true = estimator.predict(data_val_transform, data_val_time)
        predict_time = time.time()-t
        ## problem: we need the time index of every element.

        # sample-wise inverse transform
        total_pred, total_true = [],[]  # concatenate all predictions
        for sample_pred, sample_true in zip(y_pred, y_true):
            df_sample_pred = pd.DataFrame(columns=[target], data=sample_pred)
            df_sample_true = pd.DataFrame(columns=[target], data=sample_true)
            df_sample_pred = smart_transform_inverse(df_sample_pred, transformers)
            df_sample_true = smart_transform_inverse(df_sample_true, transformers)
            total_pred.append(df_sample_pred)
            total_true.append(df_sample_true)
        df_sample_pred = pd.concat(total_pred).reset_index(drop=True)  # reset index is here since we do not have the time index
        df_sample_true = pd.concat(total_true).reset_index(drop=True)  # reset index is here since we do not have the time index

        row = dict()
        for metric in metric_dict:
            if metric!="mase":
                row[metric] = metric_dict[metric](df_sample_true, df_sample_pred)
            else:
                row[metric] = metric_dict[metric](df_sample_true, df_sample_pred, data_train[target])
        records.append({"fold":fold_id, "forecaster.training_time":training_time, "forecaster.predict_time":predict_time, **row})
    return records, df_sample_pred, df_sample_true


def create_fs_instances_for_cv(tuning_df, target, fs_constructor, fs_base_config, **cv_generator_kwargs):
    fs_instances = []
    for fold_id, data_train, _ in generator_cross_validation_folds(data=tuning_df, **cv_generator_kwargs):
        # transform / scale
        transformers = smart_transform_fit(data_train, data_train.columns)
        data_train_transform = smart_transform_apply(data_train, transformers)
        # configure
        fs_instance = fs_constructor(fs_base_config, target, data_train_transform)
        fs_instances.append(fs_instance)
    return fs_instances

def sum_cv_metrics(records)->dict:
    # records is a list of dict, each dict having the CV-metrics of one fold.
    # perform the average across folds
    return dict([(k,np.mean([records[i][k] for i in range(len(records))])) for k in records[0] if k!="fold"])  


def generate_optuna_objective_function_single_solution(
        fs_constructor,
        fs_base_config,  # parameters that cannot be infered from the data or randomly chosen
        forecaster_constructor,
        forecaster_base_config,   # parameters that cannot be infered from the data or randomly chosen
        tuning_df,
        target, 
        causal_ground_truth:Optional[List[List[str]]] = None,
        objective="r2",
        **cv_generator_kwargs):
    
    memorize = defaultdict(list)  # object to put all configurations for simple retrieval.
    
    # define once for all the feature selection instance, to allow for fast parameter tuning when only significance thresholds are changing.
    # due to cross-validation, we must define one instance per fold
    fs_instances = create_fs_instances_for_cv(tuning_df, target, fs_constructor, fs_base_config, **cv_generator_kwargs)
    # unique experiment identifier with start time
    experiment_timestamp = time.time()

    def optuna_objective_function(trial):
        # get parameters
        fs_config = {**fs_constructor.generate_optuna_parameters(trial), **fs_base_config}
        forecaster_config = {**forecaster_constructor.generate_optuna_parameters(trial),**forecaster_base_config}

        # feature selection
        fs_time = time.time()  # only indicative, as shared computations are possible across parameter settings
        for fs_instance in fs_instances:
            fs_instance.fit(new_config=fs_config)
        fs_time = (time.time() - fs_time) / len(fs_instances)

        fs_sets = [fs_instance.get_selected_set() for fs_instance in fs_instances]
        fs_sets = [f if target in f else f+[target] for f in fs_sets] # put back target in case it was not selected. We use autoregressive models.
        
        records_fs_metrics = [metrics.evaluate_selection_metrics(fs_set, causal_ground_truth=causal_ground_truth) for fs_set in fs_sets]
        avg_records_fs_metrics = sum_cv_metrics(records_fs_metrics)
        avg_records_fs_metrics["fs_time"] = fs_time

        # forecasting
        metric_dict = metrics.get_forecasting_metrics()
        records_es_metrics, _, _ = cross_validation_single(forecaster_constructor, forecaster_config, target, "trial"+str(trial.number), metric_dict, tuning_df, fs_sets=fs_sets, **cv_generator_kwargs)
        avg_records_es_metrics = sum_cv_metrics(records_es_metrics)
        
        # metrics record formating
        row_results = {**avg_records_fs_metrics,
                       **avg_records_es_metrics}
        row_results["trial"] = trial.number
        row_results["experiment_timestamp"] = experiment_timestamp
        
        row_params = {**flatten_dict(fs_config, "feature_selector"), **flatten_dict(forecaster_config, "forecaster"),
                       "trial":trial.number, "experiment_timestamp": experiment_timestamp}
        memorize["params"].append(row_params)
        memorize["results"].append(row_results)
        
        return row_results[objective]
        
    return optuna_objective_function, memorize



def tune_single_solution(
        tuning_data: pd.DataFrame,
        data_name,
        fs_name,
        forecaster_name,
        fs_constructor,
        es_constructor,
        fs_config,
        es_config,
        target, 
        causal_ground_truth:Optional[List[List[str]]] = None,
        objective="r2",
        seed=0,
        first_phase_trials:Optional[int]=80,
        second_phase_trials:Optional[int]=20,
        time_limit:Optional[int] = None,
        **cv_generator_kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    space = {**fs_constructor.generate_optuna_parameters_grid(),**es_constructor.generate_optuna_parameters_grid()}

    sampler_phase_1 = optuna.samplers.TPESampler(seed=seed)
    sampler_phase_2 = optuna.samplers.RandomSampler(seed=seed)
    num_trials_phase_1 = first_phase_trials
    num_trials_phase_2 = second_phase_trials

    # launching optuna with TPE sampler for fast estimation of many hyperparameters
    objective, records = generate_optuna_objective_function_single_solution(fs_constructor, fs_config, es_constructor, es_config, tuning_data, target, causal_ground_truth=causal_ground_truth, objective="r2", **cv_generator_kwargs)
    study = optuna.create_study(sampler=sampler_phase_1, direction="maximize")
    study.optimize(objective, n_trials=num_trials_phase_1, timeout=time_limit)

    # # launching optuna with random sampler ONLY FOR THE FEATURE SELECTION to refine the parameters
    # best_params = study.best_params
    # fixed_params = dict([(x,best_params[x]) for x in best_params if x.split(".")[0]=="forecaster"])
    # partial_sampler = optuna.samplers.PartialFixedSampler(fixed_params, sampler_phase_2)
    # study.sampler = partial_sampler
    # study.optimize(objective, n_trials=num_trials_phase_2, timeout=time_limit)

    # save results
    df_params = pd.DataFrame.from_records(records["params"])
    df_params["forecaster_name"] = forecaster_name
    df_params["fs_name"] = fs_name
    df_params["data_name"] = data_name
    df_results = pd.DataFrame.from_records(records["results"])

    return df_params, df_results






def get_best_configuration(path, data_name, fs_name, fs_config, forecaster_name:Optional[str]=None, es_config=dict()):
    df_params = pd.read_csv(path+"_params.csv", compression="gzip")
    df_results = pd.read_csv(path+"_results.csv", compression="gzip")
    df_all = pd.merge(df_params, df_results,on=["trial","experiment_timestamp"])

    if forecaster_name is not None:
        idx = df_all.groupby(["fs_name","forecaster_name","data_name"])['r2'].transform(max) == df_all['r2']
        df_best = df_all[idx].groupby(["fs_name","forecaster_name","data_name"]).first().reset_index()
        df_best = df_best[df_best["fs_name"]==fs_name]
        df_best = df_best[df_best["forecaster_name"]==forecaster_name]
        df_best = df_best[df_best["data_name"]==data_name]
    else:
        idx = df_all.groupby(["fs_name","data_name"])['r2'].transform(max) == df_all['r2']
        df_best = df_all[idx].groupby(["fs_name","data_name"]).first().reset_index()
        df_best = df_best[df_best["fs_name"]==fs_name]
        df_best = df_best[df_best["data_name"]==data_name]

    df_best = df_best.convert_dtypes()

    previous_best_configuration = df_best.to_dict()
    idx = df_best.index[0]

    # feature selector
    new_parameters = dict()
    for c in previous_best_configuration:
        if c.split(".")[0]=="feature_selector":
            new_parameters[".".join(c.split(".")[1:])] = previous_best_configuration[c][idx]
    best_fs_config = {**new_parameters,**fs_config}

    # forecaster
    new_parameters = dict()
    for c in previous_best_configuration:
        if c.split(".")[0]=="forecaster":
            new_parameters[".".join(c.split(".")[1:])] = previous_best_configuration[c][idx]
    best_es_config = {**new_parameters,**es_config}
    if forecaster_name is None:
        forecaster_name = previous_best_configuration["forecaster_name"][idx]

    return best_fs_config, forecaster_name, best_es_config





def save_tuning_results(path, df_params, df_results):
    if not os.path.isfile(path+"_params.csv"):
        all_records_params = pd.DataFrame()
        all_records_results = pd.DataFrame()
    else:
        all_records_params = pd.read_csv(path+"_params.csv", compression="gzip")
        all_records_results = pd.read_csv(path+"_results.csv", compression="gzip")

    all_records_params = pd.concat([all_records_params, df_params],axis=0)
    all_records_results = pd.concat([all_records_results, df_results],axis=0)

    all_records_params.to_csv(path+"_params.csv", index=False, compression="gzip")
    all_records_results.to_csv(path+"_results.csv", index=False, compression="gzip")

