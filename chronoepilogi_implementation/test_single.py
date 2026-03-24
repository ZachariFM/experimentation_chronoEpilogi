import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
import time


import metrics


from tuning_single import flatten_dict
from tuning_single import smart_transform_fit, smart_transform_apply, smart_transform_inverse
from tuning_single import create_time_related_features
from tuning_single import get_best_configuration
from tuning_single import save_tuning_results as save_testing_results


def forecasting_single(es_constructor, es_config, target, model_name, metric_dict, tuning_df, holdout_df, fs_set=None):
    records = []
    data_train, data_val = tuning_df, holdout_df

    # select TS according to fs_set
    if (fs_set is not None) and isinstance(fs_set, list):
        data_train = data_train[fs_set]
        data_val = data_val[fs_set]

    # generate time features
    data_train_time = create_time_related_features(data_train.index)
    data_val_time = create_time_related_features(data_val.index)

    # transform / scale
    transformers = smart_transform_fit(data_train, data_train.columns)
    data_train_transform = smart_transform_apply(data_train, transformers)
    data_val_transform = smart_transform_apply(data_val, transformers)
    # configure
    es_config = es_constructor.infer_data_shape_parameters(es_config, data_train_transform, data_train_time)
    estimator = es_constructor(es_config, target, model_name=model_name)

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
    row = {"forecaster.training_time":training_time, "forecaster.predict_time":predict_time, **row}
    return row, df_sample_pred, df_sample_true


def create_fs_instance(tuning_df, target, fs_constructor, fs_base_config):
    fs_instances = []
    data_train = tuning_df
    # transform / scale
    transformers = smart_transform_fit(data_train, data_train.columns)
    data_train_transform = smart_transform_apply(data_train, transformers)
    # configure
    fs_instance = fs_constructor(fs_base_config, target, data_train_transform)
    return fs_instance


def test_configuration(fs_constructor, 
                        fs_config, 
                        forecaster_constructor, 
                        forecaster_config, 
                        tuning_df, 
                        holdout_df, 
                        target, 
                        causal_ground_truth:Optional[List[List[str]]] = None):
    # setup selection instance
    fs_instance = create_fs_instance(tuning_df, target, fs_constructor, fs_config)

    # feature selection
    fs_time = time.time()  # only indicative, as shared computations are possible across parameter settings
    fs_instance.fit(new_config=fs_config)
    fs_time = time.time() - fs_time

    fs_set = fs_instance.get_selected_set()
    fs_set = fs_set if target in fs_set else fs_set+[target] # put back target in case it was not selected. We use autoregressive models.
    
    records_fs_metrics = metrics.evaluate_selection_metrics(fs_set, causal_ground_truth=causal_ground_truth)
    records_fs_metrics["fs_time"] = fs_time
    records_fs_metrics["fs_set"] = fs_set

    # forecasting
    metric_dict = metrics.get_forecasting_metrics()
    records_es_metrics, _, _ = forecasting_single(forecaster_constructor, forecaster_config, target, "model", metric_dict, tuning_df, holdout_df, fs_set=fs_set)

    
    # metrics record formating
    row_results = {**records_fs_metrics,
                    **records_es_metrics}
    
    row_params = {**flatten_dict(fs_config, "feature_selector"), **flatten_dict(forecaster_config, "forecaster")}
    
    return [row_params], [row_results]
        



def test_single_solution(
        tuning_results_path: str,
        tuning_data: pd.DataFrame,
        holdout_data:pd.DataFrame,
        data_name,
        fs_name,
        forecaster_name,
        fs_constructor,
        es_constructor,
        fs_config,
        es_config,
        target, 
        causal_ground_truth:Optional[List[List[str]]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # get best configuration
    best_fs_config, _, best_es_config = get_best_configuration(tuning_results_path, data_name, fs_name, fs_config, forecaster_name=forecaster_name, es_config=es_config)

    # launch testing of configuration
    records_params, records_results = test_configuration(fs_constructor, best_fs_config,
                                                         es_constructor, best_es_config,
                                                         tuning_data, holdout_data,
                                                         target, causal_ground_truth)

    # save results - jointure is fine since records have only one row
    df_params = pd.DataFrame.from_records(records_params)
    df_params["forecaster_name"] = forecaster_name
    df_params["fs_name"] = fs_name
    df_params["data_name"] = data_name
    df_results = pd.DataFrame.from_records(records_results)
    df_results["forecaster_name"] = forecaster_name
    df_results["fs_name"] = fs_name
    df_results["data_name"] = data_name

    return df_params, df_results