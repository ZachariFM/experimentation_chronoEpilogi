import pandas as pd
import numpy as np
from typing import Optional
from collections import defaultdict
import time

import optuna
from scipy.stats import skew

import metrics

import tuning_single




def sample_equivalent_solutions(fs_instance,target,  number_sols_to_sample, rng):
    # sample equivalent solutions to evaluate with the 
    total_number_of_solutions = fs_instance.get_total_number_sets()
    fs_sets_to_evaluate = []
    fs_set = fs_instance.get_selected_set()
    fs_sets_to_evaluate.append(fs_set)
    if total_number_of_solutions <= number_sols_to_sample:
        sampled_solutions = list(range(total_number_of_solutions))
    else:
        sampled_solutions = rng.integers(0,total_number_of_solutions, size=(number_sols_to_sample,))
    for solution_index in sampled_solutions:
        fs_set = fs_instance.model.get_solution_from_multiset_index(solution_index)
        fs_sets_to_evaluate.append(fs_set)
    
    fs_sets_to_evaluate = [fs_set if target in fs_set else fs_set+[target] for fs_set in fs_sets_to_evaluate]
    
    return fs_sets_to_evaluate


def cross_validation_multiple(es_constructor, es_config, target, model_name, metric_dict, dataset, equivalent_solutions, memorized_solutions_dict=dict(), **cv_generator_kwargs):
    records = []
    for fold_id, data_train, data_val in tuning_single.generator_cross_validation_folds(data=dataset, **cv_generator_kwargs):
        # find if equivalent_solutions are specific to folds
        fold_equivalent_solutions = None
        if isinstance(equivalent_solutions, list) and isinstance(equivalent_solutions[0],list):
            # each fold has its own set of equivalent solutions
            fold_equivalent_solutions = equivalent_solutions[fold_id]
        elif isinstance(equivalent_solutions, list):
            # the equivalent solutions are identically applied to all folds
            fold_equivalent_solutions = equivalent_solutions
        else:
            raise Exception("Expected equivalent solutions to be confirmed")
        
        fold_results = []
        for fs_set in fold_equivalent_solutions:
            # have we already trained the forecaster on this fold?
            memorize_key = (fold_id, str(sorted(fs_set)))
            if memorized_solutions_dict is not None and memorize_key in memorized_solutions_dict:
                records_metrics = memorized_solutions_dict[memorize_key]
            else:
                data_train2 = data_train[fs_set]
                data_val2 = data_val[fs_set]

                # generate time features
                data_train_time = tuning_single.create_time_related_features(data_train2.index)
                data_val_time = tuning_single.create_time_related_features(data_val2.index)

                # transform / scale
                transformers = tuning_single.smart_transform_fit(data_train2, data_train2.columns)
                data_train_transform = tuning_single.smart_transform_apply(data_train2, transformers)
                data_val_transform = tuning_single.smart_transform_apply(data_val2, transformers)
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
                    df_sample_pred = tuning_single.smart_transform_inverse(df_sample_pred, transformers)
                    df_sample_true = tuning_single.smart_transform_inverse(df_sample_true, transformers)
                    total_pred.append(df_sample_pred)
                    total_true.append(df_sample_true)
                df_sample_pred = pd.concat(total_pred).reset_index(drop=True)  # reset index is here since we do not have the time index
                df_sample_true = pd.concat(total_true).reset_index(drop=True)  # reset index is here since we do not have the time index

                records_metrics = dict()
                for metric in metric_dict:
                    if metric!="mase":
                        records_metrics[metric] = metric_dict[metric](df_sample_true, df_sample_pred)
                    else:
                        records_metrics[metric] = metric_dict[metric](df_sample_true, df_sample_pred, data_train2[target])

                records_metrics = {"fold":fold_id, "forecaster.training_time":training_time, "forecaster.predict_time":predict_time, **records_metrics}
                memorized_solutions_dict[memorize_key] = records_metrics
            
            fold_results.append(records_metrics)
        
        # get average performance across solutions
        records.append({**tuning_single.sum_cv_metrics(fold_results),"fold":fold_id})
    return records



def generate_optuna_objective_function_multiple_solutions(
        fs_constructor,
        fs_base_config,  # parameters that cannot be infered from the data or randomly chosen
        forecaster_constructor,
        forecaster_base_config,   # parameters that cannot be infered from the data or randomly chosen
        tuning_df,
        target, 
        causal_ground_truth = None,
        objective="r2",
        number_sols_to_sample = 50,
        seed=0,
        **cv_generator_kwargs):
    
    memorize = defaultdict(list)  # object to put all configurations for simple retrieval.
    
    # define once for all the feature selection instance, to allow for fast parameter tuning when only significance thresholds are changing.
    # due to cross-validation, we must define one instance per fold
    fs_instances = tuning_single.create_fs_instances_for_cv(tuning_df, target, fs_constructor, fs_base_config, **cv_generator_kwargs)
    # unique experiment identifier with start time
    experiment_timestamp = time.time()
    rng = np.random.default_rng(seed)

    memorize_feature_set_performances = dict()  # since forecaster parameters are fixed, set performance is independent of configuration. We can memorize identical sets metrics between configurations.


    def optuna_objective_function(trial):
        # get parameters
        fs_config = {**fs_constructor.generate_optuna_parameters(trial), **fs_base_config}
        forecaster_config = forecaster_base_config

        # feature selection
        fs_time = time.time()  # only indicative, as shared computations are possible across parameter settings
        for fs_instance in fs_instances:
            fs_instance.fit(new_config=fs_config)
        fs_time = (time.time() - fs_time) / len(fs_instances)

        # format sets
        fs_sets = [fs_instance.get_selected_set() for fs_instance in fs_instances]
        fs_sets = [f if target in f else f+[target] for f in fs_sets] # put back target in case it was not selected. We use autoregressive models.
        equivalence_classes = fs_instance.get_equivalence_classes_as_list()

        records_fs_metrics = [metrics.evaluate_selection_metrics(fs_set, equivalence_classes=equivalence_classes, causal_ground_truth=causal_ground_truth) for fs_set in fs_sets]
        avg_records_fs_metrics = tuning_single.sum_cv_metrics(records_fs_metrics)
        avg_records_fs_metrics["fs_time"] = fs_time

        # get multiple solutions:
        equivalent_solutions = [sample_equivalent_solutions(fs_instance, target, number_sols_to_sample, rng) for fs_instance in fs_instances]

        # forecasting
        metric_dict = metrics.get_forecasting_metrics()
        records_es_metrics = cross_validation_multiple(forecaster_constructor, forecaster_config, target, "trial"+str(trial.number), metric_dict, tuning_df, equivalent_solutions, memorized_solutions_dict=memorize_feature_set_performances, **cv_generator_kwargs)
        avg_records_es_metrics = tuning_single.sum_cv_metrics(records_es_metrics)
        
        # metrics record formating
        row_results = {**avg_records_fs_metrics,
                       **avg_records_es_metrics}
        row_results["trial"] = trial.number
        row_results["experiment_timestamp"] = experiment_timestamp
        
        row_params = {**tuning_single.flatten_dict(fs_config, "feature_selector"), **tuning_single.flatten_dict(forecaster_config, "forecaster"),
                       "trial":trial.number, "experiment_timestamp": experiment_timestamp}
        memorize["params"].append(row_params)
        memorize["results"].append(row_results)
        
        # maximization objective
        performance_score = row_results[objective]
        size_score = row_results["total size"]
        return performance_score, size_score
        
    return optuna_objective_function, memorize







def tune_multiple_solutions(
        tuning_data: pd.DataFrame,
        data_name,
        fs_name,
        forecaster_name,
        fs_constructor,
        es_constructor,
        fs_config,
        es_config,
        target,
        causal_ground_truth=None,
        number_sols_to_sample=50,
        seed=0,
        num_trials = None,
        time_limit:Optional[int] = None,
        **cv_generator_kwargs):
    

    space = {**fs_constructor.generate_optuna_parameters_grid(),**es_constructor.generate_optuna_parameters_grid()}
    sampler = optuna.samplers.TPESampler(seed=seed)

    # launching optuna with the appropriate hp sampler
    # multiobjective tuning: we want the larger number of equivalences without decreasing forecasting performance
    objective, records = generate_optuna_objective_function_multiple_solutions(fs_constructor, fs_config, es_constructor, es_config, tuning_data, target, causal_ground_truth=causal_ground_truth, number_sols_to_sample=number_sols_to_sample, objective="r2", **cv_generator_kwargs)
    study = optuna.create_study(sampler=sampler, directions=["maximize","maximize"])  # we maximize r2 and number of selected features to get the most equivalent features.
    study.optimize(objective, n_trials=num_trials, timeout=time_limit)

    # save results
    df_params = pd.DataFrame.from_records(records["params"])
    df_params["forecaster_name"] = forecaster_name
    df_params["fs_name"] = fs_name
    df_params["data_name"] = data_name
    df_results = pd.DataFrame.from_records(records["results"])
    df_results["forecaster_name"] = forecaster_name
    df_results["fs_name"] = fs_name
    df_results["data_name"] = data_name

    return df_params, df_results
