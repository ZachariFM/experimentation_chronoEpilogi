import numpy as np
import pandas as pd
import deepdiff.diff
from collections import defaultdict

from associations import ModelBasedPartialCorrelation,H2PartialCorrelation,H3PartialCorrelation, HkPartialCorrelation, CrossSectionalH0


from associations import TemporalSlowAssociation, CrossSectionalAssociation
from associations import TemporalSlowHk, CrossSectionalHk

from models import ARDLModel, TemporalAdaptation
from models import OLSCrossSectional, PoissonCrossSectional, LogitCrossSectional

class ChronoEpilogi():

    def __init__(self, 
                 # main parameters
                 data,
                 target,
                 phases="FB",
                 equivalence_early_stopping = True,
                 # other important parameters
                 forward_test_threshold = 0.05,
                 backward_test_threshold = 0.05,
                 equivalence_test_threshold = 0.05,
                 equivalence_correlation_threshold = 0.05,
                 equivalence_heuristic = "parcorr",
                 maximal_selected_size = np.inf,
                 # module settings
                 model_class = None,
                 model_config = None,
                 association_class = None,
                 association_config = None,
                 partial_correlation_class = None,
                 partial_correlation_config = None,
                 # information needed to set default modules
                 start_with_univariate_autoregressive_model = "infer",
                 model_test_method = None,
                 target_type = "continuous",
                 default_k = 1,
                 default_max_lag = 1,
                 variable_types = None,
                 # less important parameters
                 backward_removal_strategy = "first",
                 valid_obs_param_ratio = 0.,
                 ):
        """Initialize ChronoEpilogi.

        Parameters
        ----------
        data: pd.DataFrame
            DataFrame containing the multivariate time series.
            The index of the DataFrame should correspond to timesteps, and the columns to covariates.
            The column index may have one or two levels. See Notes.
        target: str | tuple(str,str)
            The forecasting/regression target, as the name of the column in the provided DataFrame.
            Whether a single string or a tuple of two strings depends on the columns levels in data.
            See Notes.
        phases: str, optional
            Successions of phases to use. Should be one of "F","FB","Fg","FgV","FBG","FBGV","FBE","FBEV".
            F stands for forward phase, B for backward phase, E for equivalence phase, V for verification phase.
            Alternatively, equivalences can be checked during the forward phase with Fg and FgV.
            Recommanded choice should be "FB" for the selection of a single set of TS, and "FBEV" for computing equivalences.
        equivalence_early_stopping: bool, optional
            Set to True to use the Early Stopping heuristic during the equivalence phase.
            This heuristic checks equivalences to a selected TS in decreasing order of correlation with the residuals,
            and skips testing equivalences after the first non-equivalent TS.
            Recommanded choice would be True.
        forward_test_threshold: float, optional
            Threshold of the model difference metric (returned by models.LearningModel.stopping_metric), 
            Value used during the forward phase only.
            A lower threshold corresponds to stricter tests of performance increase, and leads to a smaller selected set after the forward phase.
        backward_test_threshold: float, optional
            Threshold of model equivalence used during the backward phase.
            A lower threshold leads to more removals from the selected set.
        equivalence_test_threshold: float, optional
            Threshold of model equivalence / partial correlation used during the equivalence phase.
            A lower threshold leads to more detected equivalences.
        equivalence_correlation_threshold: float, optional
            Threshold of a correlation test during the equivalence phase when equivalence_heuristic is set to "parcorr".
            A lower threshold leads to less detected equivalences. 
        equivalence_heuristic: str, optional
            Version of the equivalence detection test to use. Should be one of "exact", "resid", "parcorr".
            "exact" uses full models tests to test X equiv(T) Y | Z, leading to high computation times.
            "resid" (heuristic) replaces the above test by X equiv(Residuals(T~Z)) Y.
            It removes the dependency on the size of the selected set, here represented by Z.
            "parcorr" (heuristic) is a composite test attending to a subset of lags individually.
            It removes the dependency on the size of the selected set and on the number of lags (or size of the second column level).
            Recommanded choice would be "parcorr".
        maximal_selected_size: int, optional
            Bounds the maximal number of covariates to include during the forward phase.
            Default value is set to np.inf, so the forward phase only end when model equivalence is reached.

        Modules Parameters
        ------------------
        model_class: None | LearningModel, optional
            The user may provide a custom model inheriting from LearningModel, suited to the specific task and data.
            When left to None, a default model is infered using arguments target_type, start_with_univariate_autoregressive_model, and default_max_lag.
            It is recommanded to pass explicitely a model class and its arguments.
        model_config: None | dict, optional
            Configuration parameter dictionary to pass to the model class.
            If model_class is None, model_config will be infered similarily.
        association_class: None | Association, optional
            The user may provide a custom association inheriting from Association, suited to the specific task and data.
            When left to None, a default association is infered.
            It is recommanded to pass explicitely an association class and its arguments.
        association_config: None | dict, optional
            Configuration parameter dictionary to pass to the association class.
            If association_class is None, association_config will be infered depending on default_max_lag.
        partial_correlation_class: None | PartialCorrelation, optional
            The user may provide a custom association inheriting from PartialCorrelation, suited to the specific task and data.
            When left to None, a default partial correlation is infered.
            It is recommanded to pass explicitely an partial correlation class and its arguments.
        partial_correlation_config: None | dict, optional
            Configuration parameter dictionary to pass to the partial correlation class.
            If partial_correlation_class is None, partial_correlation_config will be infered depending on default_max_lag and default_k.

        Other Parameters
        ----------------
        start_with_univariate_autoregressive_model: str | bool, optional
            Whether the forecasting/regression task should include the past of the forecasted quantity.
            Value can be True for an autoregressive model, False to exclude the past of the forecasted series.
            Value "infer" sets it to True for single-level column index.
            For double-level column index, should be set to "infer" or False, never True.
        model_test_method: None | str, optional
            Parameter passed to the stopping_metric method of the LearningModel.
        target_type: str, optional
            Whether the target is "continuous", "count" or "binary".
            This parameter is only used when model_class is None, to infer the type of model to use.
        default_k: int, optional
            The number of individual lags/second level columns that are compared by the parcorr equivalence heuristic.
            This parameter is only used when partial_correlation_class is None and equivalence_heuristic is "parcorr".
        default_max_lag: int, optional
            The size of the lag window.
            This parameter is only used when the data has a single-level column index, and either model_class, association_class or partial_correlation_class is None.
        variable_types: None | dict[str,str], optional
            For each column (first level column), variable_types[column] is one of "numeric" or "categorical".
            When set to None, or when missing columns compared to the provided data, type "numeric" is used for all missing columns.
            This parameter is only used when either association_class or partial_correlation_class is None.
        backward_removal_strategy: str, optional
            Whether to remove the first found redundant covariate ("first") or the most redundant covariate ("max") during the backward phase.
            Recommanded value is "first".
        valid_obs_param_ratio: float, optional
            Parameter passed to the has_too_many_parameters method of the LearningModel.

        Returns
        -------
        None

        Notes
        -----
        When the input DataFrame has a single level column index, it is assumed that the data is in the prefered form for forecasting.
        The forecasting scenario is to predict time t of the target, given a window of covariates from time t-1 to t-lags.
        The windowing operation is left to the modules (model, association, partial correlation).

        When the input DataFrame has a two level column index, it is assumed that the data is in the prefered form for tabular data regression.
        The regression scenario is to predict time t of the target, given time t of covariates. 
        Hence, each row correspond to a pair (input, output), similarily to tabular data.
        In that situation, columns are grouped according to the first level column index. 
        Hence, ChronoEpilogi selects groups of columns according the the first level column index.
        When a first level column is selected, all corresponding second level columns are included in the new model.
        The role of the second level columns is similar to the role of the lag window for single level data.
        The "parcorr" heuristic (see parameter equivalence_heuristic) attends to a subset of those second level columns instead of lags.

        In fact, single-level DataFrame can be transformed to double-level DataFrame.
        It suffices to create a column for each lag of each covariate, 
        labeling lag on the second level index and original column as first level index.
        """
        self.target = target
        self.data = data
        self.phases = phases
        self.equivalence_early_stopping = equivalence_early_stopping
        self.target_type = target_type
        self.forward_test_threshold = forward_test_threshold
        self.backward_test_threshold = backward_test_threshold
        self.equivalence_test_threshold = equivalence_test_threshold
        self.equivalence_correlation_threshold = equivalence_correlation_threshold
        self.equivalence_heuristic = equivalence_heuristic
        self.maximal_selected_size = maximal_selected_size
        self.valid_obs_param_ratio = valid_obs_param_ratio
        self.start_with_univariate_autoregressive_model = start_with_univariate_autoregressive_model
        self.backward_removal_strategy = backward_removal_strategy
        self.model_test_method = model_test_method
        self.default_k = default_k
        self.default_max_lag = default_max_lag
        self.variable_types = variable_types

        self.model_class = model_class
        self.model_config = model_config
        self.association_class = association_class
        self.association_config = association_config
        self.partial_correlation_class = partial_correlation_class
        self.partial_correlation_config = partial_correlation_config

        self._check_config()

        self._prebuild_objects(model_class, model_config, 
                               association_class, association_config, 
                               partial_correlation_class, partial_correlation_config)
        
        # storing runs
        self.selected_set = None
        self.equivalent_variables = None
        self.computed_full_tests = dict()
        self.computed_Hk_partial_tests = dict()
        self.computed_resid_tests = dict()
        self.computed_residuals = dict()
        self.computed_associations = dict()
        self.computed_models = dict()


        
        
    def _check_config(self):
        if self.phases not in ["F","FB","Fg","FgV","FBG","FBGV","FBE","FBEV"]:
            raise ValueError("Invalid value for argument phases.")
        if not isinstance(self.equivalence_early_stopping,bool):
            raise TypeError("Expected boolean for argument equivalence_early_stopping.")
        if self.target_type not in ["continuous", "binary", "count"]:
            raise ValueError("Argument target_type expects either 'continuous', 'binary', 'count'.")
        if self.equivalence_heuristic not in ["parcorr", "resid", "exact"]:
            raise ValueError("Argument equivalence_heuristic expects either 'parcorr', 'resid' or 'exact'.")
        if self.start_with_univariate_autoregressive_model not in ["infer", True, False]:
            raise ValueError("Expected string 'infer' or boolean for argument start_with_univariate_autoregressive_model.")
        if self.backward_removal_strategy not in ["first","max"]:
            raise ValueError("Argument backward_removal_strategy expects either 'first' or 'max'.")
        
        # level of the dataframe columns
        if not isinstance(self.data.columns, pd.MultiIndex) or self.data.columns.nlevels==1:
            self.data_format_is_level_1 = True
        else:
            self.data_format_is_level_1 = False

        # autoregressive model only if the DataFrame column index has one level.
        if self.start_with_univariate_autoregressive_model == "infer":
            if self.data_format_is_level_1:
                self.start_with_univariate_autoregressive_model = True
            else:
                self.start_with_univariate_autoregressive_model = False
        
        # check if the model test is set. If model_class is also None, set to "lr-test".
        if self.model_test_method is None:
            if self.model_class is None:
                self.model_test_method = "lr-test"

        # in the case of either association_class or partial_correlation_class being None,
        # we must create or complete a variable_types dictionary
        # variables of unknown types will be considered numerical
        if (self.association_class is None) or (self.partial_correlation_class is None):
            if self.variable_types is None:
                self.variable_types = defaultdict(lambda:"numerical")
            else:
                for variable in self.data.columns:  # compatible with MultiLevel.
                    if variable not in self.variable_types:
                        self.variable_types[variable] = "numerical"


    def _prebuild_objects(self, model_class, model_config,
                          association_class, association_config,
                          partial_correlation_class, partial_correlation_config):
        if model_class is None:
            # if 1level, autoregressive, continuous: take ARDLModel
            # if 1level, autoregressive, count: take TemporalAdaptation with PoissonCrossSectional
            # if 1level, autoregressive, binary: take TemporalAdaptation with LogitCrossSectional
            # if 1level, non-autoregressive, continuous: take TemporalAdaptation with OLSCrossSectional and non-autoregressive
            # if 1level, non-autoregressive, count: take TemporalAdaptation with PoissonCrossSectional and non-autoregressive
            # if 1level, non-autoregressive, binary: take TemporalAdaptation with LogitCrossSectional and non-autoregressive
            # if 2level, continuous target: take OLSCrossSectional
            # if 2level, count target: take PoissonCrossSectional
            # if 2level, binary target: take LogitCrossSectional
            if self.data_format_is_level_1:
                if self.target_type=='continuous' and self.start_with_univariate_autoregressive_model:
                        self.model_class = ARDLModel
                        self.model_config = {"constructor":{"order":self.default_max_lag,
                                                            "lags":self.default_max_lag,
                                                            "trend":"c","causal":True},
                                            "fit":{"cov_type":"HC0"}}
                else:
                    self.model_class = TemporalAdaptation
                    base_config = {"lags":self.default_max_lag,"residuals":"raw",
                                   "constructor":{},
                                   "fit":{"disp":0}}
                    if self.target_type=='continuous':
                        base_config = {**base_config, "model_type":"OLSCrossSectional", "fit":{}}
                    elif self.target_type=="count":
                        base_config = {**base_config, "model_type":"PoissonCrossSectional"}
                    else:
                        base_config = {**base_config, "model_type":"LogitCrossSectional"}
                    if self.start_with_univariate_autoregressive_model:
                        base_config = {**base_config,"autoregressive":True}
                    else:
                        base_config = {**base_config,"autoregressive":False}
                    self.model_config = base_config
            else:
                base_config = {"residuals":"raw",
                                "constructor":{}, "fit":{"disp":0}}
                if self.target_type=='continuous':
                    self.model_class = OLSCrossSectional
                    self.model_config = {**base_config, "fit":{}}
                elif self.target_type=='count':
                    self.model_class = PoissonCrossSectional
                    self.model_config = base_config
                else:
                    self.model_class = LogitCrossSectional
                    self.model_config = base_config
        else:
            self.model_class = model_class
            self.model_config = model_config
        
        if association_class is None:
            # if 1level column, TemporalSlowAssociation
            # if 2level column, CrossSectionalAssociation
            if self.data_format_is_level_1:
                self.association_class = TemporalSlowAssociation
                self.association_config = {"lags":self.default_max_lag,"categorical_method":"f_oneway",
                                           "variable_types":self.variable_types}
            else:
                self.association_class = CrossSectionalAssociation
                self.association_config = {"categorical_method":"f_oneway","variable_types":self.variable_types}
        else:
            self.association_class = association_class
            self.association_config = association_config
        self.association_object = self.association_class(self.association_config)

        if partial_correlation_class is None:
            # if 1level column, TemporalSlowHk
            # if 2level column, CrossSectionalHk
            if self.data_format_is_level_1:
                self.partial_correlation_class = TemporalSlowHk
                self.partial_correlation_config = {"lags":self.default_max_lag,"categorical_method":"f_oneway",
                                           "variable_types":self.variable_types,"k":self.default_k}
            else:
                self.partial_correlation_class = CrossSectionalHk
                self.partial_correlation_config = {"categorical_method":"f_oneway",
                                           "variable_types":self.variable_types,"k":self.default_k}
        else:
            self.partial_correlation_class = partial_correlation_class
            self.partial_correlation_config = partial_correlation_config
        self.partial_correlation_object = self.partial_correlation_class(self.partial_correlation_config)
        
        # choose H0 object
        if self.data_format_is_level_1:
            self.H0_partial_correlation_class = ModelBasedPartialCorrelation
            self.H0_partial_correlation_config = {"lags":self.default_max_lag,"large_sample":False}
        else:
            self.H0_partial_correlation_class = CrossSectionalH0
            self.H0_partial_correlation_config = {"large_sample":False}

        
    def _reset_data(self, data):
        """
        Resets learned structures depending on dataset changes

        Currently, resets everything
        """
        self.selected_set = None
        self.equivalent_variables = None
        self.computed_full_tests = dict()
        self.computed_Hk_partial_tests = dict()
        self.computed_resid_tests = dict()
        self.computed_residuals = dict()
        self.computed_associations = dict()
        self.computed_models = dict()
        
        self.data = data

        #!TODO add adaptative change to keep analysis where evidence of no change.
    
    def _reset_config(self, config):
        # !TODO: verify all parameters. Currenly, change is only implemented for some of them.
        self.equivalent_variables = None

        # Note that equivalent threshold, correlation threshold, as well as early stopping and equivalence heuristic only need
        # resetting self.equivalent_variables, which is done systematically.
        self.equivalence_test_threshold = config.get("equivalence_test_threshold", self.equivalence_test_threshold)
        self.equivalence_correlation_threshold = config.get("equivalence_correlation_threshold",self.equivalence_correlation_threshold)
        self.equivalence_heuristic = config.get("equivalence_heuristic",self.equivalence_heuristic)
        self.equivalence_early_stopping = config.get("equivalence_early_stopping",self.equivalence_early_stopping)

        # default_k influences the parcorr partial test and must remove the computed tests.
        if "default_k" in config and self.default_k != config["default_k"]:
            self.computed_Hk_partial_tests = dict()
            self.default_k = config["default_k"]
        
        # phases, selected size, forward test and backward test thresholds influence the selected set.
        if ("phases" in config and self.phases != config["phases"]) or "g" in self.phases:  # just to make sure, in case of interweaved forward equivalence, reset the selected set
            self.selected_set = None
            self.phases = config["phases"]
        if "forward_test_threshold" in config and self.forward_test_threshold != config["forward_test_threshold"]:
            self.selected_set = None
            self.forward_test_threshold = config["forward_test_threshold"]
        if "backward_test_threshold" in config and self.backward_test_threshold != config["backward_test_threshold"]:
            self.selected_set = None
            self.backward_test_threshold = config["backward_test_threshold"]
        if "maximal_selected_size" in config and self.maximal_selected_size != config["maximal_selected_size"]:
            self.selected_set = None
            self.maximal_selected_size = config["maximal_selected_size"]

    
    def _make_key_full_tests(self,tested,full):
        """
        tested indep target | full \\setminus tested
        """
        key = (str(tested), str(sorted(full)))
        return key
    
    def _make_key_resid_tests(self,tested1,tested2,condition):
        """
        tested1 indep resid | tested2, with resid = model(target~condition)
        """
        key = (str(tested1), str(tested2), str(sorted(condition)))
        return key
        
    def _make_key_partial_tests(self,tested1,tested2,condition):
        """
        tested1 corr resid, with resid = model(target~condition)
        tested2 corr resid, with resid = model(target~condition)
        tested1 corr resid | tested2, with resid = model(target~condition)
        tested2 corr resid | tested1, with resid = model(target~condition)
        """
        key = (str(sorted([tested1,tested2])), str(sorted(condition)))
        return key
    
    def _make_key_residuals(self,condition):
        """
        storing residuals computed with model(target~condition)
        """
        key = str(sorted(condition))
        return key
    
    def _make_key_associations(self,condition):
        """
        storing associations computed with the residuals of model(target~condition)
        """
        key = str(sorted(condition))
        return key
        
    def _make_key_models(self,condition):
        """
        storing models computed as model(target~condition)
        """
        key = str(sorted(condition))
        return key
    
    def _train_model(self,variables, memorize=False):
        """
        Train a model using the columns in variables as input.
        :param memorize: if set to true, keep model object (potentially of large size) into memory
        """
        if self.model_class is None:
            raise(RuntimeError("self.model_class is None. This should not happen if the class was correctly initialized."))

        target = self.target if self.data_format_is_level_1 else self.target[0]
        if target not in variables:
            variables = list(variables)+[target]
        
        key = self._make_key_models(variables)
        if key not in self.computed_models:
            current_model = self.model_class(self.model_config, target=self.target)
            current_model.fit(self.data[variables])
            if memorize:
                self.computed_models[key] = current_model
        else:
            current_model = self.computed_models[key]

        return current_model
    
    def _compute_memorize_associations(self,condition,residuals,candidate_variables):
        mem_dict = dict()
        
        key = self._make_key_associations(condition)
        if key in self.computed_associations:
            mem_dict = self.computed_associations[key]
        else:
            self.computed_associations[key]=dict()
            
        remaining_candidates = [candidate for candidate in candidate_variables if candidate not in mem_dict]
        if len(remaining_candidates)>0:
            measured_associations = self.association_object.association(residuals, self.data[remaining_candidates])
            for i,candidate in enumerate(remaining_candidates):
                self.computed_associations[key][candidate] = measured_associations[i]
        
        measured_associations = [self.computed_associations[key][candidate] for candidate in candidate_variables]
        return np.array(measured_associations)
    
    #####
    #
    #   Forward phase utils
    #
    #####
    
    def _stopping_criterion(self, current_model, previous_model, len_selected_features):
        """
        return True if we should continue to include variables, False to stop
        """ 
        # note: checking that there are still TS to select in the candidate set is done in _forward.
        # if enough features were selected, stop
        if len_selected_features > self.maximal_selected_size:
            return False
        # if the number of observations is too low compared to the number of parameters, stop
        if current_model.has_too_many_parameters(self.valid_obs_param_ratio):
            return False
        # if this is the first iteration, continue
        if previous_model is None:
            return True
        
        threshold = self.forward_test_threshold
        metric = current_model.stopping_metric(previous_model, self.model_test_method)
        return metric < threshold
        
        
        
        
        
    def _initialize_forward(self):
        """
        Compute the starting structures of the forward phase
        """
        if self.start_with_univariate_autoregressive_model:
            initial_selected = [self.target]
        else:
            initial_selected = []

        previous_model = None  # will be defined during the first iteration
        
        # initialize the current model
        current_model = self._train_model(initial_selected, memorize=True)
        residuals = current_model.residuals()
        key = self._make_key_residuals(initial_selected)
        if key not in self.computed_residuals:
            self.computed_residuals[key] = residuals
        
        # candidate variable list
        candidate_variables = list(self.data.columns.get_level_values(0).unique())
        if not self.start_with_univariate_autoregressive_model:
            if self.data_format_is_level_1:
                candidate_variables.remove(self.target)
            else:
                candidate_variables.remove(self.target[0])
            
        for variable in initial_selected:
            candidate_variables.remove(variable)

        return initial_selected, candidate_variables, previous_model, current_model, residuals
    
    
    #####
    #
    #   Equivalence heuristic
    #
    #####
    
    
    
    
    def _equivalent_test(self, chosen_variable, candidate_variables, condition_variables):
        """
        !TODO: take into account that if we do FBE then C indep T|X1...Xn is redundant to test since MB
        """

        equivalence_heuristic = self.equivalence_heuristic
        equivalence_threshold = self.equivalence_test_threshold

        equivalent_list = []

        key = self._make_key_residuals(condition_variables)
        residuals = self.computed_residuals[key]
        key = self._make_key_residuals(condition_variables + [chosen_variable])
        residuals_full = self.computed_residuals[key]

        if self.equivalence_early_stopping:
            pvalues = - self._compute_memorize_associations(condition_variables,residuals,candidate_variables)
            sort_indexes = np.argsort(pvalues)
            candidate_variables = np.array(list(candidate_variables))[sort_indexes]

        for candidate_count, candidate in enumerate(candidate_variables):
            if self.equivalence_early_stopping and len(equivalent_list) != candidate_count:
                break

        
            if equivalence_heuristic == "exact":
        
                equivalence_method = self.model_test_method
                conditioning_set = condition_variables
                key = self._make_key_full_tests(chosen_variable,conditioning_set + [candidate, chosen_variable])
                if key not in self.computed_full_tests:
                    restricted_model = self._train_model(conditioning_set + [candidate])
                    full_model = self._train_model(conditioning_set + [candidate, chosen_variable])
                    pvalue = full_model.stopping_metric(restricted_model, equivalence_method) # low p-value indicates models are different
                    self.computed_full_tests[key] = pvalue
                pvalue = self.computed_full_tests[key]
                
                if pvalue > equivalence_threshold: # tested that chosen variable does not have more info than candidate
                    key = self._make_key_full_tests(candidate,conditioning_set + [candidate, chosen_variable])
                    if key not in self.computed_full_tests:
                        restricted_model = self._train_model(conditioning_set + [chosen_variable])
                        full_model = self._train_model(conditioning_set + [candidate, chosen_variable])
                        pvalue = full_model.stopping_metric(restricted_model, equivalence_method) # low p-value indicates models are different
                        self.computed_full_tests[key] = pvalue
                    pvalue = self.computed_full_tests[key]
                    
                    if pvalue > equivalence_threshold:  # tested that candidate variable does not have more info than chosen
                        equivalent_list.append(candidate)
        
            elif equivalence_heuristic == "resid":
            
                residuals_df = residuals
                chosen_df = self.data[[chosen_variable]]
                candidate_df = self.data[[candidate]]
                
                OLS_object = self.H0_partial_correlation_class(self.H0_partial_correlation_config)
                
                key = self._make_key_resid_tests(chosen_variable,candidate,condition_variables)
                if key not in self.computed_resid_tests:
                    pvalue = OLS_object.partial_corr(residuals_df, chosen_df, candidate_df)
                    self.computed_resid_tests[key] = pvalue
                pvalue = self.computed_resid_tests[key]
                
                if pvalue > equivalence_threshold:  # no relation found between chosen and residuals given candidate
                
                    key = self._make_key_resid_tests(candidate,chosen_variable,condition_variables)
                    if key not in self.computed_resid_tests:
                        pvalue = OLS_object.partial_corr(residuals_df, candidate_df,  chosen_df)
                        self.computed_resid_tests[key] = pvalue
                    pvalue = self.computed_resid_tests[key]
                    
                    if pvalue > equivalence_threshold:  # no relation found between candidate and residuals given chosen
                        equivalent_list.append(candidate)
        

            elif equivalence_heuristic == "parcorr":
                correlation_threshold = self.equivalence_correlation_threshold

                chosen_df = self.data[[chosen_variable]]
                residuals_df = residuals
                candidate_df = self.data[[candidate]]

                # keep in memory the entire vector
                key = self._make_key_partial_tests(chosen_variable, candidate, condition_variables)
                if key not in self.computed_Hk_partial_tests:
                    p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo = self.partial_correlation_object.partial_corr(residuals_df,
                                                                                                    candidate_df, chosen_df)

                    self.computed_Hk_partial_tests[key] = [p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo]
                p_RCa_Co, p_RCo_Ca, p_RCa, p_RCo = self.computed_Hk_partial_tests[key]

                flag = False
                for i in range(len(p_RCa)):
                    if p_RCa[i] < correlation_threshold:
                        for j in range(len(p_RCo)):
                            if p_RCo[j] < correlation_threshold:
                                if p_RCa_Co[i, j] > equivalence_threshold and p_RCo_Ca[j, i] > equivalence_threshold:
                                    equivalent_list.append(candidate)
                                    flag = True
                                    break
                    if flag:
                        break

        return equivalent_list

    
    
    
    
    
    ######
    #
    #     Algorithm phases
    #
    ######
    
    def _forward(self):
        
        selected_features, candidate_variables, previous_model, current_model, residuals = self._initialize_forward()
        self.selected_set = selected_features

        # keep track of the equivalent covariates to each covariate.
        interweaved_equivalence_computation = self.phases in ["Fg", "FgV"]
        if interweaved_equivalence_computation:
            self.equivalent_variables = dict()
        
        significant_model_change = self._stopping_criterion(current_model, previous_model, len(selected_features))
        while significant_model_change:
            if len(candidate_variables)==0:  # verify that we still have candidates
                break
            measured_associations = self._compute_memorize_associations(self.selected_set,residuals,candidate_variables)
            
            chosen_index = np.argmax(measured_associations)
            chosen_variable = candidate_variables[chosen_index]
            
            new_model = self._train_model(self.selected_set+[chosen_variable], memorize=True)
            
            # change/save residuals
            key = self._make_key_residuals(self.selected_set+[chosen_variable])
            if key not in self.computed_residuals:
                self.computed_residuals[key] = new_model.residuals()
            residuals = self.computed_residuals[key]
            
            significant_model_change = self._stopping_criterion(new_model, current_model, len(self.selected_set)+1)
            
            if significant_model_change:
                # compute equivalent set and remove equivalent features
                if interweaved_equivalence_computation: 
                    if self.equivalent_variables is None:  # automated type checks fail here.
                        self.equivalent_variables = dict()

                    equivalence_candidate_variables = list(candidate_variables).copy()
                    equivalence_candidate_variables.remove(chosen_variable)
                
                    self.equivalent_variables[chosen_variable] = self._equivalent_test(chosen_variable, equivalence_candidate_variables, selected_features)
                    for to_remove in self.equivalent_variables[chosen_variable]:
                        candidate_variables.remove(to_remove)
                    self.equivalent_variables[chosen_variable].append(chosen_variable)
                
                ## next iteration preparation
                # put the chosen variable in the selected feature set and remove from forward candidate set
                self.selected_set.append(chosen_variable)
                if chosen_variable in candidate_variables:  # can have been removed at equivalence step
                    candidate_variables.remove(chosen_variable)
                # update the models
                previous_model = current_model
                current_model = new_model
        
    
    def _backward(self):
        """
        Backward pass testing each of the covariate for nonzero coefficient.
        The test is Y_t indep X_t-L...X_t-1 | Z_t-L..Z_t-1 where Z are all other covariates including Y_t-L...Y_t-1
        
        If the config parameter "backward_removal_strategy" is "max", the independent series X of maximal confidence w.r. to a threshold is removed from the selected set.
        If it is "first", the first found covariate is removed.
        
        While there is such a change, keep on conducting backward tests.
        """
        if self.selected_set is None:
            raise(RuntimeError("Backward phase entered but self.selected_set is None. Have you made sure to launch the forward phase?"))

        selected_set_has_changed = True
        while selected_set_has_changed:
            selected_set_has_changed = False  # flag reset
            
            full_model = self._train_model(self.selected_set)
            
            threshold = self.backward_test_threshold
            max_metric = -np.inf
            max_column = None

            for column in self.selected_set:
                # not removing target if autoregressive
                if self.start_with_univariate_autoregressive_model and self.target==column:
                    continue

                # if test is not in memory, compute it
                key = self._make_key_full_tests(column, self.selected_set)
                if key not in self.computed_full_tests:
                    restricted_model = self._train_model([x for x in self.selected_set if x!=column])
                    metric = full_model.stopping_metric(restricted_model, self.model_test_method)
                    self.computed_full_tests[key] = metric
                
                # keep track of the column with maximal p-value for the model comparison test.
                metric = self.computed_full_tests[key]
                if metric>max_metric:
                    max_metric = metric
                    max_column = column
                
                if self.backward_removal_strategy == "first":
                    if max_metric >= threshold:  # there is no significative difference between the models.
                        self.selected_set.remove(max_column)
                        selected_set_has_changed = True  # set change flag to true
                    break
                    
            if self.backward_removal_strategy == "max":
                if max_metric >= threshold:  # there is no significative difference between the models.
                    self.selected_set.remove(max_column)
                    selected_set_has_changed = True  # set change flag to true
    
    def _equivalent_search(self):
        if self.selected_set is None:
            raise(RuntimeError("Equivalence phase entered but self.selected_set is None. Have you made sure to launch the forward phase?"))

        # build candidate set
        candidate_variables = set(self.data.columns.get_level_values(0).unique())
        if not self.data_format_is_level_1:
            candidate_variables.remove(self.target[0])
        for variable in self.selected_set:
            candidate_variables.remove(variable)
        
        self.equivalent_variables = dict()

        #precompute residuals, including full model
        for index in range(len(self.selected_set)+1):
            # create the conditioning set model
            if "G" in self.phases:
                 condition_variables = self.selected_set[:index]
            elif "E" in self.phases:
                condition_variables = self.selected_set[:index]+(self.selected_set[index+1:] if index<len(self.selected_set) else [])

            #!TODO: here add skip if ALL necessary equivalence data is in memory.
            key = self._make_key_residuals(condition_variables)
            if key not in self.computed_residuals:
                current_model = self._train_model(condition_variables)
                self.computed_residuals[key] = current_model.residuals()
            residuals = self.computed_residuals[key]

        for index in range(len(self.selected_set)):
            # if model is autoregressive (data_format is temporal), we cannot test target equivalences
            if self.start_with_univariate_autoregressive_model and self.selected_set[index]==self.target:
                self.equivalent_variables[self.target] = [self.target]
                continue

            # create the conditioning set model
            if "G" in self.phases:
                condition_variables = self.selected_set[:index]
            elif "E" in self.phases:
                condition_variables = self.selected_set[:index]+(self.selected_set[index+1:] if index<len(self.selected_set) else [])

            chosen_variable = self.selected_set[index]

            # compute equivalent set
            if "G" in self.phases:
                self.equivalent_variables[chosen_variable] = self._equivalent_test(chosen_variable, candidate_variables,
                                                                              condition_variables)
            elif "E" in self.phases:
                self.equivalent_variables[chosen_variable] = self._equivalent_test(chosen_variable, candidate_variables,
                                                                              condition_variables)
            # remove equivalent variables
            for to_remove in self.equivalent_variables[chosen_variable]:
                candidate_variables.remove(to_remove)
            self.equivalent_variables[chosen_variable].append(chosen_variable)

    
    
    
    
    def _verify_equivalence_relevance(self):
        """
        Test for each equivalent series X if they are informative given the rest of the series in the MB:
        Given S1....Sn, and X equiv Sn, tests if X indep T | S1....Sn-1.
        If the independence is verified, this test result contradicts the equivalence test of the equivalence phase.
        In this case, remove X from equivalent TS to reduce False Positive.
        """
        threshold = self.backward_test_threshold
        if self.selected_set is None:
            raise(RuntimeError("Verification phase entered but self.selected_set is None. Have you made sure to launch the forward phase?"))
        if self.equivalent_variables is None:
            raise(RuntimeError("Verification phase entered but self.equivalent_variables is None. Have you made sure to launch the equivalence phase?"))
        
        for key in self.equivalent_variables:
        
            covariates = [var for var in self.selected_set if var!=key]
            restricted_model = self._train_model(covariates)
            
            to_remove_list = []
            for candidate in self.equivalent_variables[key]:
                if candidate == key:
                    continue
                full_set_variables = covariates+[candidate]
                
                key_full_test = self._make_key_full_tests(candidate, full_set_variables)
                if key_full_test not in self.computed_full_tests:
                    full_model = self._train_model(full_set_variables)
                    metric = full_model.stopping_metric(restricted_model, self.model_test_method)
                    self.computed_full_tests[key_full_test] = metric
                metric = self.computed_full_tests[key_full_test]
                
                if metric>=threshold:
                    to_remove_list.append(candidate)
            for to_remove in to_remove_list:
                self.equivalent_variables[key].remove(to_remove)
    
    ######
    #
    #     User-callable methods
    #
    ######
    
    def fit(self, data=None, config=None, verbose=False):
        """
        Currently, all parameters are unused.
        Returns: None
        """
        # reset previous fit
        if data is not None:
            self._reset_data(data)
        if config is not None:
            self._reset_config(config)
        
        skip_forward_backward = not (self.selected_set is None)
        
        # forward phase
        if not skip_forward_backward:
            self._forward()

        # backward phase
        if "B" in self.phases and not skip_forward_backward:
            self._backward()
        
        # post-equivalent search
        if "E" in self.phases or "G" in self.phases:
            self._equivalent_search()
        
        # final relevance verification
        if "V" in self.phases:
            self._verify_equivalence_relevance()
        
        
    def get_total_number_sets(self):
        """
        Return the total number of solutions (the product of the size of every equivalence class)
        
        Returns: int
        """
        if self.equivalent_variables is None:
            raise(RuntimeError("self.equivalent_variables is None. Have you made sure to launch the equivalence phase?"))
        
        total = 1
        for key in self.equivalent_variables:
            total*=len(self.equivalent_variables[key])
        return total
        
    def get_solution_from_multiset_index(self,index):
        """
        Return the feature set corresponding to the provided index.
        Each integer index between 0 and self.get_total_number_sets()-1 included correspond to a unique feature set.
         - param index (int)
        
        Returns: list of column names 
        """
        if self.equivalent_variables is None:
            raise(RuntimeError("self.equivalent_variables is None. Have you made sure to launch the equivalence phase?"))
        
        total_number = self.get_total_number_sets()
        n = index%total_number
        denom = total_number
        selected = []
        for key in self.equivalent_variables:
            candidates = self.equivalent_variables[key]
            denom = denom // len(candidates)
            selected.append(candidates[n//denom])
            n = n % denom
        return selected
    
    def get_equivalences_as_list(self):
        if self.equivalent_variables is None:
            raise(RuntimeError("self.equivalent_variables is None. Have you made sure to launch the equivalence phase?"))

        return list(map(lambda x:self.equivalent_variables[x],self.equivalent_variables))

"""
!TODO: unit test suite for the association, parcorr, models, and CE
!TODO: document the fit, get_total_number_sets, get_solution_from_multiset_index

Later on:
!TODO: timeout and verbosity settings for the fit
!TODO: fix config reset protocol, currently inactive
!TODO: simpler functions, unclutter residual and partial correlation saving.
"""